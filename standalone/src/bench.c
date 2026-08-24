// Standalone benchmark / correctness harness for the ANUGA shallow-water
// OpenMP-offload kernels.
//
// Links the production kernel sources (anuga/shallow_water/gpu/*.c) directly:
// no meson, no Cython, no Python, no MPI.  The timestep taken here is the same
// gpu_evolve_one_rk2_step() the Python mode-2 ('unified') path calls.
//
//   ./bin/bench_gpu --nx 400 --ny 400 --steps 100 --phases
//   ./bin/bench_cpu --nx 200 --ny 200 --steps 50 --save golden.bin
//   ./bin/bench_gpu --nx 200 --ny 200 --steps 50 --check golden.bin

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_domain.h"
#include "mesh.h"
#include "setup.h"
#include "snapshot.h"

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

typedef struct {
    int64_t nx, ny;
    int64_t steps;
    int64_t warmup;
    int     repeat;
    int     phases;
    int     verbose;
    int     apply_forcing;
    const char *save_path;
    const char *check_path;
    const char *csv_path;
    int     morton;
    double  rtol, atol;
} bench_opts;

static void usage(const char *argv0) {
    printf(
"usage: %s [options]\n"
"\n"
"  mesh / problem\n"
"    --nx N            cells in x                       (default 200)\n"
"    --ny N            cells in y                       (default 200)\n"
"                      the mesh is a rectangular cross: 4*nx*ny triangles\n"
"    --lenx L          domain width  in metres          (default 1000)\n"
"    --leny L          domain height in metres          (default 1000)\n"
"    --case NAME       dam | dambumps | lake            (default dam)\n"
"                        dam       flat bed, wet dam break (every cell wet)\n"
"                        dambumps  bumpy bed dam break (wet/dry branches)\n"
"                        lake      water at rest over bumps (well-balanced)\n"
"    --manning V       Manning n                        (default 0.03)\n"
"    --water V         still-water / downstream stage   (default 5)\n"
"    --dam V           upstream stage                   (default 10)\n"
"    --no-friction     skip the Manning forcing term\n"
"    --order NAME      row | morton -- element numbering  (default row)\n"

"                        morton renumbers triangles along a Z-order curve so\n"
"                        both grid directions' neighbours stay cache-near;\n"
"                        snapshots stay in canonical order either way\n"
"\n"
"  run\n"
"    --steps N         timed RK2 steps                  (default 100)\n"
"    --warmup N        untimed RK2 steps first          (default 5)\n"
"    --repeat N        repeat the timed loop N times, report the best\n"
"    --cfl V           CFL number                       (default 1.0)\n"
"    --phases          per-kernel timing breakdown\n"
"    --verbose         let the kernels print their own setup messages\n"
"\n"
"  correctness\n"
"    --save FILE       write the final centroid state to FILE\n"
"    --check FILE      compare the final centroid state against FILE\n"
"    --atol V          absolute tolerance for --check    (default 1e-10)\n"
"    --rtol V          relative tolerance for --check    (default 1e-8)\n"
"\n"
"  reporting\n"
"    --csv FILE        append one machine-readable result row to FILE\n"
"                      (writes the header if FILE does not exist yet)\n"
"\n", argv0);
}

static int64_t arg_i(int argc, char **argv, int *i, const char *name) {
    if (++(*i) >= argc) { fprintf(stderr, "bench: %s needs a value\n", name); exit(2); }
    return strtoll(argv[*i], NULL, 10);
}

static double arg_d(int argc, char **argv, int *i, const char *name) {
    if (++(*i) >= argc) { fprintf(stderr, "bench: %s needs a value\n", name); exit(2); }
    return strtod(argv[*i], NULL);
}

static const char *arg_s(int argc, char **argv, int *i, const char *name) {
    if (++(*i) >= argc) { fprintf(stderr, "bench: %s needs a value\n", name); exit(2); }
    return argv[*i];
}

// ---------------------------------------------------------------------------
// Timed RK2 step
//
// Mirrors gpu_evolve_one_rk2_step() in gpu_kernels.c exactly (single process,
// no fixed timestep), with a timer around each kernel.  Keep the two in sync:
// --phases and the plain loop must produce identical state.
// ---------------------------------------------------------------------------

enum {
    PH_PREPARE = 0, PH_EXTRAPOLATE, PH_BOUNDARY,
    PH_FLUXES, PH_FORCING_UPDATE, PH_NPHASES
};

static const char *phase_names[PH_NPHASES] = {
    "prepare", "extrapolate", "boundary",
    "compute_fluxes", "forcing+update"
};

static double phase_time[PH_NPHASES];

#define TIME_PHASE(id, call) do {                 \
        const double _t0 = omp_get_wtime();       \
        call;                                     \
        phase_time[id] += omp_get_wtime() - _t0;  \
    } while (0)

static void evaluate_boundaries(struct gpu_domain *GD) {
    // The benchmark only sets up reflective edges, but call the full set so
    // the timing matches what the production step does.
    gpu_evaluate_reflective_boundary(GD);
    gpu_evaluate_dirichlet_boundary(GD);
    gpu_evaluate_transmissive_boundary(GD);
    gpu_evaluate_transmissive_n_zero_t_boundary(GD);
    gpu_evaluate_time_boundary(GD);
    gpu_evaluate_file_boundary(GD);
    gpu_evaluate_absorbing_wave_boundary(GD);
    gpu_evaluate_characteristic_wave_boundary(GD);
    gpu_evaluate_flather_boundary(GD);
}

static double rk2_step_timed(struct gpu_domain *GD, double max_timestep, int apply_forcing) {
    double timestep;

    // ---- first Euler stage
    // prepare = fused RK2 backup + protect + extrapolate centroid pass
    TIME_PHASE(PH_PREPARE,     gpu_prepare_step(GD, 1));
    TIME_PHASE(PH_EXTRAPOLATE, gpu_extrapolate_edges(GD));
    TIME_PHASE(PH_BOUNDARY,    evaluate_boundaries(GD));

    double local_timestep;
    TIME_PHASE(PH_FLUXES, local_timestep = gpu_compute_fluxes(GD, 0, 2));

    timestep = GD->CFL * local_timestep;
    GD->recorded_flux_timestep =
        (timestep < GD->evolve_max_timestep) ? timestep : GD->evolve_max_timestep;
    if (timestep > max_timestep) timestep = max_timestep;

    TIME_PHASE(PH_FORCING_UPDATE,
               gpu_forcing_and_update(GD, timestep, apply_forcing, 0, 0.0, 0.0));

    // ---- second Euler stage
    TIME_PHASE(PH_PREPARE,     gpu_prepare_step(GD, 0));
    TIME_PHASE(PH_EXTRAPOLATE, gpu_extrapolate_edges(GD));
    TIME_PHASE(PH_BOUNDARY,    evaluate_boundaries(GD));
    TIME_PHASE(PH_FLUXES,      gpu_compute_fluxes(GD, 1, 2));

    TIME_PHASE(PH_FORCING_UPDATE,
               gpu_forcing_and_update(GD, timestep, apply_forcing, 1, 0.5, 0.5));

    return timestep;
}

// Peak resident set size in bytes, or 0 if /proc is unavailable.
static size_t peak_host_rss(void) {
    FILE *fp = fopen("/proc/self/status", "r");
    if (!fp) return 0;
    char line[256];
    size_t kb = 0;
    while (fgets(line, sizeof(line), fp))
        if (sscanf(line, "VmHWM: %zu kB", &kb) == 1) break;
    fclose(fp);
    return kb * 1024;
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    bench_opts O;
    memset(&O, 0, sizeof(O));
    O.nx = 200; O.ny = 200;
    O.steps = 100; O.warmup = 5; O.repeat = 1;
    O.apply_forcing = 1;
    O.atol = 1.0e-10;
    O.rtol = 1.0e-8;

    bench_params P;
    bench_params_defaults(&P);

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--nx"))         O.nx = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--ny"))         O.ny = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--steps"))      O.steps = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--warmup"))     O.warmup = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--repeat"))     O.repeat = (int)arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--lenx"))       P.length_x = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--leny"))       P.length_y = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--manning"))    P.manning = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--water"))      P.water_level = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--dam"))        P.dam_height = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--cfl"))        P.cfl = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--atol"))       O.atol = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--rtol"))       O.rtol = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--save"))       O.save_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--check"))      O.check_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--csv"))        O.csv_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--phases"))     O.phases = 1;
        else if (!strcmp(a, "--verbose"))    O.verbose = 1;
        else if (!strcmp(a, "--no-friction")) O.apply_forcing = 0;
        else if (!strcmp(a, "--order")) {
            const char *o = arg_s(argc, argv, &i, a);
            if      (!strcmp(o, "row"))    O.morton = 0;
            else if (!strcmp(o, "morton")) O.morton = 1;
            else { fprintf(stderr, "bench: unknown order '%s'\n", o); return 2; }
        }
        else if (!strcmp(a, "--case")) {
            const char *c = arg_s(argc, argv, &i, a);
            if      (!strcmp(c, "dam"))      P.which_case = BENCH_CASE_DAM;
            else if (!strcmp(c, "dambumps")) P.which_case = BENCH_CASE_DAMBUMPS;
            else if (!strcmp(c, "lake"))     P.which_case = BENCH_CASE_LAKE;
            else { fprintf(stderr, "bench: unknown case '%s'\n", c); return 2; }
        }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "bench: unknown option '%s' (try --help)\n", a); return 2; }
    }

    // ---- build -----------------------------------------------------------
    bench_mesh M;
    bench_mesh_rectangular_cross(O.nx, O.ny, P.length_x, P.length_y, 0.0, 0.0, &M);
    if (O.morton)
        bench_mesh_reorder_morton(&M, O.nx, O.ny);

    bench_domain B;
    const double t_build0 = omp_get_wtime();
    bench_domain_build(&B, &M, &P);
    const double t_build = omp_get_wtime() - t_build0;


    const double t_map0 = omp_get_wtime();
    bench_domain_to_device(&B, &P, O.verbose);
    const double t_map = omp_get_wtime() - t_map0;

    struct gpu_domain *GD = &B.GD;
    const int64_t n = GD->D.number_of_elements;

    const char *case_name = P.which_case == BENCH_CASE_DAM      ? "dam"
                          : P.which_case == BENCH_CASE_DAMBUMPS ? "dambumps" : "lake";
#ifdef CPU_ONLY_MODE
    const char *build_kind = "host OpenMP (CPU_ONLY_MODE)";
#else
    const char *build_kind = "OpenMP target offload";
#endif

    printf("ANUGA shallow-water miniapp -- %s\n", build_kind);
    printf("  mesh      : %lld x %lld cross -> %lld triangles, %lld boundary edges\n",
           (long long)O.nx, (long long)O.ny, (long long)n, (long long)GD->D.boundary_length);
    printf("  case      : %s, %.0f x %.0f m, manning %.4g%s\n",
           case_name, P.length_x, P.length_y, P.manning,
           O.apply_forcing ? "" : " (friction off)");
    printf("  scheme    : rk2, CFL %.3g, DE1 limiter betas\n", P.cfl);
    printf("  ordering  : %s\n",
           O.morton ? "morton (Z-order curve)" : "row-major (ANUGA rectangular_cross)");
    printf("  devices   : %d visible, using %d\n", omp_get_num_devices(), GD->device_id);
    printf("  setup     : %.3f s build, %.3f s map-to-device\n", t_build, t_map);

    const size_t dev_need = gpu_estimate_required_memory(n, GD->D.boundary_length);
    size_t dev_free = 0, dev_total = 0;
    const int have_devmem = gpu_query_device_memory(&dev_free, &dev_total);
    if (have_devmem)
        printf("  memory    : %.2f GiB mapped to device, %.2f of %.2f GiB free after\n",
               dev_need / 1073741824.0, dev_free / 1073741824.0, dev_total / 1073741824.0);
    else
        printf("  memory    : %.2f GiB mapped to device (device query unavailable)\n",
               dev_need / 1073741824.0);
    fflush(stdout);

    const double volume0 = gpu_compute_water_volume(GD);

    // ---- warmup ----------------------------------------------------------
    double t_sim = 0.0, dt = 0.0;
    for (int64_t s = 0; s < O.warmup; s++)
        t_sim += (dt = gpu_evolve_one_rk2_step(GD, P.evolve_max_timestep, O.apply_forcing));

    // ---- timed loop ------------------------------------------------------
    double best = 1.0e300, total_all = 0.0;
    uint64_t flops_total = 0;

    for (int r = 0; r < O.repeat; r++) {
        memset(phase_time, 0, sizeof(phase_time));
        gpu_flop_counters_reset(GD);
        gpu_flop_counters_enable(GD, 1);

        const double t0 = omp_get_wtime();
        for (int64_t s = 0; s < O.steps; s++) {
            if (O.phases)
                t_sim += (dt = rk2_step_timed(GD, P.evolve_max_timestep, O.apply_forcing));
            else
                t_sim += (dt = gpu_evolve_one_rk2_step(GD, P.evolve_max_timestep, O.apply_forcing));
        }
        const double elapsed = omp_get_wtime() - t0;

        gpu_flop_counters_enable(GD, 0);
        flops_total = gpu_flop_counters_get_total(GD);
        total_all += elapsed;
        if (elapsed < best) best = elapsed;
        if (O.repeat > 1)
            printf("  run %2d/%d : %8.4f s  (%.4f ms/step)\n",
                   r + 1, O.repeat, elapsed, 1.0e3 * elapsed / (double)O.steps);
    }

    const double per_step = best / (double)O.steps;
    const double cellsteps_per_s = (double)n * (double)O.steps / best;

    printf("\n  timed     : %lld steps (+%lld warmup) in %.4f s%s\n",
           (long long)O.steps, (long long)O.warmup, best,
           O.repeat > 1 ? " (best of runs)" : "");
    printf("              %.4f ms/step, %.3f Mcell-steps/s\n",
           1.0e3 * per_step, 1.0e-6 * cellsteps_per_s);
    printf("              t = %.9g s, last dt = %.6g s\n", t_sim, dt);
    if (flops_total > 0)
        printf("  flops     : %.3f GFLOP over the timed loop, %.2f GFLOP/s\n",
               1.0e-9 * (double)flops_total, 1.0e-9 * (double)flops_total / best);

    if (O.phases) {
        printf("\n  per-kernel breakdown (per step, averaged over %lld steps)\n",
               (long long)O.steps);
        double summed = 0.0;
        for (int p = 0; p < PH_NPHASES; p++) summed += phase_time[p];
        for (int p = 0; p < PH_NPHASES; p++) {
            if (phase_time[p] == 0.0) continue;
            printf("    %-16s %9.4f ms   %5.1f%%\n", phase_names[p],
                   1.0e3 * phase_time[p] / (double)O.steps,
                   100.0 * phase_time[p] / summed);
        }
        printf("    %-16s %9.4f ms   (%.1f%% of wall time accounted for)\n",
               "sum", 1.0e3 * summed / (double)O.steps, 100.0 * summed / total_all);
    }

    // ---- diagnostics -----------------------------------------------------
    const double volume1 = gpu_compute_water_volume(GD);
    printf("\n  volume    : %.12g -> %.12g m^3 (drift %.3e relative)\n",
           volume0, volume1, volume0 != 0.0 ? (volume1 - volume0) / volume0 : 0.0);

    gpu_domain_sync_from_device(GD);

    double max_speed_sq = 0.0, max_stage = -1.0e300, min_stage = 1.0e300;
    int nan_count = 0;
    for (int64_t k = 0; k < n; k++) {
        const double w  = GD->D.stage_centroid_values[k];
        const double uh = GD->D.xmom_centroid_values[k];
        const double vh = GD->D.ymom_centroid_values[k];
        if (isnan(w) || isnan(uh) || isnan(vh)) nan_count++;
        if (w > max_stage) max_stage = w;
        if (w < min_stage) min_stage = w;
        const double m2 = uh * uh + vh * vh;
        if (m2 > max_speed_sq) max_speed_sq = m2;
    }
    printf("  state     : stage in [%.6g, %.6g], max |momentum| %.6e%s\n",
           min_stage, max_stage, sqrt(max_speed_sq),
           nan_count ? "  *** NaNs present ***" : "");
    if (P.which_case == BENCH_CASE_LAKE)
        printf("  lake test : water started at rest; max |momentum| above should stay ~0\n");

    int rc = nan_count ? 1 : 0;
    const int64_t total_steps = O.warmup + O.steps * O.repeat;

    if (O.save_path)
        rc |= snapshot_save(O.save_path, GD, M.orig_id, O.nx, O.ny, (int)P.which_case,
                            total_steps, t_sim, dt);
    if (O.check_path)
        rc |= snapshot_check(O.check_path, GD, M.orig_id, O.rtol, O.atol);

    if (O.csv_path) {
        FILE *fp = fopen(O.csv_path, "r");
        const int fresh = (fp == NULL);
        if (fp) fclose(fp);
        fp = fopen(O.csv_path, "a");
        if (!fp) {
            perror(O.csv_path);
            rc |= 1;
        } else {
            if (fresh)
                fprintf(fp, "nx,ny,triangles,case,steps,ms_per_step,mcellsteps_per_s,"
                            "gflops,build_s,map_s,dev_bytes,host_peak_bytes,"
                            "volume_drift,max_momentum,nans\n");
            fprintf(fp, "%lld,%lld,%lld,%s,%lld,%.6f,%.4f,%.4f,%.4f,%.4f,"
                        "%zu,%zu,%.6e,%.6e,%d\n",
                    (long long)O.nx, (long long)O.ny, (long long)n, case_name,
                    (long long)O.steps, 1.0e3 * per_step, 1.0e-6 * cellsteps_per_s,
                    1.0e-9 * (double)flops_total / best, t_build, t_map,
                    dev_need, peak_host_rss(),
                    volume0 != 0.0 ? (volume1 - volume0) / volume0 : 0.0,
                    sqrt(max_speed_sq), nan_count);
            fclose(fp);
        }
    }

    bench_domain_free(&B);
    bench_mesh_free(&M);
    return rc;
}

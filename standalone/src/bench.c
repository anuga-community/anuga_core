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
#include "core_kernels.h"
#include "gpu_omp_macros.h"
#include "mesh.h"

// ---- CUDA experiment machinery: NVIDIA toolchain only ------------------
#if !defined(CPU_ONLY_MODE) && defined(__NVCOMPILER)
// CUDA experiment: hand-written reconstruction kernel in a pure-nvcc shared
// library (build/gpu/cuextrap.so), dlopen()ed to keep the CUDA runtime out of
// the OpenMP-target link (which it breaks).  We resolve the device pointers
// for the OpenMP-mapped arrays here and hand them across; both runtimes use
// the CUDA primary context, so the pointers are valid on the other side.
#include <dlfcn.h>
#include "cuda_extrap.h"
static int g_cuda_extrap_tpb = 0;

static cuda_extrap_fn g_cuda_extrap_launch = NULL;

static void *dev_ptr(const void *host) {
    void *p = omp_get_mapped_ptr(host, omp_get_default_device());
    if (!p) { fprintf(stderr, "bench: %p not device-mapped\n", host); exit(1); }
    return p;
}

static void cuda_extrap_load(void) {
    void *h = dlopen("./build/gpu/cuextrap.so", RTLD_NOW);
    if (!h) { fprintf(stderr, "bench: %s\n", dlerror()); exit(1); }
    g_cuda_extrap_launch = (cuda_extrap_fn)dlsym(h, "cuda_extrapolate_launch");
    if (!g_cuda_extrap_launch) { fprintf(stderr, "bench: %s\n", dlerror()); exit(1); }
}

#else
// CPU builds (gcc/libgomp lacks omp_get_mapped_ptr) and non-NVIDIA
// offload toolchains (AMD/Intel: no CUDA) compile the experiment out.
static int g_cuda_extrap_tpb = 0;
static void cuda_extrap_load(void) {
    fprintf(stderr, "bench: --cuda-extrap needs the NVIDIA GPU build\n");
    exit(2);
}
#endif

// ---- portable machinery (all toolchains) --------------------------------
// Active-set mode: skip cells/edges that provably cannot change (dry with an
// all-dry neighbourhood).  Rebuilt every step with a 2-ring cell halo; the
// first step always runs full so every cell's edge values and protect clamps
// exist once.  Requires scatter fluxes.  Bit-exact (verified via goldens).
static int g_active_set = 0;
static int g_active_ready = 0;            // 0 until the first full step ran
static anuga_int *g_as_wet, *g_as_ring1, *g_as_cells, *g_as_edges;
static anuga_int g_as_counts[2];
// --active-every K: rebuild the full classification only every K steps and
// run the per-step rebuild over a CANDIDATE superset instead of the mesh.
// The candidate is the ring-2 cell list dilated by 2K+2 further rings (plus
// every ghost, and the ring-2 around them, so a front arriving from another
// rank is never missed): CFL lets wetness advance at most 2 rings per RK2
// step, the update reaches one ring beyond that, and one more ring keeps the
// neighbour flags the restricted passes read correct.  Cells outside are
// provably unchanged for the next K steps, so the per-step lists are the
// same sets the full scan would produce.
static int g_active_every = 1;
static int g_active_verify = 0;
static anuga_int *g_cand = NULL, *g_cand_flag = NULL, *g_cand_edges = NULL;
static anuga_int g_cand_n = 0, g_cand_ne = 0;
static long g_cand_rebuilds = 0;
static double g_cand_frac_sum = 0.0;
static int g_as_phase = 0;                // steps since the last full rebuild
static double g_as_cellfrac_sum; static long g_as_samples;
// --tile-stats: per-cell count of steps in the active set (device-resident
// in a GPU build), reduced per owned tile at the end -> the wetness weights
// tools/tile_assign.py balances the next run with.
static int *g_act_count = NULL;

// Uniform rain forcing (fractional-step style, applied after each step with
// that step's dt, over ALL cells -- deliberately including inactive ones,
// which is exactly what stresses the active-set classification).
static double g_rain_mmhr = 0.0;      // rate while raining
static double g_rain_every = 0.0;     // pulse period in sim seconds (0 = always)
static double g_rain_for = 0.0;       // pulse duration
static double g_rain_x0 = -1.0, g_rain_x1 = -1.0;   // optional x-band (fraction)

// Gridded rain replay ("ANUGARN1", from tools/rain_csv_to_grid.py): a
// piecewise-constant-in-time raster of rates (mm/hr) covering the domain,
// sampled per cell via a precomputed centroid -> grid-cell index.  This is
// the shape real model-driven rainfall arrives in (forecast / design-storm
// CSVs); production ANUGA feeds the same data through its per-cell-rate
// operator (gpu_rate_operator_apply_array).
static int64_t g_rg_nx, g_rg_ny, g_rg_nt;
static double *g_rg_times;      // [nt] sim seconds, sorted
static float  *g_rg_rates;      // [nt * ny * nx] mm/hr
static anuga_int *g_rg_cellidx; // [n] centroid -> raster index
static const char *g_rain_grid_path = NULL;
static void rain_grid_load(struct gpu_domain *GD) {
    FILE *fp = fopen(g_rain_grid_path, "rb");
    if (!fp) { perror(g_rain_grid_path); exit(1); }
    char magic[8];
    double x0, y0, dx, dy;
    if (fread(magic, 1, 8, fp) != 8 || memcmp(magic, "ANUGARN1", 8) != 0 ||
        fread(&g_rg_nx, 8, 1, fp) != 1 || fread(&g_rg_ny, 8, 1, fp) != 1 ||
        fread(&x0, 8, 1, fp) != 1 || fread(&y0, 8, 1, fp) != 1 ||
        fread(&dx, 8, 1, fp) != 1 || fread(&dy, 8, 1, fp) != 1 ||
        fread(&g_rg_nt, 8, 1, fp) != 1) {
        fprintf(stderr, "bench: %s is not ANUGARN1\n", g_rain_grid_path);
        exit(1);
    }
    const size_t cells = (size_t)g_rg_nx * (size_t)g_rg_ny;
    g_rg_times = (double *)malloc((size_t)g_rg_nt * sizeof(double));
    g_rg_rates = (float *)malloc((size_t)g_rg_nt * cells * sizeof(float));
    if (fread(g_rg_times, 8, (size_t)g_rg_nt, fp) != (size_t)g_rg_nt ||
        fread(g_rg_rates, 4, (size_t)g_rg_nt * cells, fp) != (size_t)g_rg_nt * cells) {
        fprintf(stderr, "bench: truncated %s\n", g_rain_grid_path);
        exit(1);
    }
    fclose(fp);

    struct domain *D = &GD->D;
    const anuga_int nel = D->number_of_elements;
    g_rg_cellidx = (anuga_int *)malloc((size_t)nel * sizeof(anuga_int));
    for (anuga_int k = 0; k < nel; k++) {
        int64_t gi = (int64_t)((D->centroid_coordinates[2 * k] - x0) / dx);
        int64_t gj = (int64_t)((D->centroid_coordinates[2 * k + 1] - y0) / dy);
        if (gi < 0) gi = 0; if (gi >= g_rg_nx) gi = g_rg_nx - 1;
        if (gj < 0) gj = 0; if (gj >= g_rg_ny) gj = g_rg_ny - 1;
        g_rg_cellidx[k] = gj * g_rg_nx + gi;
    }
    {
        float *r = g_rg_rates; anuga_int *ci = g_rg_cellidx;
        const size_t rn = (size_t)g_rg_nt * cells;
        #pragma omp target enter data map(to: r[0:rn], ci[0:nel])
    }
    printf("  rain grid : %s (%lldx%lld, %lld intervals, t %.0f..%.0f s)\n",
           g_rain_grid_path, (long long)g_rg_nx, (long long)g_rg_ny,
           (long long)g_rg_nt, g_rg_times[0], g_rg_times[g_rg_nt - 1]);
}

static void apply_rain_grid(struct gpu_domain *GD, double t_sim, double dt) {
    if (t_sim < g_rg_times[0]) return;
    int64_t lo = 0, hi = g_rg_nt - 1;
    while (lo < hi) { int64_t mid = (lo + hi + 1) / 2;
        if (g_rg_times[mid] <= t_sim) lo = mid; else hi = mid - 1; }
    const size_t base = (size_t)lo * (size_t)g_rg_nx * (size_t)g_rg_ny;
    const double conv = dt / 1000.0 / 3600.0;
    struct domain *D = &GD->D;
    const anuga_int nel = D->number_of_elements;
    double * restrict stage_cv = D->stage_centroid_values;
    float * restrict rates = g_rg_rates;
    anuga_int * restrict ci = g_rg_cellidx;
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < nel; k++) {
        stage_cv[k] += conv * (double)rates[base + ci[k]];
    }
}

static void apply_rain(struct gpu_domain *GD, double t_sim, double dt) {
    if (g_rain_grid_path) { apply_rain_grid(GD, t_sim, dt); return; }
    if (g_rain_mmhr <= 0.0) return;
    if (g_rain_every > 0.0) {
        const double phase = fmod(t_sim, g_rain_every);
        if (phase >= g_rain_for) return;
    }
    const double dstage = g_rain_mmhr / 1000.0 / 3600.0 * dt;
    struct domain *D = &GD->D;
    const anuga_int n = D->number_of_elements;
    double * restrict stage_cv = D->stage_centroid_values;
    double * restrict cc = D->centroid_coordinates;
    const double x0 = g_rain_x0, x1 = g_rain_x1;
    const int banded = (x0 >= 0.0);
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        if (!banded || (cc[2 * k] >= x0 && cc[2 * k] <= x1))
            stage_cv[k] += dstage;
    }
}

// Full rebuild, then grow the candidate superset the next K-1 steps iterate.
static void rebuild_candidate(struct gpu_domain *GD) {
    struct domain *D = &GD->D;
    const anuga_int n = D->number_of_elements;

    // Clear only what the previous candidate marked -- O(candidate), not O(n).
    if (g_cand_n > 0) {
        anuga_int * restrict fl = g_cand_flag;
        const anuga_int * restrict cd = g_cand;
        const anuga_int nc = g_cand_n;
        OMP_PARALLEL_LOOP
        for (anuga_int q = 0; q < nc; q++) fl[cd[q]] = 0;
    }

    core_build_active_sets(D, g_as_wet, g_as_ring1, g_as_cells, g_as_edges,
                           D->owned_edges, D->num_owned_edges, g_as_counts);

    // Seed: this step's cell list, plus every ghost.  A ghost's stage is
    // rewritten by the halo exchange rather than by the kernels, so its
    // wetness can change with nothing on this rank predicting it; seeding
    // all of them (a perimeter-sized set) and dilating means a front
    // crossing a rank boundary is inside the candidate before it arrives.
    anuga_int m = g_as_counts[0];
    {
        anuga_int * restrict cd = g_cand;
        anuga_int * restrict fl = g_cand_flag;
        const anuga_int * restrict ac = g_as_cells;
        OMP_PARALLEL_LOOP
        for (anuga_int q = 0; q < m; q++) { cd[q] = ac[q]; fl[ac[q]] = 1; }
    }
    if (D->tri_full_flag != NULL) {
        anuga_int * restrict cd = g_cand;
        anuga_int * restrict fl = g_cand_flag;
        const anuga_int * restrict full = D->tri_full_flag;
        anuga_int cur = m;
        #ifdef CPU_ONLY_MODE
        #pragma omp parallel for
        #else
        #pragma omp target teams distribute parallel for map(tofrom: cur)
        #endif
        for (anuga_int k = 0; k < n; k++) {
            if (full[k] == 1) continue;
            anuga_int was;
            #pragma omp atomic capture
            { was = fl[k]; fl[k] = 1; }
            if (!was) {
                anuga_int idx;
                #pragma omp atomic capture
                idx = cur++;
                cd[idx] = k;
            }
        }
        m = cur;
    }

    core_active_dilate_rings(D, g_cand_flag, g_cand, m,
                             2 * g_active_every + 2);
    // Ascending order, so the lists the restricted rebuild emits keep the
    // locality the step kernels get from the ordered full scan.
    g_cand_n = core_active_compact_flag(D, g_cand_flag, g_cand);
    // The candidate's own owned-edge slots, still in slot order: what the
    // per-step pass 4 filters in place of the whole mesh's.
    g_cand_ne = core_active_edges_of(D, g_cand_flag, D->owned_edges,
                                     D->num_owned_edges, g_cand_edges);
    g_cand_rebuilds++;
    g_cand_frac_sum += (double)g_cand_n / (double)n;
}

// --active-verify: the restricted rebuild must produce the SAME SETS as a
// full scan.  Compares sorted copies of both lists every step; the physics
// gate then covers everything downstream of them.
static int cmp_ai(const void *a, const void *b) {
    const anuga_int x = *(const anuga_int *)a, y = *(const anuga_int *)b;
    return (x > y) - (x < y);
}
static void verify_active_sets(struct gpu_domain *GD) {
    struct domain *D = &GD->D;
    const anuga_int n = D->number_of_elements;
    static anuga_int *w2, *r2, *c2, *e2, *mine_c, *mine_e;
    static long nfail = 0;
    anuga_int cnt2[2];
    if (!w2) {
        w2 = (anuga_int *)malloc((size_t)n * sizeof(anuga_int));
        r2 = (anuga_int *)malloc((size_t)n * sizeof(anuga_int));
        c2 = (anuga_int *)malloc((size_t)n * sizeof(anuga_int));
        e2 = (anuga_int *)malloc((size_t)D->num_owned_edges * sizeof(anuga_int));
        mine_c = (anuga_int *)malloc((size_t)n * sizeof(anuga_int));
        mine_e = (anuga_int *)malloc((size_t)D->num_owned_edges * sizeof(anuga_int));
        const anuga_int ne = D->num_owned_edges;
        #pragma omp target enter data map(alloc: w2[0:n], r2[0:n], c2[0:n], e2[0:ne])
    }
    // Snapshot the restricted result before the full scan overwrites the flags.
    const anuga_int nc = g_as_counts[0], nee = g_as_counts[1];
    {
        anuga_int *src_c = g_as_cells, *src_e = g_as_edges;
        (void)src_c; (void)src_e;
        #ifndef CPU_ONLY_MODE
        #pragma omp target update from(src_c[0:nc], src_e[0:nee])
        #endif
        memcpy(mine_c, g_as_cells, (size_t)nc * sizeof(anuga_int));
        memcpy(mine_e, g_as_edges, (size_t)nee * sizeof(anuga_int));
    }
    core_build_active_sets(D, w2, r2, c2, e2, D->owned_edges,
                           D->num_owned_edges, cnt2);
    {
        anuga_int *rc = c2, *re = e2;
        const anuga_int a = cnt2[0], b = cnt2[1];
        (void)rc; (void)re; (void)a; (void)b;
        #ifndef CPU_ONLY_MODE
        #pragma omp target update from(rc[0:a], re[0:b])
        #endif
    }
    int bad = (cnt2[0] != nc) || (cnt2[1] != nee);
    if (!bad) {
        qsort(mine_c, (size_t)nc, sizeof(anuga_int), cmp_ai);
        qsort(mine_e, (size_t)nee, sizeof(anuga_int), cmp_ai);
        qsort(c2, (size_t)nc, sizeof(anuga_int), cmp_ai);
        qsort(e2, (size_t)nee, sizeof(anuga_int), cmp_ai);
        bad = memcmp(mine_c, c2, (size_t)nc * sizeof(anuga_int)) != 0
           || memcmp(mine_e, e2, (size_t)nee * sizeof(anuga_int)) != 0;
    }
    if (bad && nfail++ < 5)
        fprintf(stderr, "active-verify MISMATCH: cells %lld vs %lld, edges "
                        "%lld vs %lld\n", (long long)nc, (long long)cnt2[0],
                        (long long)nee, (long long)cnt2[1]);
    // Restore the restricted result: the step runs on the lists it built.
    memcpy(g_as_cells, mine_c, (size_t)nc * sizeof(anuga_int));
    memcpy(g_as_edges, mine_e, (size_t)nee * sizeof(anuga_int));
    {
        anuga_int *dc = g_as_cells, *de = g_as_edges;
        (void)dc; (void)de;
        #ifndef CPU_ONLY_MODE
        #pragma omp target update to(dc[0:nc], de[0:nee])
        #endif
    }
    g_as_counts[0] = nc; g_as_counts[1] = nee;
}

static void active_step_lists(struct gpu_domain *GD,
                              const anuga_int **cells, anuga_int *ncells,
                              const anuga_int **edges, anuga_int *nedges) {
    if (!g_active_set || !g_active_ready) {
        *cells = NULL; *ncells = 0;
        *edges = GD->D.owned_edges; *nedges = GD->D.num_owned_edges;
        return;
    }
    if (g_active_every <= 1) {
        core_build_active_sets(&GD->D, g_as_wet, g_as_ring1, g_as_cells,
                               g_as_edges, GD->D.owned_edges,
                               GD->D.num_owned_edges, g_as_counts);
    } else {
        if (g_as_phase == 0) rebuild_candidate(GD);
        else
            core_build_active_sets_on(&GD->D, g_as_wet, g_as_ring1, g_as_cells,
                                      g_as_edges, g_cand_edges, g_cand_ne,
                                      g_as_counts, g_cand, g_cand_n);
        if (++g_as_phase >= g_active_every) g_as_phase = 0;
        if (g_active_verify) verify_active_sets(GD);
    }
    if (g_act_count) {
        int * restrict cnt = g_act_count;
        const anuga_int * restrict ac = g_as_cells;
        const anuga_int nac = g_as_counts[0];
        OMP_PARALLEL_LOOP
        for (anuga_int q = 0; q < nac; q++) cnt[ac[q]]++;   // each cell listed once
    }
    *cells = g_as_cells; *ncells = g_as_counts[0];
    *edges = g_as_edges; *nedges = g_as_counts[1];
    g_as_cellfrac_sum += (double)g_as_counts[0] / (double)GD->D.number_of_elements;
    g_as_samples++;
}

static void extrapolate_phase(struct gpu_domain *GD, double predictor_dt,
                              const anuga_int *iter, anuga_int iter_n) {
#if !defined(CPU_ONLY_MODE) && defined(__NVCOMPILER)
    if (g_cuda_extrap_tpb <= 0) {
        if (iter) core_extrapolate_edge_pass_on(&GD->D, predictor_dt, iter, iter_n);
        else      gpu_extrapolate_edges(GD, predictor_dt);
        return;
    }
    struct domain *D = &GD->D;
    struct extrap_args a;
    a.n = D->number_of_elements;
    a.minimum_allowed_height = D->minimum_allowed_height;
    a.extrapolate_velocity_second_order = D->extrapolate_velocity_second_order;
    a.g = D->g;
    a.beta_w = D->beta_w;    a.beta_w_dry = D->beta_w_dry;
    a.beta_uh = D->beta_uh;  a.beta_uh_dry = D->beta_uh_dry;
    a.beta_vh = D->beta_vh;  a.beta_vh_dry = D->beta_vh_dry;
    a.predictor_dt = predictor_dt;
    a.stage_cv  = dev_ptr(D->stage_centroid_values);
    a.xmom_cv   = dev_ptr(D->xmom_centroid_values);
    a.ymom_cv   = dev_ptr(D->ymom_centroid_values);
    a.bed_cv    = dev_ptr(D->bed_centroid_values);
    a.height_cv = dev_ptr(D->height_centroid_values);
    a.stage_ev  = dev_ptr(D->stage_edge_values);
    a.xmom_ev   = dev_ptr(D->xmom_edge_values);
    a.ymom_ev   = dev_ptr(D->ymom_edge_values);
    a.bed_ev    = dev_ptr(D->bed_edge_values);
    a.height_ev = dev_ptr(D->height_edge_values);
    a.centroid_coords = dev_ptr(D->centroid_coordinates);
    a.edge_coords     = dev_ptr(D->edge_coordinates);
    a.surrogate_neighbours = dev_ptr(D->surrogate_neighbours);
    a.number_of_boundaries = dev_ptr(D->number_of_boundaries);
    a.x_centroid_work = dev_ptr(D->x_centroid_work);
    a.y_centroid_work = dev_ptr(D->y_centroid_work);
    if (g_cuda_extrap_launch(a, g_cuda_extrap_tpb) != 0) exit(1);
#else
    (void)GD; (void)predictor_dt;
    if (iter) core_extrapolate_edge_pass_on(&GD->D, predictor_dt, iter, iter_n);
    else      gpu_extrapolate_edges(GD, predictor_dt);
#endif
}

#include "setup.h"
#include "snapshot.h"
#include "bench_mpi.h"
#include "tiles.h"

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

typedef struct {
    int64_t nx, ny;
    int64_t steps;
    int64_t warmup;
    int64_t report_every;
    double  max_wall;
    int     repeat;
    int     phases;
    int     verbose;
    int     apply_forcing;
    const char *mesh_path;
    const char *tiles_path;    // tiled distributed mesh (index.txt)
    const char *assign_path;   // tile -> rank, one int per tile
    const char *save_path;
    const char *check_path;
    const char *csv_path;
    const char *tile_stats_path;   // per-tile active fraction (needs --tiles --active-set)
    int     morton;
    int     device_init;   // build the domain on the device (generated mesh, serial, row order)
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
"    --mesh FILE       load an ANUGAMSH mesh (tools/make_basin_mesh.py)\n"
"    --tiles INDEX     tiled mesh: each rank loads its tiles + neighbour strips\n"
"                      (tools/tiles_to_bmesh.py index.txt); works serially too\n"
"    --assign FILE     tile -> rank map for --tiles (default: contiguous, balanced\n"
"                      by triangle count)\n"
"                      instead of generating the rectangular cross; brings\n"
"                      its own terrain and initial stage\n"
"    --case NAME       dam | dambumps | lake | river    (default dam)\n"
"                        dam       flat bed, wet dam break (every cell wet)\n"
"                        dambumps  bumpy bed dam break (wet/dry branches)\n"
"                        lake      water at rest over bumps (well-balanced)\n"
"                        river     reservoir breaks into a thin river in a\n"
"                                  carved channel; floodplain banks start dry\n"
"    --manning V       Manning n                        (default 0.03)\n"
"    --water V         still-water / downstream stage   (default 5)\n"
"    --dam V           upstream stage                   (default 10)\n"
"    --no-friction     skip the Manning forcing term\n"
"    --order NAME      row | morton | random -- element numbering (default row)\n"

"                        morton renumbers triangles along a Z-order curve so\n"
"                        both grid directions' neighbours stay cache-near;\n"
"                        snapshots stay in canonical order either way\n"
"\n"
"  scheme\n"
"    --scheme NAME     rk2 | ader2 | euler | rk3          (default rk2)\n"
"                      each selects its ANUGA preset (DE1 / DE_ader2 / DE0 /\n"
"                      DE2): flux calls per step, limiter betas, CFL\n"
"    --betas V         override the limiter betas (beta_w/uh/vh; dry stay 0)\n"
"    --flux NAME       cell | edge | scatter -- flux kernel  (default cell)\n"
"    --rain MMHR       uniform rainfall (mm/hr) applied after every step\n"
"    --rain-every S    pulse period in sim-seconds (with --rain-for S)\n"
"    --rain-band A B   rain only where A <= x <= B (metres)\n"
"    --rain-grid FILE  replay a gridded rain timeseries (ANUGARN1, from\n"
"                      tools/rain_csv_to_grid.py) -- model-driven rainfall\n"
"    --active-set      skip dry-with-dry-neighbourhood cells (flood domains;\n"
"                      needs --flux scatter; first step runs full; bit-exact)\n"
"    --cuda-extrap N   use the hand-written CUDA reconstruction kernel with\n"
"                      N threads/block (experiment; forces stepped loops)\n"
"                        edge solves each unique edge's Riemann problem ONCE\n"
"                        (cell solves interior edges twice) and fuses the\n"
"                        flux gather into the update; same discretization,\n"
"                        exactly antisymmetric flux exchange, results differ\n"
"                        from cell-based only at floating-point roundoff\n"
"\n"
"  run\n"
"    --steps N         timed RK2 steps                  (default 100)\n"
"    --warmup N        untimed RK2 steps first          (default 5)\n"
"    --repeat N        repeat the timed loop N times, report the best\n"
"    --cfl V           CFL number                       (default 1.0)\n"
"    --phases          per-kernel timing breakdown\n"
"    --device-init     build the domain ON the device (generated mesh, serial,\n"
"                      row order): map(alloc) + fill kernels instead of a host\n"
"                      build + transfer; the host never holds the big arrays\n"
"    --verbose         let the kernels print their own setup messages\n"
"\n"
"  correctness\n"
"    --save FILE       write the final centroid state to FILE\n"
"    --check FILE      compare the final centroid state against FILE\n"
"    --atol V          absolute tolerance for --check    (default 1e-10)\n"
"    --rtol V          relative tolerance for --check    (default 1e-8)\n"
"    --ftol V          instead of rtol: max diff <= atol + V * max|reference|\n"
"                      per field (the roundoff gate for scatter/MPI runs)\n"
"\n"
"  reporting\n"
"    --csv FILE        append one machine-readable result row to FILE\n"
"    --active-every K  rebuild the full classification every K steps only; the\n"
"                      other steps reclassify a candidate superset (ring 2K+2)\n"
"                      instead of the mesh.  Same sets, O(active) not O(n).\n"
"    --active-verify   check every restricted rebuild against a full scan\n"
"    --tile-stats FILE write per-tile mean active fraction (with --tiles --active-set)\n"
"                      -> tools/tile_assign.py turns it into a balanced --assign\n"
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
// Mirrors gpu_evolve_one_rk2_step() in gpu_kernels.c exactly (no fixed
// timestep; dt allreduce + mid-step halo exchange under MPI), with a timer
// around each kernel.  Keep the two in sync:
// --phases and the plain loop must produce identical state.
// ---------------------------------------------------------------------------

enum {
    PH_ACTIVE = 0, PH_PREPARE, PH_EXTRAPOLATE, PH_BOUNDARY,
    PH_FLUXES, PH_FORCING_UPDATE, PH_NPHASES
};

static const char *phase_names[PH_NPHASES] = {
    "active_sets", "prepare", "extrapolate+ck", "boundary",
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

    const anuga_int *ac, *ae; anuga_int nac, nae;
    TIME_PHASE(PH_ACTIVE, active_step_lists(GD, &ac, &nac, &ae, &nae));
    const int zeu = gpu_prepare_should_zero_eu(GD);

    // ---- first Euler stage
    // prepare = fused RK2 backup + protect + extrapolate centroid pass
    TIME_PHASE(PH_PREPARE,
               ac ? (void)core_prepare_step_on(&GD->D, 1, zeu, ac, nac)
                  : (void)gpu_prepare_step(GD, 1, zeu));
    TIME_PHASE(PH_EXTRAPOLATE, extrapolate_phase(GD, 0.0, ac, nac));
    TIME_PHASE(PH_BOUNDARY,    evaluate_boundaries(GD));

    double local_timestep;
    TIME_PHASE(PH_FLUXES, local_timestep =
               ac ? core_compute_fluxes_scatter_on(&GD->D, 0, 2, ae, nae)
                  : gpu_flux_phase(GD, 0, 2));

    timestep = GD->CFL * bmpi_min_d(local_timestep);   // global CFL min under MPI
    GD->recorded_flux_timestep =
        (timestep < GD->evolve_max_timestep) ? timestep : GD->evolve_max_timestep;
    if (timestep > max_timestep) timestep = max_timestep;

    TIME_PHASE(PH_FORCING_UPDATE,
               ac ? core_forcing_and_update_on(&GD->D, timestep, apply_forcing, 0, 0.0, 0.0, ac, nac)
                  : gpu_apply_phase(GD, timestep, apply_forcing, 0, 0.0, 0.0, 0));
    // Mid-step halo exchange, exactly where gpu_evolve_one_rk2_step does it.
    // The active-set lists were built from the ghost columns as they stood at
    // the start of the step, which is what the serial 2-ring sees too: a
    // ghost cell's own update is garbage (it only saw its owned-side edges)
    // and is replaced here before the second stage reads it.
    if (GD->nprocs > 1) gpu_exchange_ghosts(GD);

    // ---- second Euler stage
    TIME_PHASE(PH_PREPARE,
               ac ? (void)core_prepare_step_on(&GD->D, 0, zeu, ac, nac)
                  : (void)gpu_prepare_step(GD, 0, zeu));
    TIME_PHASE(PH_EXTRAPOLATE, extrapolate_phase(GD, 0.0, ac, nac));
    TIME_PHASE(PH_BOUNDARY,    evaluate_boundaries(GD));
    TIME_PHASE(PH_FLUXES,
               ac ? core_compute_fluxes_scatter_on(&GD->D, 1, 2, ae, nae)
                  : gpu_flux_phase(GD, 1, 2));

    TIME_PHASE(PH_FORCING_UPDATE,
               ac ? core_forcing_and_update_on(&GD->D, timestep, apply_forcing, 1, 0.5, 0.5, ac, nac)
                  : gpu_apply_phase(GD, timestep, apply_forcing, 1, 0.5, 0.5, 1));

    g_active_ready = 1;
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

// Mirrors gpu_evolve_one_ader2_step() (single flux call; fused edge C-K
// predictor shifts edge values to Q^{n+1/2}).  Keep in sync like rk2 above.
static double ader2_step_timed(struct gpu_domain *GD, double max_timestep,
                               int apply_forcing, double prev_dt) {
    const anuga_int *ac, *ae; anuga_int nac, nae;
    TIME_PHASE(PH_ACTIVE, active_step_lists(GD, &ac, &nac, &ae, &nae));
    const int zeu = gpu_prepare_should_zero_eu(GD);

    TIME_PHASE(PH_PREPARE,
               ac ? (void)core_prepare_step_on(&GD->D, 0, zeu, ac, nac)
                  : (void)gpu_prepare_step(GD, 0, zeu));
    // reconstruction + C-K predictor fused into one launch (0.0 = bootstrap)
    TIME_PHASE(PH_EXTRAPOLATE,
               extrapolate_phase(GD, prev_dt > 0.0 ? prev_dt * 0.5 : 0.0, ac, nac));
    TIME_PHASE(PH_BOUNDARY,    evaluate_boundaries(GD));

    double local_timestep;
    TIME_PHASE(PH_FLUXES, local_timestep =
               ac ? core_compute_fluxes_scatter_on(&GD->D, 0, 1, ae, nae)
                  : gpu_flux_phase(GD, 0, 1));

    double timestep = GD->CFL * bmpi_min_d(local_timestep);   // global CFL min
    GD->recorded_flux_timestep =
        (timestep < GD->evolve_max_timestep) ? timestep : GD->evolve_max_timestep;
    if (timestep > max_timestep) timestep = max_timestep;

    TIME_PHASE(PH_FORCING_UPDATE,
               ac ? core_forcing_and_update_on(&GD->D, timestep, apply_forcing, 0, 0.0, 0.0, ac, nac)
                  : gpu_apply_phase(GD, timestep, apply_forcing, 0, 0.0, 0.0, 0));
    g_active_ready = 1;
    return timestep;
}

// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// MPI support (np > 1): slab decomposition, gathered snapshots, global stats.
// All MPI traffic in the driver goes through bench_mpi.h; the gpu kernels
// use their own MPI (dt allreduce, halo exchange) via gpu_domain.
// ---------------------------------------------------------------------------

static const double *snap_field_ptr(const struct gpu_domain *GD, int f) {
    switch (f) {
        case SNAP_STAGE:  return GD->D.stage_centroid_values;
        case SNAP_XMOM:   return GD->D.xmom_centroid_values;
        case SNAP_YMOM:   return GD->D.ymom_centroid_values;
        case SNAP_HEIGHT: return GD->D.height_centroid_values;
        default:          return GD->D.bed_centroid_values;
    }
}

// Water volume over OWNED cells only, from host arrays (ghosts excluded --
// gpu_compute_water_volume() counts every local cell and would double-count
// the ghost columns across ranks).
static double host_water_volume(const struct gpu_domain *GD,
                                const int64_t *full_flag) {
    const anuga_int n = GD->D.number_of_elements;
    const double *stage = GD->D.stage_centroid_values;
    const double *bed   = GD->D.bed_centroid_values;
    const anuga_geom_t *areas = GD->D.areas;
    double v = 0.0;
    for (anuga_int k = 0; k < n; k++) {
        if (full_flag && !full_flag[k]) continue;
        const double h = stage[k] - bed[k];
        if (h > 0.0) v += h * (double)areas[k];
    }
    return v;
}

// Gather every rank's owned triangles into canonical global order on rank 0,
// then save or check through the canon snapshot API.  Collective.
static int mpi_snapshot(const char *save_path, const char *check_path,
                        const struct gpu_domain *GD, const bench_mesh *M,
                        const int64_t *full_flag, int64_t n_full, int64_t n_global,
                        int64_t nx, int64_t ny, int which_case,
                        int64_t total_steps, double t, double last_dt,
                        double rtol, double atol) {
    const anuga_int n_local = GD->D.number_of_elements;
    const int np = bmpi_size(), rank = bmpi_rank();
    const int cnt = (int)n_full;

    int64_t *ids = (int64_t *)malloc((size_t)cnt * sizeof(int64_t));
    double  *val = (double  *)malloc((size_t)cnt * sizeof(double));
    int *rcounts = (int *)malloc((size_t)np * sizeof(int));
    int *displs  = (int *)malloc((size_t)np * sizeof(int));

    int64_t *rids = NULL;
    double *canon[SNAP_NFIELDS] = {0};
    double *rval = NULL;
    if (rank == 0) {
        rids = (int64_t *)malloc((size_t)n_global * sizeof(int64_t));
        rval = (double *)malloc((size_t)n_global * sizeof(double));
        for (int f = 0; f < SNAP_NFIELDS; f++)
            canon[f] = (double *)malloc((size_t)n_global * sizeof(double));
    }

    int c = 0;
    for (anuga_int k = 0; k < n_local; k++)
        if (!full_flag || full_flag[k]) ids[c++] = M->orig_id ? M->orig_id[k] : k;
    bmpi_gatherv_i64(ids, cnt, rids, rcounts, displs);

    for (int f = 0; f < SNAP_NFIELDS; f++) {
        const double *src = snap_field_ptr(GD, f);
        c = 0;
        for (anuga_int k = 0; k < n_local; k++)
            if (!full_flag || full_flag[k]) val[c++] = src[k];
        const int64_t got = bmpi_gatherv_d(val, cnt, rval, rcounts, displs);
        if (rank == 0) {
            if (got != n_global) {
                fprintf(stderr, "bench: gathered %lld of %lld triangles\n",
                        (long long)got, (long long)n_global);
            }
            for (int64_t e = 0; e < got; e++) canon[f][rids[e]] = rval[e];
        }
    }

    int rc = 0;
    if (rank == 0) {
        const int64_t nb_global = 2 * (nx + ny);   // header metadata only
        if (save_path)
            rc |= snapshot_save_canon(save_path, n_global, nb_global, nx, ny,
                                      which_case, total_steps, t, last_dt, canon);
        if (check_path)
            rc |= snapshot_check_canon(check_path, n_global, canon, rtol, atol);
    }
    bmpi_bcast_i(&rc);

    free(ids); free(val); free(rcounts); free(displs);
    if (rank == 0) {
        free(rids); free(rval);
        for (int f = 0; f < SNAP_NFIELDS; f++) free(canon[f]);
    }
    return rc;
}

int main(int argc, char **argv) {
    bmpi_init(&argc, &argv);
    atexit(bmpi_finalize);
    const int g_rank = bmpi_rank();
    const int g_np   = bmpi_size();
    if (g_rank != 0) {
        // One reporting stream: silence stdout on non-root ranks (stderr stays).
        FILE *devnull = freopen("/dev/null", "w", stdout);
        (void)devnull;
    }
    bench_opts O;
    memset(&O, 0, sizeof(O));
    O.nx = 200; O.ny = 200;
    O.steps = 100; O.warmup = 5; O.repeat = 1;
    O.apply_forcing = 1;
    O.atol = 1.0e-10;
    O.rtol = 1.0e-8;

    bench_params P;
    bench_params_defaults(&P);
    int scheme_set = 0, cfl_set = 0;
    double betas_override = -1.0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a, "--nx"))         O.nx = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--ny"))         O.ny = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--steps"))      O.steps = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--warmup"))     O.warmup = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--repeat"))     O.repeat = (int)arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--report"))     O.report_every = arg_i(argc, argv, &i, a);
        else if (!strcmp(a, "--max-wall"))   O.max_wall = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--lenx"))       P.length_x = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--leny"))       P.length_y = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--manning"))    P.manning = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--water"))      P.water_level = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--dam"))        P.dam_height = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--cfl"))        { P.cfl = arg_d(argc, argv, &i, a); cfl_set = 1; }
        else if (!strcmp(a, "--betas"))      { betas_override = arg_d(argc, argv, &i, a); }
        else if (!strcmp(a, "--flux")) {
            const char *fx = arg_s(argc, argv, &i, a);
            if      (!strcmp(fx, "cell"))    P.flux_mode = 0;
            else if (!strcmp(fx, "edge"))    P.flux_mode = 1;
            else if (!strcmp(fx, "scatter")) P.flux_mode = 2;
            else { fprintf(stderr, "bench: unknown flux mode '%s'\n", fx); return 2; }
        }
        else if (!strcmp(a, "--scheme")) {
            const char *sc = arg_s(argc, argv, &i, a);
            if      (!strcmp(sc, "rk2"))   P.scheme = BENCH_SCHEME_RK2;
            else if (!strcmp(sc, "ader2")) P.scheme = BENCH_SCHEME_ADER2;
            else if (!strcmp(sc, "euler")) P.scheme = BENCH_SCHEME_EULER;
            else if (!strcmp(sc, "rk3"))   P.scheme = BENCH_SCHEME_RK3;
            else { fprintf(stderr, "bench: unknown scheme '%s'\n", sc); return 2; }
            scheme_set = 1;
        }
        else if (!strcmp(a, "--atol"))       O.atol = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--rtol"))       O.rtol = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--ftol"))       O.rtol = -arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--mesh"))       O.mesh_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--tiles"))      O.tiles_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--assign"))     O.assign_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--save"))       O.save_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--check"))      O.check_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--csv"))        O.csv_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--tile-stats")) O.tile_stats_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--phases"))     O.phases = 1;
        else if (!strcmp(a, "--verbose"))    O.verbose = 1;
        else if (!strcmp(a, "--device-init")) O.device_init = 1;
        else if (!strcmp(a, "--no-friction")) O.apply_forcing = 0;
        else if (!strcmp(a, "--cuda-extrap")) {
            g_cuda_extrap_tpb = (int)arg_i(argc, argv, &i, a);
#ifdef ANUGA_GEOM_FP32
            fprintf(stderr, "bench: --cuda-extrap is fp64-geometry only\n");
            return 2;
#endif
        }
        else if (!strcmp(a, "--active-set"))  g_active_set = 1;
        else if (!strcmp(a, "--active-every")) g_active_every = (int)arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--active-verify")) g_active_verify = 1;
        else if (!strcmp(a, "--rain"))        g_rain_mmhr = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--rain-every"))  g_rain_every = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--rain-for"))    g_rain_for = arg_d(argc, argv, &i, a);
        else if (!strcmp(a, "--rain-band")) { g_rain_x0 = arg_d(argc, argv, &i, a);
                                              g_rain_x1 = arg_d(argc, argv, &i, a); }
        else if (!strcmp(a, "--rain-grid"))   g_rain_grid_path = arg_s(argc, argv, &i, a);
        else if (!strcmp(a, "--order")) {
            const char *o = arg_s(argc, argv, &i, a);
            if      (!strcmp(o, "row"))    O.morton = 0;
            else if (!strcmp(o, "morton")) O.morton = 1;
            else if (!strcmp(o, "random")) O.morton = 2;
            else { fprintf(stderr, "bench: unknown order '%s'\n", o); return 2; }
        }
        else if (!strcmp(a, "--case")) {
            const char *c = arg_s(argc, argv, &i, a);
            if      (!strcmp(c, "dam"))      P.which_case = BENCH_CASE_DAM;
            else if (!strcmp(c, "dambumps")) P.which_case = BENCH_CASE_DAMBUMPS;
            else if (!strcmp(c, "lake"))     P.which_case = BENCH_CASE_LAKE;
            else if (!strcmp(c, "river"))    P.which_case = BENCH_CASE_RIVER;
            else { fprintf(stderr, "bench: unknown case '%s'\n", c); return 2; }
        }
        else if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "bench: unknown option '%s' (try --help)\n", a); return 2; }
    }

    if (scheme_set) {
        const double user_cfl = P.cfl;
        bench_params_apply_scheme(&P);
        if (cfl_set) P.cfl = user_cfl;          // explicit --cfl beats the preset
    }
    if (betas_override >= 0.0) {
        P.beta_w = P.beta_uh = P.beta_vh = betas_override;
        P.beta_w_dry = P.beta_uh_dry = P.beta_vh_dry = 0.0;
    }

    // ---- build -----------------------------------------------------------
    bench_mesh M;
    bench_slab S = {0};
    bench_tiles T = {0};
    const int64_t *full_flag = NULL;   // ownership under MPI (NULL = no ghosts)
    int64_t n_full = 0, n_global = 0;
    double *bed_node = NULL, *stage_node = NULL;
    if (O.tiles_path) {
        // Tiled distributed mesh: works for any np (np=1 is the stitching
        // check against the merged mesh).  Same kernel restrictions as the
        // slab path.
        if (O.mesh_path || O.morton != 0 || P.flux_mode == 1 || g_cuda_extrap_tpb > 0) {
            if (g_rank == 0)
                fprintf(stderr, "bench: --tiles excludes --mesh/--order/--flux edge/--cuda-extrap\n");
            return 2;
        }
        bench_mesh_load_tiles(O.tiles_path, O.assign_path, g_rank, g_np,
                              &M, &bed_node, &stage_node, &T);
        if (g_np > 1) full_flag = T.tri_full_flag;
        n_full = T.n_own; n_global = T.n_global;
    } else if (g_np > 1) {
        // MPI restrictions: generated rectangular-cross mesh in row order (the
        // slab cut relies on cell id = i*n + j), no --cuda-extrap.  Cell and
        // scatter fluxes, --phases and --active-set all work (see
        // rk2_step_timed for where the halo exchange and dt reduction go).
        if (O.mesh_path || O.morton != 0 || P.flux_mode == 1 ||
            g_cuda_extrap_tpb > 0) {
            if (g_rank == 0)
                fprintf(stderr, "bench: MPI runs support the generated mesh with "
                        "--order row, --flux cell|scatter, no --cuda-extrap\n");
            return 2;
        }
        bench_mesh_rectangular_cross_slab(O.nx, O.ny, P.length_x, P.length_y,
                                          g_rank, g_np, &M, &S);
        full_flag = S.tri_full_flag; n_full = S.n_full; n_global = 4 * O.nx * O.ny;
    } else if (O.mesh_path) {
        bench_mesh_load(O.mesh_path, &M, &bed_node, &stage_node);
        if (O.morton == 1)      bench_mesh_reorder_tris_morton(&M);
        else if (O.morton == 2) bench_mesh_reorder_tris_random(&M);
    } else {
        bench_mesh_rectangular_cross(O.nx, O.ny, P.length_x, P.length_y, 0.0, 0.0, &M);
        if (O.morton == 1)      bench_mesh_reorder_morton(&M, O.nx, O.ny);
        else if (O.morton == 2) bench_mesh_reorder_random(&M, O.nx, O.ny);
    }

    if (O.device_init &&
        (O.mesh_path || O.tiles_path || g_np > 1 || O.morton != 0 ||
         g_cuda_extrap_tpb > 0)) {
        if (g_rank == 0)
            fprintf(stderr, "bench: --device-init supports the serial generated "
                    "mesh in row order only (no --mesh/--tiles/MPI/--order/"
                    "--cuda-extrap yet)\n");
        return 2;
    }

    bench_domain B;
    const double t_build0 = omp_get_wtime();
    bench_domain_build(&B, &M, &P, bed_node, stage_node,
                       (const anuga_int *)full_flag, !O.device_init);
    const double t_build = omp_get_wtime() - t_build0;


    const double t_map0 = omp_get_wtime();
    if (O.device_init) {
        if (bench_device_to_device(&B, &P, O.nx, O.ny, O.verbose, g_rank, g_np) != 0)
            return 2;
    } else {
        bench_domain_to_device(&B, &P, O.verbose, g_rank, g_np);
    }
    const double t_map = omp_get_wtime() - t_map0;

    // Halo exchange setup: with the slab cut along the first grid axis and
    // cell id = i*n + j, each column's 4*ny triangles are one contiguous id
    // range, so every send/recv list is a simple run of indices.
    if (g_np > 1 && O.tiles_path) {
        // Lists from the tile stitching; a rank with no foreign neighbour
        // tile simply has no halo (exchange is then a no-op).
        if (gpu_halo_init(&B.GD, T.nnb, T.nb_ranks, T.scnt, T.rcnt, T.fsend, T.frecv) != 0) {
            fprintf(stderr, "bench: gpu_halo_init failed on rank %d\n", g_rank);
            return 2;
        }
        gpu_exchange_ghosts(&B.GD);
    } else if (g_np > 1) {
        int nb_ranks[2], scnt[2], rcnt[2], nnb = 0;
        const int per_col = (int)(4 * O.ny);
        int *fs = (int *)malloc(2 * (size_t)per_col * sizeof(int));
        int *fr = (int *)malloc(2 * (size_t)per_col * sizeof(int));
        int so = 0, ro = 0;
        if (S.gl) {   // lower neighbour: send first owned column, recv ghost col 0
            nb_ranks[nnb] = g_rank - 1; scnt[nnb] = rcnt[nnb] = per_col;
            for (int e = 0; e < per_col; e++) {
                fs[so + e] = (int)(4 * S.gl * O.ny) + e;
                fr[ro + e] = e;
            }
            so += per_col; ro += per_col; nnb++;
        }
        if (S.gh) {   // upper neighbour: send last owned column, recv last col
            nb_ranks[nnb] = g_rank + 1; scnt[nnb] = rcnt[nnb] = per_col;
            const int send_base = (int)(4 * (S.gl + (S.i1 - S.i0) - 1) * O.ny);
            const int recv_base = (int)(4 * (S.m_local - 1) * O.ny);
            for (int e = 0; e < per_col; e++) {
                fs[so + e] = send_base + e;
                fr[ro + e] = recv_base + e;
            }
            so += per_col; ro += per_col; nnb++;
        }
        if (gpu_halo_init(&B.GD, nnb, nb_ranks, scnt, rcnt, fs, fr) != 0) {
            fprintf(stderr, "bench: gpu_halo_init failed on rank %d\n", g_rank);
            return 2;
        }
        free(fs); free(fr);
        // Ghost columns start identical on both owners (same generator), but
        // exchange once so any roundoff asymmetry is settled before stepping.
        gpu_exchange_ghosts(&B.GD);
    }

    struct gpu_domain *GD = &B.GD;
    const int64_t n = GD->D.number_of_elements;
    const int64_t n_report = (g_np > 1 || O.tiles_path) ? n_global : n;

    const char *case_name = P.which_case == BENCH_CASE_DAM      ? "dam"
                          : P.which_case == BENCH_CASE_DAMBUMPS ? "dambumps"
                          : P.which_case == BENCH_CASE_RIVER    ? "river" : "lake";
#ifdef CPU_ONLY_MODE
    const char *build_kind = "host OpenMP (CPU_ONLY_MODE)";
#else
    const char *build_kind = "OpenMP target offload";
#endif

    printf("ANUGA shallow-water miniapp -- %s\n", build_kind);
    if (O.tiles_path)
        printf("  mesh      : %s -> %d tiles, %lld triangles (this rank: %d tiles, "
               "%lld owned + %lld ghost, %d neighbour rank(s))\n",
               O.tiles_path, T.ntiles, (long long)n_global, T.ntiles_own,
               (long long)T.n_own, (long long)(n - T.n_own), T.nnb);
    else if (O.mesh_path)
        printf("  mesh      : %s -> %lld triangles, %lld boundary edges\n",
               O.mesh_path, (long long)n, (long long)GD->D.boundary_length);
    else
        printf("  mesh      : %lld x %lld cross -> %lld triangles, %lld boundary edges\n",
               (long long)O.nx, (long long)O.ny, (long long)n_report,
               (long long)(g_np > 1 ? 2 * (O.nx + O.ny) : GD->D.boundary_length));
    if (g_np > 1 && O.tiles_path)
        printf("  mpi       : %d ranks, tile assignment%s\n", g_np,
               O.assign_path ? O.assign_path : " (contiguous, balanced by triangle count)");
    else if (g_np > 1)
        printf("  mpi       : %d ranks, slab cut along x; this rank owns columns "
               "[%lld, %lld) + %lld ghost col(s), %lld local triangles\n",
               g_np, (long long)S.i0, (long long)S.i1,
               (long long)(S.gl + S.gh), (long long)n);
    printf("  case      : %s, manning %.4g%s\n",
           (O.mesh_path || O.tiles_path) ? "from mesh file" : case_name, P.manning,
           O.apply_forcing ? "" : " (friction off)");
    {
        const char *sn = P.scheme == BENCH_SCHEME_ADER2 ? "ader2 (DE_ader2)"
                       : P.scheme == BENCH_SCHEME_EULER ? "euler (DE0)"
                       : P.scheme == BENCH_SCHEME_RK3   ? "rk3 (DE2)"
                       : "rk2 (DE1)";
        printf("  scheme    : %s, CFL %.3g, betas %.3g, %s-based fluxes\n",
               sn, P.cfl, P.beta_w,
               P.flux_mode == 1 ? "edge" : P.flux_mode == 2 ? "scatter" : "cell");
    }
    printf("  ordering  : %s\n",
           O.morton == 1 ? "morton (Z-order curve)"
         : O.morton == 2 ? "random (pessimistic locality bound)"
         : "row-major (ANUGA rectangular_cross)");
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

    const double volume0 = (g_np > 1)
        ? bmpi_sum_d(host_water_volume(GD, full_flag))
        : gpu_compute_water_volume(GD);

    if (g_cuda_extrap_tpb > 0) {
        O.phases = 1;                          // route through the stepped loops
        cuda_extrap_load();
    }
    if (g_rain_grid_path) rain_grid_load(GD);
    if (g_active_set) {
        if (P.flux_mode != 2) {
            fprintf(stderr, "bench: --active-set requires --flux scatter\n");
            return 2;
        }
        O.phases = 1;                          // route through the stepped loops
        g_as_wet   = (anuga_int *)calloc((size_t)n, sizeof(anuga_int));
        g_as_ring1 = (anuga_int *)calloc((size_t)n, sizeof(anuga_int));
        g_as_cells = (anuga_int *)calloc((size_t)n, sizeof(anuga_int));
        g_as_edges = (anuga_int *)calloc((size_t)GD->D.num_owned_edges, sizeof(anuga_int));
        {
            anuga_int *w = g_as_wet, *r1 = g_as_ring1, *c = g_as_cells, *e = g_as_edges;
            const anuga_int ne = GD->D.num_owned_edges;
            #pragma omp target enter data map(alloc: w[0:n], r1[0:n], c[0:n], e[0:ne])
        }
        if (g_active_every > 1) {
            if (g_rain_mmhr > 0.0 || g_rain_grid_path) {
                // Rain wets cells the candidate cannot predict: it adds stage
                // wherever it likes, so a rained cell outside the superset
                // would go unnoticed for up to K steps.  (Widespread rain
                // also activates the whole mesh, so there is nothing to win.)
                fprintf(stderr, "bench: --active-every > 1 is incompatible with rain\n");
                return 2;
            }
            g_cand      = (anuga_int *)calloc((size_t)n, sizeof(anuga_int));
            g_cand_flag = (anuga_int *)calloc((size_t)n, sizeof(anuga_int));
            g_cand_edges = (anuga_int *)calloc((size_t)GD->D.num_owned_edges,
                                               sizeof(anuga_int));
            anuga_int *cd = g_cand, *cf = g_cand_flag, *ce = g_cand_edges;
            const anuga_int noe = GD->D.num_owned_edges;
            #pragma omp target enter data map(alloc: cd[0:n], cf[0:n], ce[0:noe])
            #pragma omp target update to(cf[0:n])
        }
        if (O.tile_stats_path) {
            if (!O.tiles_path) {
                fprintf(stderr, "bench: --tile-stats needs --tiles\n");
                return 2;
            }
            g_act_count = (int *)calloc((size_t)n, sizeof(int));
            int *cnt = g_act_count;
            #pragma omp target enter data map(to: cnt[0:n])
        }
    } else if (O.tile_stats_path) {
        fprintf(stderr, "bench: --tile-stats needs --active-set\n");
        return 2;
    }

    // ---- warmup ----------------------------------------------------------
    // ADER2 carries the previous step's dt into the C-K predictor (0.0 on the
    // very first call = plain Euler bootstrap), so it threads through warmup
    // and the timed loop alike.
    double t_sim = 0.0, dt = 0.0;
    for (int64_t s = 0; s < O.warmup; s++) {
        switch (P.scheme) {
            case BENCH_SCHEME_ADER2:
                dt = gpu_evolve_one_ader2_step(GD, P.evolve_max_timestep, O.apply_forcing, dt);
                break;
            case BENCH_SCHEME_EULER:
                dt = gpu_evolve_one_euler_step(GD, P.evolve_max_timestep, O.apply_forcing);
                break;
            case BENCH_SCHEME_RK3:
                dt = gpu_evolve_one_rk3_step(GD, P.evolve_max_timestep, O.apply_forcing);
                break;
            default:
                dt = gpu_evolve_one_rk2_step(GD, P.evolve_max_timestep, O.apply_forcing);
        }
        apply_rain(GD, t_sim, dt);
        if (g_np > 1) gpu_exchange_ghosts(GD);   // end-of-step ghost sync
        t_sim += dt;
    }

    // ---- timed loop ------------------------------------------------------
    double best = 1.0e300, total_all = 0.0;
    uint64_t flops_total = 0;

    for (int r = 0; r < O.repeat; r++) {
        memset(phase_time, 0, sizeof(phase_time));
        gpu_flop_counters_reset(GD);
        gpu_flop_counters_enable(GD, 1);

        const double t0 = omp_get_wtime();
        double win_t0 = t0;
        int64_t win_s0 = 0;
        int64_t steps_done = 0;
        for (int64_t s = 0; s < O.steps; s++) {
            switch (P.scheme) {
                case BENCH_SCHEME_ADER2:
                    dt = O.phases ? ader2_step_timed(GD, P.evolve_max_timestep, O.apply_forcing, dt)
                                  : gpu_evolve_one_ader2_step(GD, P.evolve_max_timestep, O.apply_forcing, dt);
                    break;
                case BENCH_SCHEME_EULER:
                    dt = gpu_evolve_one_euler_step(GD, P.evolve_max_timestep, O.apply_forcing);
                    break;
                case BENCH_SCHEME_RK3:
                    dt = gpu_evolve_one_rk3_step(GD, P.evolve_max_timestep, O.apply_forcing);
                    break;
                default:
                    dt = O.phases ? rk2_step_timed(GD, P.evolve_max_timestep, O.apply_forcing)
                                  : gpu_evolve_one_rk2_step(GD, P.evolve_max_timestep, O.apply_forcing);
            }
            apply_rain(GD, t_sim, dt);
            if (g_np > 1) gpu_exchange_ghosts(GD);   // end-of-step ghost sync
            t_sim += dt;
            steps_done = s + 1;

            if (O.report_every > 0 && (s + 1) % O.report_every == 0) {
                const double now = omp_get_wtime();
                const double win_ms = 1.0e3 * (now - win_t0) / (double)(s + 1 - win_s0);
                printf("  @step %8lld  t=%10.2f s  dt=%.5f  %8.3f ms/step  active=%6.2f%%  simrate=%.3f\n",
                       (long long)(s + 1), t_sim, dt, win_ms,
                       g_as_samples > 0 ? 100.0 * g_as_cellfrac_sum / (double)g_as_samples : 100.0,
                       dt / (win_ms / 1.0e3));
                fflush(stdout);
                win_t0 = now; win_s0 = s + 1;
                g_as_cellfrac_sum = 0.0; g_as_samples = 0;   // window-local stats
            }
            if (O.max_wall > 0.0 && omp_get_wtime() - t0 > O.max_wall) break;
        }
        const double elapsed = omp_get_wtime() - t0;
        if (steps_done < O.steps) O.steps = steps_done;   // honest averages below

        gpu_flop_counters_enable(GD, 0);
        flops_total = gpu_flop_counters_get_total(GD);
        total_all += elapsed;
        if (elapsed < best) best = elapsed;
        if (O.repeat > 1)
            printf("  run %2d/%d : %8.4f s  (%.4f ms/step)\n",
                   r + 1, O.repeat, elapsed, 1.0e3 * elapsed / (double)O.steps);
    }

    best = bmpi_max_d(best);          // slowest rank is the honest wall time
    const double per_step = best / (double)O.steps;
    const double cellsteps_per_s = (double)n_report * (double)O.steps / best;

    printf("\n  timed     : %lld steps (+%lld warmup) in %.4f s%s\n",
           (long long)O.steps, (long long)O.warmup, best,
           O.repeat > 1 ? " (best of runs)" : "");
    printf("              %.4f ms/step, %.3f Mcell-steps/s\n",
           1.0e3 * per_step, 1.0e-6 * cellsteps_per_s);
    printf("              t = %.9g s, last dt = %.6g s\n", t_sim, dt);
    printf("              sim rate: %.3f simulated s per wall s (this loop)\n",
           (double)O.steps * dt / best);
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
        if (g_np > 1) {
            // Kernel time excludes the halo/allreduce waits, so its spread
            // across ranks is the load imbalance (wall time is the max).
            // One stderr line per rank with what the cost model needs.
            fprintf(stderr, "rank %d: %lld owned, %lld ghost, %.0f active cells/step, %.4f ms/step kernels\n",
                    g_rank, (long long)n_full, (long long)(n - n_full),
                    g_as_samples > 0 ? g_as_cellfrac_sum / (double)g_as_samples * (double)n : (double)n,
                    1.0e3 * summed / (double)O.steps);
            const double smax = bmpi_max_d(summed), smin = -bmpi_max_d(-summed);
            printf("    per-rank kernel time %.4f .. %.4f ms/step  (imbalance %.1f%%)\n",
                   1.0e3 * smin / (double)O.steps, 1.0e3 * smax / (double)O.steps,
                   100.0 * (smax / smin - 1.0));
        }
    }

    if (g_act_count && g_as_samples > 0) {
        // Per owned tile: mean active fraction over the rebuilds.  Ranks write
        // in turn so one file holds every tile (index order within a rank).
        int *cnt = g_act_count;
        #pragma omp target update from(cnt[0:n])
        for (int r = 0; r < g_np; r++) {
            if (r == g_rank) {
                FILE *fp = fopen(O.tile_stats_path, r == 0 ? "w" : "a");
                if (!fp) { perror(O.tile_stats_path); return 2; }
                if (r == 0) fprintf(fp, "# tile ntris active_fraction rank  (%ld rebuilds)\n", g_as_samples);
                for (int t = 0; t < T.ntiles_own; t++) {
                    double sum = 0.0;
                    for (int64_t k = T.own_tile_start[t]; k < T.own_tile_start[t + 1]; k++) sum += cnt[k];
                    const int64_t nt = T.own_tile_start[t + 1] - T.own_tile_start[t];
                    fprintf(fp, "%d %lld %.6f %d\n", T.own_tile_id[t], (long long)nt,
                            nt > 0 ? sum / ((double)nt * (double)g_as_samples) : 0.0, r);
                }
                fclose(fp);
            }
            bmpi_barrier();
        }
    }
    if (g_active_set && g_as_samples > 0) {
        // Each rank's fraction is over its local cells; under MPI report the
        // rank average and the spread -- the slowest rank is the one with the
        // most active cells, so max/min is the load imbalance in one number.
        const double mine = g_as_cellfrac_sum / (double)g_as_samples;
        const double frac = bmpi_sum_d(mine) / g_np;
        if (g_cand_rebuilds > 0)
            printf("  candidate : %.2f%% of cells, every %d steps (%ld full rebuilds)\n",
                   100.0 * g_cand_frac_sum / (double)g_cand_rebuilds,
                   g_active_every, g_cand_rebuilds);
        printf("  active    : %.2f%% of cells on average (%ld rebuilds)",
               100.0 * frac, g_as_samples);
        if (g_np > 1)
            printf(", per rank %.2f%% .. %.2f%%",
                   100.0 * -bmpi_max_d(-mine), 100.0 * bmpi_max_d(mine));
        printf("\n");
    }

    // ---- diagnostics -----------------------------------------------------
    gpu_domain_sync_from_device(GD);
    const double volume1 = (g_np > 1)
        ? bmpi_sum_d(host_water_volume(GD, full_flag))
        : gpu_compute_water_volume(GD);
    printf("\n  volume    : %.12g -> %.12g m^3 (drift %.3e relative)\n",
           volume0, volume1, volume0 != 0.0 ? (volume1 - volume0) / volume0 : 0.0);

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
    if (g_np > 1) {
        max_stage    = bmpi_max_d(max_stage);
        min_stage    = -bmpi_max_d(-min_stage);
        max_speed_sq = bmpi_max_d(max_speed_sq);
        nan_count    = bmpi_max_i(nan_count);
    }
    printf("  state     : stage in [%.6g, %.6g], max |momentum| %.6e%s\n",
           min_stage, max_stage, sqrt(max_speed_sq),
           nan_count ? "  *** NaNs present ***" : "");
    if (P.which_case == BENCH_CASE_LAKE)
        printf("  lake test : water started at rest; max |momentum| above should stay ~0\n");

    int rc = nan_count ? 1 : 0;
    const int64_t total_steps = O.warmup + O.steps * O.repeat;

    if (g_np > 1) {
        if (O.save_path || O.check_path)
            rc |= mpi_snapshot(O.save_path, O.check_path, GD, &M,
                               full_flag, n_full, n_global,
                               O.nx, O.ny, (int)P.which_case,
                               total_steps, t_sim, dt, O.rtol, O.atol);
    } else {
        if (O.save_path)
            rc |= snapshot_save(O.save_path, GD, M.orig_id, O.nx, O.ny, (int)P.which_case,
                                total_steps, t_sim, dt);
        if (O.check_path)
            rc |= snapshot_check(O.check_path, GD, M.orig_id, O.rtol, O.atol);
    }

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
                    (long long)O.nx, (long long)O.ny, (long long)n_report, case_name,
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
    if (O.tiles_path) bench_tiles_free(&T);
    free(bed_node);
    free(stage_node);
    return rc;
}

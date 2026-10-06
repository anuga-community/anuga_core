// Device-side domain initialisation for generated rectangular-cross meshes.
//
// The closed-form twin of the host build in setup.c: every O(n) array the
// kernels consume is mapped with map(alloc:) and filled by target kernels,
// so the host never writes (or transfers) a byte of the big arrays -- their
// calloc'd mirrors stay untouched virtual pages until --save faults a few in.
//
// Everything here reproduces the host path EXACTLY:
//   - node coordinates use the generator's expressions verbatim
//     (delta1 * (double)i + x0, centre = 0.25 * (v1+v2+v3+v4) in that order);
//   - the geometry and quantity loop bodies are copied from setup.c, and the
//     case functions are the shared declare-target ones in setup_cases.h;
//   - connectivity comes from the closed form of the cross pattern
//     (derived from rectangular_cross_core + the edgemap rule):
//         cell c = k/4, role t = k%4  (t0 left, t1 bottom, t2 right, t3 top)
//         edge 0 -> same cell, tri (t+3)%4, neighbour edge 2
//         edge 2 -> same cell, tri (t+1)%4, neighbour edge 0
//         edge 1 -> the e1 of the facing tri in the adjacent cell
//                   (t0 <-> t2 across i, t1 <-> t3 across j), or the rim;
//   - boundary numbering and the owned-edge list use deterministic
//     block prefix scans, reproducing the host's k-ascending order.
//
// Scope: serial, generated mesh, row order.  --mesh/--tiles/MPI/morton keep
// the host build (bench.c enforces this).

#include "setup.h"
#include "setup_cases.h"
#include "gpu_omp_macros.h"   // OMP_PARALLEL_LOOP: target teams loop on GPU,
                              // parallel for on the CPU-only build

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SCAN_BLK 32768

// The geometry kernel needs reduction + firstprivate(PP) together; nvc 25.9
// miscompiles the OMP_PARALLEL_LOOP_REDUCTION_PLUS expansion (distribute
// parallel for + implicit struct map) here, so pin the exact construct that
// the rest of the kernel set uses.
#ifdef CPU_ONLY_MODE
#define DI_GEOM_LOOP _Pragma("omp parallel for reduction(+:bad_areas)")
#elif defined(ANUGA_OMP_TEAMS_DPF)   // amdclang: `teams loop` runs ~10x slower
#define DI_GEOM_LOOP _Pragma("omp target teams distribute parallel for reduction(+:bad_areas) firstprivate(PP)")
#else
#define DI_GEOM_LOOP _Pragma("omp target teams loop reduction(+:bad_areas) firstprivate(PP)")
#endif

static void *dxmalloc(size_t bytes) {
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "bench: out of memory (%zu bytes)\n", bytes); exit(1); }
    return p;
}

int bench_device_to_device(bench_domain *B, const bench_params *P,
                           int64_t mx, int64_t my,
                           int verbose, int rank, int nprocs) {
    struct gpu_domain *GD = &B->GD;
    struct domain *D = &GD->D;
    const anuga_int n  = D->number_of_elements;
    const anuga_int nb = D->boundary_length;

    if (n != 4 * mx * my || nb != 2 * (mx + my)) {
        fprintf(stderr, "bench: device init size mismatch (n=%lld, nb=%lld)\n",
                (long long)n, (long long)nb);
        return 1;
    }

    // Device selection first: every mapping below must land on the right GPU.
    gpu_domain_init(GD, MPI_COMM_WORLD, rank, nprocs);
    GD->verbose             = verbose;
    GD->CFL                 = P->cfl;
    GD->evolve_max_timestep = P->evolve_max_timestep;
    GD->fixed_flux_timestep = -1.0;
    GD->use_sloped_mannings = P->use_sloped_mannings;

    // Generator scalars -- the exact expressions rectangular_cross_core uses.
    const double delta1 = P->length_x / (double)mx;
    const double delta2 = P->length_y / (double)my;
    const bench_params PP = *P;   // by-value copy for firstprivate use in kernels

    // ---- local pointer extraction (target pragmas can't take D->member) ----
    double *stage_cv = D->stage_centroid_values;
    double *xmom_cv  = D->xmom_centroid_values;
    double *ymom_cv  = D->ymom_centroid_values;
    double *bed_cv   = D->bed_centroid_values;
    double *height_cv = D->height_centroid_values;
    double *friction_cv = D->friction_centroid_values;
    double *stage_ev = D->stage_edge_values;
    double *xmom_ev  = D->xmom_edge_values;
    double *ymom_ev  = D->ymom_edge_values;
    double *bed_ev   = D->bed_edge_values;
    double *height_ev = D->height_edge_values;
    double *stage_eu = D->stage_explicit_update;
    double *xmom_eu  = D->xmom_explicit_update;
    double *ymom_eu  = D->ymom_explicit_update;
    double *stage_siu = D->stage_semi_implicit_update;
    double *xmom_siu  = D->xmom_semi_implicit_update;
    double *ymom_siu  = D->ymom_semi_implicit_update;
    anuga_int *neighbours = D->neighbours;
    anuga_int *neighbour_edges = D->neighbour_edges;
    anuga_int *surrogate_neighbours = D->surrogate_neighbours;
    anuga_int *number_of_boundaries = D->number_of_boundaries;
    anuga_geom_t *normals = D->normals;
    anuga_geom_t *edgelengths = D->edgelengths;
    anuga_geom_t *areas = D->areas;
    anuga_geom_t *radii = D->radii;
    anuga_geom_t *centroid_coords = D->centroid_coordinates;
    anuga_geom_t *edge_coords = D->edge_coordinates;
    double *max_speed = D->max_speed;
    double *x_centroid_work = D->x_centroid_work;
    double *y_centroid_work = D->y_centroid_work;
    double *stage_backup = D->stage_backup_values;
    double *xmom_backup  = D->xmom_backup_values;
    double *ymom_backup  = D->ymom_backup_values;
    double *stage_bv = D->stage_boundary_values;
    double *xmom_bv  = D->xmom_boundary_values;
    double *ymom_bv  = D->ymom_boundary_values;
    double *bed_bv   = D->bed_boundary_values;
    double *height_bv = D->height_boundary_values;

    // ---- premap: the exact surface gpu_domain_map_arrays would map, as
    // alloc instead of to.  (Serial bench: no tri_full_flag, no halo, no
    // riverwalls.)  gpu_initialized is set at the end so map_arrays -- and
    // its free-memory check, which would double-count this -- never runs.
    #pragma omp target enter data map(alloc: \
        stage_cv[0:n], xmom_cv[0:n], ymom_cv[0:n], \
        bed_cv[0:n], height_cv[0:n], friction_cv[0:n], \
        stage_ev[0:3*n], xmom_ev[0:3*n], ymom_ev[0:3*n], \
        bed_ev[0:3*n], height_ev[0:3*n], \
        stage_eu[0:n], xmom_eu[0:n], ymom_eu[0:n], \
        stage_siu[0:n], xmom_siu[0:n], ymom_siu[0:n], \
        neighbours[0:3*n], neighbour_edges[0:3*n], \
        surrogate_neighbours[0:3*n], number_of_boundaries[0:n], \
        x_centroid_work[0:n], y_centroid_work[0:n], \
        normals[0:6*n], edgelengths[0:3*n], \
        areas[0:n], radii[0:n], max_speed[0:n], \
        centroid_coords[0:2*n], edge_coords[0:6*n])
    #pragma omp target enter data map(alloc: \
        stage_bv[0:nb], xmom_bv[0:nb], ymom_bv[0:nb], \
        bed_bv[0:nb], height_bv[0:nb])
    #pragma omp target enter data map(alloc: \
        stage_backup[0:n], xmom_backup[0:n], ymom_backup[0:n])

    // ---- geometry + quantities ------------------------------------------
    // One pass: vertex coordinates come straight from the closed form, then
    // the geometry and quantity bodies are the setup.c loops verbatim.
    anuga_int bad_areas = 0;
    DI_GEOM_LOOP
    for (anuga_int k = 0; k < n; k++) {
        const anuga_int c = k / 4;
        const int       t = (int)(k % 4);
        const anuga_int i = c / my;
        const anuga_int j = c % my;

        // Grid-node coordinates, exactly as rectangular_cross_core writes them.
        const double xA = delta1 * (double)i,       yA = delta2 * (double)j;        // (i,   j)
        const double xB = delta1 * (double)i,       yB = delta2 * (double)(j + 1);  // (i,   j+1)
        const double xD = delta1 * (double)(i + 1), yD = delta2 * (double)(j + 1);  // (i+1, j+1)
        const double xE = delta1 * (double)(i + 1), yE = delta2 * (double)j;        // (i+1, j)
        // Centre node: 0.25 * (v1 + v2 + v3 + v4) in the generator's order
        // v1=(i,j+1), v2=(i,j), v3=(i+1,j+1), v4=(i+1,j).
        const double xC = 0.25 * (xB + xA + xD + xE);
        const double yC = 0.25 * (yB + yA + yD + yE);

        // Triangle vertices in the generator's (v0, v1, v2) order:
        //   t0 [A C B]   t1 [E C A]   t2 [D C E]   t3 [B C D]
        double x0, y0, x1, y1, x2, y2;
        x1 = xC; y1 = yC;
        switch (t) {
            case 0:  x0 = xA; y0 = yA; x2 = xB; y2 = yB; break;
            case 1:  x0 = xE; y0 = yE; x2 = xA; y2 = yA; break;
            case 2:  x0 = xD; y0 = yD; x2 = xE; y2 = yE; break;
            default: x0 = xB; y0 = yB; x2 = xD; y2 = yD; break;
        }

        // --- geometry: setup.c bodies verbatim ---
        const double area =
            -((x1 * y0 - x0 * y1) + (x2 * y1 - x1 * y2) + (x0 * y2 - x2 * y0)) / 2.0;
        areas[k] = area;
        if (!(area > 0.0)) bad_areas++;

        double xn0 = x2 - x1, yn0 = y2 - y1;
        double xn1 = x0 - x2, yn1 = y0 - y2;
        double xn2 = x1 - x0, yn2 = y1 - y0;
        const double l0 = sqrt(xn0 * xn0 + yn0 * yn0);
        const double l1 = sqrt(xn1 * xn1 + yn1 * yn1);
        const double l2 = sqrt(xn2 * xn2 + yn2 * yn2);
        xn0 /= l0; yn0 /= l0;
        xn1 /= l1; yn1 /= l1;
        xn2 /= l2; yn2 /= l2;

        normals[6 * k + 0] =  yn0;  normals[6 * k + 1] = -xn0;
        normals[6 * k + 2] =  yn1;  normals[6 * k + 3] = -xn1;
        normals[6 * k + 4] =  yn2;  normals[6 * k + 5] = -xn2;

        edgelengths[3 * k + 0] = l0;
        edgelengths[3 * k + 1] = l1;
        edgelengths[3 * k + 2] = l2;

        const double cxk = (x0 + x1 + x2) / 3.0;
        const double cyk = (y0 + y1 + y2) / 3.0;
        centroid_coords[2 * k + 0] = cxk;
        centroid_coords[2 * k + 1] = cyk;

        const double xm0 = 0.5 * (x1 + x2), ym0 = 0.5 * (y1 + y2);
        const double xm1 = 0.5 * (x2 + x0), ym1 = 0.5 * (y2 + y0);
        const double xm2 = 0.5 * (x0 + x1), ym2 = 0.5 * (y0 + y1);
        edge_coords[6 * k + 0] = xm0;  edge_coords[6 * k + 1] = ym0;
        edge_coords[6 * k + 2] = xm1;  edge_coords[6 * k + 3] = ym1;
        edge_coords[6 * k + 4] = xm2;  edge_coords[6 * k + 5] = ym2;

        const double d0 = hypot(cxk - xm0, cyk - ym0);
        const double d1 = hypot(cxk - xm1, cyk - ym1);
        const double d2 = hypot(cxk - xm2, cyk - ym2);
        radii[k] = fmin(fmin(d0, d1), d2);

        // --- quantities: the generated-mesh (bed_node == NULL) branch of
        // the setup.c loop, verbatim.  Vertex arrays are host-only and
        // unmapped, so their writes are dropped.
        double zv[3];
        const double vx[3] = { x0, x1, x2 };
        const double vy[3] = { y0, y1, y2 };
        for (int e = 0; e < 3; e++)
            zv[e] = bench_bed_value(&PP, vx[e], vy[e]);

        for (int e = 0; e < 3; e++)
            bed_ev[3 * k + e] = 0.5 * (zv[(e + 1) % 3] + zv[(e + 2) % 3]);
        bed_cv[k] = (zv[0] + zv[1] + zv[2]) / 3.0;

        const double zc = bed_cv[k];
        // The host reads the centroid through the (possibly float) geometry
        // array; do the same so both paths round identically.
        const double w = bench_stage_value(&PP, (double)centroid_coords[2 * k + 0],
                                                (double)centroid_coords[2 * k + 1], zc);
        for (int e = 0; e < 3; e++) {
            stage_ev[3 * k + e]  = w;
            height_ev[3 * k + e] = fmax(w - bed_ev[3 * k + e], 0.0);
        }

        stage_cv[k]    = w;
        xmom_cv[k]     = 0.0;
        ymom_cv[k]     = 0.0;
        height_cv[k]   = fmax(w - zc, 0.0);
        friction_cv[k] = PP.manning;

        // --- zero-init the state the host path callocs ---
        stage_eu[k] = 0.0;  xmom_eu[k] = 0.0;  ymom_eu[k] = 0.0;
        stage_siu[k] = 0.0; xmom_siu[k] = 0.0; ymom_siu[k] = 0.0;
        x_centroid_work[k] = 0.0; y_centroid_work[k] = 0.0;
        max_speed[k] = 0.0;
        stage_backup[k] = 0.0; xmom_backup[k] = 0.0; ymom_backup[k] = 0.0;
        xmom_ev[3 * k + 0] = 0.0; xmom_ev[3 * k + 1] = 0.0; xmom_ev[3 * k + 2] = 0.0;
        ymom_ev[3 * k + 0] = 0.0; ymom_ev[3 * k + 1] = 0.0; ymom_ev[3 * k + 2] = 0.0;
    }
    // The bed never changes, so gpu_domain_sync_from_device never pulls it --
    // on the host path it was born on the host.  Snapshots (--save/--check)
    // read the host mirror, so sync it once here (8 B/tri, the one host-side
    // O(n) cost of this path).
    #pragma omp target update from(bed_cv[0:n])

    if (bad_areas > 0) {
        fprintf(stderr, "bench: device init produced %lld non-positive areas\n",
                (long long)bad_areas);
        return 1;
    }

    OMP_PARALLEL_LOOP
    for (anuga_int b = 0; b < nb; b++) {
        stage_bv[b] = 0.0; xmom_bv[b] = 0.0; ymom_bv[b] = 0.0;
        bed_bv[b] = 0.0; height_bv[b] = 0.0;
    }

    // ---- connectivity: the closed form -----------------------------------
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++) {
        const anuga_int c = k / 4;
        const int       t = (int)(k % 4);
        const anuga_int i = c / my;
        const anuga_int j = c % my;

        // e0 / e2: the intra-cell ring around the centre vertex.
        neighbours[3 * k + 0]      = 4 * c + (anuga_int)((t + 3) % 4);
        neighbour_edges[3 * k + 0] = 2;
        neighbours[3 * k + 2]      = 4 * c + (anuga_int)((t + 1) % 4);
        neighbour_edges[3 * k + 2] = 0;

        // e1: the outer edge -- facing tri of the adjacent cell, or the rim
        // (-1 for now; the numbering scan below assigns -(index+1)).
        anuga_int nbr = -1;
        switch (t) {
            case 0:  if (i > 0)      nbr = 4 * (c - my) + 2; break;   // (i-1, j) t2
            case 1:  if (j > 0)      nbr = 4 * (c - 1)  + 3; break;   // (i, j-1) t3
            case 2:  if (i < mx - 1) nbr = 4 * (c + my) + 0; break;   // (i+1, j) t0
            default: if (j < my - 1) nbr = 4 * (c + 1)  + 1; break;   // (i, j+1) t1
        }
        neighbours[3 * k + 1]      = nbr;
        neighbour_edges[3 * k + 1] = (nbr >= 0) ? 1 : -1;
        number_of_boundaries[k]    = (nbr >= 0) ? 0 : 1;
    }

    // ---- boundary numbering: deterministic block prefix scan -------------
    // Host order is k ascending (each rim triangle has exactly one rim edge,
    // always e1), so a scan over k reproduces the numbering exactly.
    anuga_int *bcells = (anuga_int *)dxmalloc((size_t)nb * sizeof(anuga_int));
    anuga_int *bedges = (anuga_int *)dxmalloc((size_t)nb * sizeof(anuga_int));
    {
        const anuga_int nblk = (n + SCAN_BLK - 1) / SCAN_BLK;
        anuga_int *bsum = (anuga_int *)dxmalloc((size_t)nblk * sizeof(anuga_int));

        #pragma omp target enter data map(alloc: bsum[0:nblk])
        OMP_PARALLEL_LOOP
        for (anuga_int b = 0; b < nblk; b++) {
            const anuga_int k0 = b * SCAN_BLK;
            const anuga_int k1 = (k0 + SCAN_BLK < n) ? k0 + SCAN_BLK : n;
            anuga_int s = 0;
            for (anuga_int k = k0; k < k1; k++)
                if (neighbours[3 * k + 1] < 0) s++;
            bsum[b] = s;
        }
        #pragma omp target update from(bsum[0:nblk])
        anuga_int total = 0;                       // exclusive scan on the host
        for (anuga_int b = 0; b < nblk; b++) {
            const anuga_int s = bsum[b]; bsum[b] = total; total += s;
        }
        if (total != nb) {
            fprintf(stderr, "bench: device boundary scan found %lld rim edges "
                    "(expected %lld)\n", (long long)total, (long long)nb);
            return 1;
        }
        #pragma omp target update to(bsum[0:nblk])

        #pragma omp target enter data map(alloc: bcells[0:nb], bedges[0:nb])
        OMP_PARALLEL_LOOP
        for (anuga_int b = 0; b < nblk; b++) {
            const anuga_int k0 = b * SCAN_BLK;
            const anuga_int k1 = (k0 + SCAN_BLK < n) ? k0 + SCAN_BLK : n;
            anuga_int idx = bsum[b];
            for (anuga_int k = k0; k < k1; k++) {
                if (neighbours[3 * k + 1] < 0) {
                    neighbours[3 * k + 1] = -(idx + 1);
                    bcells[idx] = k;
                    bedges[idx] = 1;
                    idx++;
                }
            }
        }
        #pragma omp target update from(bcells[0:nb], bedges[0:nb])
        #pragma omp target exit data map(delete: bcells[0:nb], bedges[0:nb], bsum[0:nblk])
        free(bsum);
    }

    // ---- surrogate neighbours (needs the final, numbered neighbours) -----
    OMP_PARALLEL_LOOP
    for (anuga_int k = 0; k < n; k++)
        for (int e = 0; e < 3; e++)
            surrogate_neighbours[3 * k + e] =
                (neighbours[3 * k + e] < 0) ? k : neighbours[3 * k + e];

    // ---- scatter owned-edge list (flux_mode 2): same scan pattern --------
    if (P->flux_mode == 2) {
        anuga_int *owned = D->owned_edges;
        const anuga_int n3 = 3 * n;
        const anuga_int nblk = (n3 + SCAN_BLK - 1) / SCAN_BLK;
        anuga_int *bsum = (anuga_int *)dxmalloc((size_t)nblk * sizeof(anuga_int));

        #pragma omp target enter data map(alloc: owned[0:n3], bsum[0:nblk])
        OMP_PARALLEL_LOOP
        for (anuga_int b = 0; b < nblk; b++) {
            const anuga_int p0 = b * SCAN_BLK;
            const anuga_int p1 = (p0 + SCAN_BLK < n3) ? p0 + SCAN_BLK : n3;
            anuga_int s = 0;
            for (anuga_int p2 = p0; p2 < p1; p2++) {
                const anuga_int k2 = p2 / 3;
                const anuga_int nbr2 = neighbours[p2];
                if (nbr2 < 0 || nbr2 > k2) s++;
            }
            bsum[b] = s;
        }
        #pragma omp target update from(bsum[0:nblk])
        anuga_int total = 0;
        for (anuga_int b = 0; b < nblk; b++) {
            const anuga_int s = bsum[b]; bsum[b] = total; total += s;
        }
        #pragma omp target update to(bsum[0:nblk])

        OMP_PARALLEL_LOOP
        for (anuga_int b = 0; b < nblk; b++) {
            const anuga_int p0 = b * SCAN_BLK;
            const anuga_int p1 = (p0 + SCAN_BLK < n3) ? p0 + SCAN_BLK : n3;
            anuga_int idx = bsum[b];
            for (anuga_int p2 = p0; p2 < p1; p2++) {
                const anuga_int k2 = p2 / 3;
                const anuga_int nbr2 = neighbours[p2];
                if (nbr2 < 0 || nbr2 > k2) owned[idx++] = p2;
            }
        }
        D->num_owned_edges = total;
        #pragma omp target exit data map(delete: bsum[0:nblk])
        free(bsum);
    } else if (P->flux_mode == 1 && D->edge_flux_work != NULL) {
        double *slots = D->edge_flux_work;
        const anuga_int nslots = 6 * 3 * n;
        #pragma omp target enter data map(alloc: slots[0:nslots])
    }

    // ---- reflective boundary: the one small host-side piece --------------
    {
        int *bidx = (int *)dxmalloc((size_t)nb * sizeof(int));
        int *vids = (int *)dxmalloc((size_t)nb * sizeof(int));
        int *eids = (int *)dxmalloc((size_t)nb * sizeof(int));
        for (anuga_int b = 0; b < nb; b++) {
            bidx[b] = (int)b;
            vids[b] = (int)bcells[b];
            eids[b] = (int)bedges[b];
        }
        if (gpu_reflective_init(GD, (int)nb, bidx, vids, eids) != 0) {
            fprintf(stderr, "bench: gpu_reflective_init failed\n");
            return 1;
        }
        free(bidx); free(vids); free(eids);

        // gpu_reflective_init keeps its own copies; map them exactly as
        // gpu_domain_map_arrays would have.
        struct reflective_boundary *R = &GD->reflective;
        int ne2 = R->num_edges;
        int *b_idx = R->boundary_indices;
        int *v_ids = R->vol_ids;
        int *e_ids = R->edge_ids;
        #pragma omp target enter data map(to: b_idx[0:ne2], v_ids[0:ne2], e_ids[0:ne2])
        R->mapped = 1;
    }
    free(bcells); free(bedges);

    // Mark the domain mapped: gpu_domain_map_arrays (and its device-memory
    // pre-check, which would double-count the premapped footprint) must not
    // run.  gpu_domain_finalize's map(delete:) tears everything down
    // regardless of reference counts, so no extra bookkeeping is needed.
    GD->backup_arrays_mapped = 1;
    GD->gpu_initialized = 1;

    if (verbose && rank == 0)
        printf("  device-side init: %lld triangles, %lld boundary edges\n",
               (long long)n, (long long)nb);
    return 0;
}

/* miniapp_part.c - MPI-aware standalone driver (C owns MPI, self-distributes).
 *
 * Reads ONE global .agm mesh (written by tools/global_mesh.py via meshpy+pymetis),
 * self-extracts this rank's sub-domain (owned + 2-layer ghosts) entirely in C,
 * runs the evolve loop with the halo exchange done in C, and writes its owned
 * cells as (global_id, stage) pairs for validation.
 *
 *   mpirun -n N ./anuga_miniapp_part <global.agm>
 * writes <global.agm>.P<N>_<rank>.result   (int64 count; int64 gids[]; double stage[])
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#include "anuga_sw.h"
#include "anuga_dump.h"

int main(int argc, char **argv) {
    anuga_mpi_init();
    int rank = anuga_comm_world_rank();
    int size = anuga_comm_world_size();
    const char *path = (argc > 1) ? argv[1] : "/tmp/anuga_global/dam_break.agm";

    AnugaLoadedDomain *ld = anuga_global_load(path, rank, size);
    if (!ld) { fprintf(stderr, "rank %d: load failed %s\n", rank, path);
               anuga_mpi_finalize(); return 1; }

    AnugaDomain *dom = anuga_dump_domain(ld);
    double t_setup0 = omp_get_wtime();
    anuga_domain_use_comm_world(dom);   /* sets rank + pins this rank's GPU */
    if (!anuga_domain_map_to_device(dom)) { fprintf(stderr, "rank %d: map failed\n", rank);
        anuga_dump_free(ld); anuga_mpi_finalize(); return 1; }
    anuga_sync_to_device(dom);
    double setup_wall = omp_get_wtime() - t_setup0;

    const double T = anuga_dump_finaltime(ld);
    const int method = anuga_dump_timestepping_method(ld);
    double t = 0.0; int steps = 0;
    double t_ev0 = omp_get_wtime();
    while (t < T - 1e-12) {
        double dt;
        switch (method) {
            case 1: dt = anuga_evolve_one_rk2_step(dom, T - t, 1); break;
            case 2: dt = anuga_evolve_one_rk3_step(dom, T - t, 1); break;
            case 3: dt = anuga_evolve_one_ader2_step(dom, T - t, 1, 0.0); break;
            default: dt = anuga_evolve_one_euler_step(dom, T - t, 1); break;
        }
        if (size > 1) anuga_exchange_ghosts(dom);   /* refresh ghosts for next step */
        t += dt; steps++;
    }
    double evolve_wall = omp_get_wtime() - t_ev0;
    anuga_sync_from_device(dom);

    long n = anuga_dump_num_elements(ld);
    double *sbuf = (double *)malloc((n ? n : 1) * sizeof(double));
    long   *gids = (long *)malloc((n ? n : 1) * sizeof(long));
    long mo = anuga_dump_get_owned_stage(ld, sbuf);
    anuga_dump_get_owned_global_ids(ld, gids);

    char out[4096];
    snprintf(out, sizeof(out), "%s.P%d_%d.result", path, size, rank);
    FILE *f = fopen(out, "wb");
    if (f) {
        int64_t m64 = mo, gid;
        fwrite(&m64, sizeof(int64_t), 1, f);
        for (long i = 0; i < mo; i++) { gid = gids[i]; fwrite(&gid, sizeof(int64_t), 1, f); }
        fwrite(sbuf, sizeof(double), (size_t)mo, f);
        fclose(f);
    }
    double s = 0.0; for (long i = 0; i < mo; i++) s += sbuf[i];
    printf("rank %d/%d: GPU %d  owned=%ld  evolve=%.3fs  stage_sum=%.10g\n",
           rank, size, anuga_device_id(dom), mo, evolve_wall, s);
    fflush(stdout);

    /* global statistics (max wall across ranks, total owned cells) */
    double max_evolve = anuga_mpi_allreduce_max_double(evolve_wall);
    double max_setup  = anuga_mpi_allreduce_max_double(setup_wall);
    double tot_cells  = anuga_mpi_allreduce_sum_double((double)mo);
    if (rank == 0) {
        double thr = (max_evolve > 0.0) ? tot_cells * steps / max_evolve / 1e6 : 0.0;
        printf("== global: ranks=%d cells=%.0f t=%.3f steps=%d | setup=%.3fs "
               "evolve=%.3fs (%.3f ms/step, %.1f Mcell-steps/s) ==\n",
               size, tot_cells, t, steps, max_setup, max_evolve,
               max_evolve / (steps > 0 ? steps : 1) * 1e3, thr);
        fflush(stdout);
    }

    free(sbuf); free(gids);
    anuga_dump_free(ld); anuga_mpi_finalize();
    return 0;
}

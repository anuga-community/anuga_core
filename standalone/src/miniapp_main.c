/* miniapp_main.c - pure-C, MPI-owned portable driver for libanuga_sw.
 *
 * No Python. MPI is owned here (system/HPC-X mpirun launches it); each rank
 * loads its .adm partition (written offline by tools/partition_dump.py), runs
 * the evolve loop on the GPU, and writes its owned-cell stage to a .result file.
 *
 *   mpirun -n N ./anuga_miniapp <dump_dir> <case_name>
 *
 * reads  <dump_dir>/<case_name>_P<N>_<rank>.adm
 * writes <dump_dir>/<case_name>_P<N>_<rank>.result  (int64 count, then doubles)
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "anuga_sw.h"
#include "anuga_dump.h"

int main(int argc, char **argv) {
    anuga_mpi_init();
    int rank = anuga_comm_world_rank();
    int size = anuga_comm_world_size();

    const char *dir  = (argc > 1) ? argv[1] : ".";
    const char *name = (argc > 2) ? argv[2] : "dam_break";

    char path[4096];
    snprintf(path, sizeof(path), "%s/%s_P%d_%d.adm", dir, name, size, rank);
    AnugaLoadedDomain *ld = anuga_dump_load(path);
    if (!ld) {
        fprintf(stderr, "rank %d: failed to load %s\n", rank, path);
        anuga_mpi_finalize();
        return 1;
    }

    AnugaDomain *dom = anuga_dump_domain(ld);
    anuga_domain_use_comm_world(dom);     /* bind to system MPI_COMM_WORLD */
    if (!anuga_domain_map_to_device(dom)) {
        fprintf(stderr, "rank %d: map_to_device failed\n", rank);
        anuga_dump_free(ld); anuga_mpi_finalize();
        return 1;
    }
    anuga_sync_to_device(dom);

    const double T = anuga_dump_finaltime(ld);
    const int method = anuga_dump_timestepping_method(ld);

    double t = 0.0;
    while (t < T - 1e-12) {
        double max_dt = T - t;   /* evolve_max_timestep >> dt for these cases */
        double dt;
        switch (method) {
            case 1: dt = anuga_evolve_one_rk2_step(dom, max_dt, 1); break;
            case 2: dt = anuga_evolve_one_rk3_step(dom, max_dt, 1); break;
            case 3: dt = anuga_evolve_one_ader2_step(dom, max_dt, 1, 0.0); break;
            default:
                dt = anuga_evolve_one_euler_step(dom, max_dt, 1);
                if (size > 1) anuga_exchange_ghosts(dom);  /* euler kernel doesn't */
                break;
        }
        t += dt;
    }

    anuga_sync_from_device(dom);

    long n = anuga_dump_num_elements(ld);
    double *buf = (double *)malloc((n ? n : 1) * sizeof(double));
    long m = anuga_dump_get_owned_stage(ld, buf);

    char out[4096];
    snprintf(out, sizeof(out), "%s/%s_P%d_%d.result", dir, name, size, rank);
    FILE *f = fopen(out, "wb");
    if (f) {
        int64_t m64 = m;
        fwrite(&m64, sizeof(int64_t), 1, f);
        fwrite(buf, sizeof(double), (size_t)m, f);
        fclose(f);
    }

    double s = 0.0;
    for (long i = 0; i < m; i++) s += buf[i];
    printf("rank %d/%d: t=%.6f owned=%ld stage_sum=%.10g -> %s\n",
           rank, size, t, m, s, out);
    fflush(stdout);

    free(buf);
    anuga_dump_free(ld);
    anuga_mpi_finalize();
    return 0;
}

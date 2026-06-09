/* anuga_dump.h - portable .adm partition-dump format + loader.
 *
 * An .adm file holds ONE rank's partitioned sub-domain: mesh, geometry,
 * quantities, MPI halo, and static boundaries. It is produced offline (serial,
 * any machine) by standalone/tools/partition_dump.py and read on the target by
 * the pure-C mini-app or the ctypes driver. This is what lets the portable
 * runtime own MPI in C (system/HPC-X mpirun) with no mpi4py/conda.
 *
 * Format (all little-endian; every header field is 8 bytes so the header struct
 * has no padding):
 *   char   magic[8] = "ANUGADM1"
 *   int64  version (=1)
 *   int64  n, nb, rank, nprocs,
 *          optimise_dry_cells, extrapolate_velocity_second_order, low_froude,
 *          timestep_fluxcalls, timestepping_method (0 euler,1 rk2,2 rk3,3 ader2),
 *          num_neighbors, total_send, total_recv,
 *          n_reflective, n_dirichlet, n_transmissive, transmissive_use_centroid
 *   double epsilon,H0,g,evolve_max_timestep,evolve_min_timestep,
 *          minimum_allowed_height,maximum_allowed_speed,
 *          beta_w,beta_w_dry,beta_uh,beta_uh_dry,beta_vh,beta_vh_dry,
 *          CFL,fixed_flux_timestep,finaltime,yieldstep,
 *          dirichlet_stage,dirichlet_xmom,dirichlet_ymom
 * Then arrays, contiguous, in this exact order:
 *   int64 : neighbours[3n] neighbour_edges[3n] surrogate_neighbours[3n]
 *           number_of_boundaries[n] tri_full_flag[n]
 *   double: normals[6n] edgelengths[3n] radii[n] areas[n]
 *           centroid_coordinates[2n] edge_coordinates[6n]
 *   double: {stage,xmom,ymom,bed,height} x {centroid[n], edge[3n]}, friction_centroid[n]
 *   int32 : neighbor_ranks[num_neighbors] send_counts[num_neighbors]
 *           recv_counts[num_neighbors] flat_send[total_send] flat_recv[total_recv]
 *   int32 : reflective  {boundary_indices,vol_ids,edge_ids}[n_reflective]
 *           dirichlet   {boundary_indices,vol_ids,edge_ids}[n_dirichlet]
 *           transmissive{boundary_indices,vol_ids,edge_ids}[n_transmissive]
 * boundary_values and the explicit/semi-implicit/backup/work/max_speed scratch
 * arrays are NOT stored; the loader allocates them zeroed (they are overwritten
 * each step, and boundaries are evaluated before the first flux).
 */
#ifndef ANUGA_DUMP_H
#define ANUGA_DUMP_H

#include "anuga_sw.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ANUGA_DUMP_MAGIC   "ANUGADM1"
#define ANUGA_DUMP_VERSION 1

/* Owns a loaded domain plus all the host arrays backing it. */
typedef struct AnugaLoadedDomain AnugaLoadedDomain;

/* Read path/<name>_P<np>_<rank>.adm, allocate arrays, create the domain and set
 * up halo + static boundaries. Returns NULL on error. Does NOT set the MPI
 * communicator or map to device - the caller does that (use_comm_world, map). */
AnugaLoadedDomain *anuga_dump_load(const char *path);

AnugaDomain *anuga_dump_domain(AnugaLoadedDomain *ld);
double anuga_dump_finaltime(AnugaLoadedDomain *ld);
double anuga_dump_yieldstep(AnugaLoadedDomain *ld);
int    anuga_dump_timestepping_method(AnugaLoadedDomain *ld);
int    anuga_dump_rank(AnugaLoadedDomain *ld);
int    anuga_dump_nprocs(AnugaLoadedDomain *ld);
long   anuga_dump_num_elements(AnugaLoadedDomain *ld);

/* Copy owned-cell (tri_full_flag==1) centroid values out, for validation.
 * out must hold at least anuga_dump_num_elements doubles; returns count written. */
long anuga_dump_get_owned_stage(AnugaLoadedDomain *ld, double *out);

void anuga_dump_free(AnugaLoadedDomain *ld);

#ifdef __cplusplus
}
#endif

#endif /* ANUGA_DUMP_H */

/* anuga_sw.h - Stable C ABI for the standalone ANUGA shallow-water solver.
 *
 * This header is the *only* thing a caller (a C driver, or the Python ctypes
 * wrapper in standalone/python/anuga_sw.py) needs to know about. It exposes the
 * production GPU/OpenMP kernels (anuga/shallow_water/gpu/ *.c) behind a flat,
 * numpy-free interface: the caller passes raw pointers to its own arrays and
 * scalar parameters; this library never owns or copies that host memory, it only
 * records the pointers (exactly the contract the Cython bridge has today).
 *
 * Phase 1 scope: domain create/map/sync, RK2/Euler/ADER2 stepping, and the
 * static boundaries (reflective / Dirichlet / transmissive). Time/file
 * boundaries and rate/inlet/culvert operators are added in later phases.
 *
 * Integer convention: array indices into the mesh (neighbours, etc.) use
 * anuga_int == int64_t, matching anuga/utilities/anuga_typedefs.h and struct
 * domain. Boundary segment index arrays use plain int (int32), matching the
 * numpy np.intc arrays ANUGA builds for boundary_cells / boundary_edges.
 */
#ifndef ANUGA_SW_H
#define ANUGA_SW_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Guarded so this header coexists with anuga_typedefs.h (which defines the same
 * typedef) when both are pulled into one translation unit, e.g. anuga_sw_capi.c. */
#ifndef ANUGA_TYPEDEFS_H
typedef int64_t anuga_int;
#endif

/* Opaque handle wrapping the internal struct gpu_domain. */
typedef struct AnugaDomain AnugaDomain;

/* Descriptor of a (possibly partitioned) domain. Every pointer aliases
 * caller-owned memory that must stay alive for the lifetime of the AnugaDomain.
 * Field order/types are mirrored exactly by the ctypes Structure in
 * python/anuga_sw.py - keep the two in lockstep.
 *
 * Nullable pointers (may be NULL when the feature is unused): tri_full_flag,
 * edge_flux_type, edge_river_wall_counter, and all riverwall_* arrays.
 */
typedef struct AnugaDomainDesc {
    /* ---- scalar sizes ---- */
    anuga_int number_of_elements;        /* n triangles (incl. ghosts)        */
    anuga_int boundary_length;           /* nb boundary edges                 */
    anuga_int number_of_riverwall_edges; /* 0 if no riverwalls                */

    /* ---- scalar parameters ---- */
    anuga_int optimise_dry_cells;
    anuga_int extrapolate_velocity_second_order;
    anuga_int low_froude;
    anuga_int timestep_fluxcalls;
    anuga_int ncol_riverwall_hydraulic_properties;
    anuga_int nrow_riverwall_hydraulic_properties;

    double epsilon;
    double H0;
    double g;
    double evolve_max_timestep;
    double evolve_min_timestep;
    double minimum_allowed_height;
    double maximum_allowed_speed;
    double beta_w;
    double beta_w_dry;
    double beta_uh;
    double beta_uh_dry;
    double beta_vh;
    double beta_vh_dry;

    /* extras carried on gpu_domain (not struct domain) */
    double CFL;
    double fixed_flux_timestep;          /* <= 0 disables (use CFL timestep)  */

    /* ---- mesh connectivity / geometry (int64 / double) ---- */
    anuga_int *neighbours;               /* [n*3]                             */
    anuga_int *neighbour_edges;          /* [n*3]                             */
    anuga_int *surrogate_neighbours;     /* [n*3]                             */
    anuga_int *number_of_boundaries;     /* [n]                               */
    anuga_int *tri_full_flag;            /* [n], nullable                     */
    double    *normals;                  /* [n*6]                             */
    double    *edgelengths;              /* [n*3]                             */
    double    *radii;                    /* [n]                               */
    double    *areas;                    /* [n]                               */
    double    *max_speed;                /* [n]                               */
    double    *centroid_coordinates;     /* [n*2]                             */
    double    *edge_coordinates;         /* [n*3*2]                           */
    double    *x_centroid_work;          /* [n]                               */
    double    *y_centroid_work;          /* [n]                               */

    /* ---- stage quantity ---- */
    double *stage_centroid_values;       /* [n]      */
    double *stage_edge_values;           /* [n*3]    */
    double *stage_boundary_values;       /* [nb]     */
    double *stage_explicit_update;       /* [n]      */
    double *stage_semi_implicit_update;  /* [n]      */
    double *stage_backup_values;         /* [n]      */

    /* ---- xmomentum quantity ---- */
    double *xmom_centroid_values;
    double *xmom_edge_values;
    double *xmom_boundary_values;
    double *xmom_explicit_update;
    double *xmom_semi_implicit_update;
    double *xmom_backup_values;

    /* ---- ymomentum quantity ---- */
    double *ymom_centroid_values;
    double *ymom_edge_values;
    double *ymom_boundary_values;
    double *ymom_explicit_update;
    double *ymom_semi_implicit_update;
    double *ymom_backup_values;

    /* ---- elevation (bed) ---- */
    double *bed_centroid_values;
    double *bed_edge_values;
    double *bed_boundary_values;

    /* ---- height ---- */
    double *height_centroid_values;
    double *height_edge_values;
    double *height_boundary_values;

    /* ---- friction ---- */
    double *friction_centroid_values;    /* [n] */

    /* ---- riverwall (all nullable) ---- */
    anuga_int *edge_flux_type;                  /* [n*3] */
    anuga_int *edge_river_wall_counter;         /* [n*3] */
    double    *riverwall_elevation;
    anuga_int *riverwall_rowIndex;
    double    *riverwall_hydraulic_properties;
} AnugaDomainDesc;

/* ============================ build info ================================== */
/* 1 if compiled with real GPU offload, 0 if CPU_ONLY_MODE.                   */
int anuga_built_with_offload(void);
/* 1 if compiled against real MPI, 0 if single-process stubs.                 */
int anuga_built_with_mpi(void);
/* 1 if an offload device is actually available at runtime.                   */
int anuga_gpu_available(void);

/* ============================ C-owned MPI =============================== */
/* For the portable runtime where MPI is owned by C (launched by the system
 * mpirun, e.g. nvhpc HPC-X) instead of mpi4py. No-op / single-rank in builds
 * without MPI. Call anuga_mpi_init once at program start, anuga_mpi_finalize at
 * end, and anuga_domain_use_comm_world to bind a domain to MPI_COMM_WORLD. */
void anuga_mpi_init(void);
int  anuga_mpi_initialized(void);
void anuga_mpi_finalize(void);
int  anuga_comm_world_rank(void);
int  anuga_comm_world_size(void);
void anuga_domain_use_comm_world(AnugaDomain *dom);

/* Collective reductions over MPI_COMM_WORLD (return x unchanged in non-MPI builds). */
double anuga_mpi_allreduce_max_double(double x);
double anuga_mpi_allreduce_sum_double(double x);

/* ============================ lifecycle ================================== */
/* Create a serial (single-rank) domain from the descriptor. The descriptor
 * itself need not outlive the call, but every array it points to must. */
AnugaDomain *anuga_domain_create(const AnugaDomainDesc *desc);

/* Attach an MPI communicator (Fortran handle from mpi4py comm.py2f()).
 * Pass -1 for serial. No-op in single-process builds. Sets rank/nprocs from it. */
void anuga_domain_set_mpi_comm_fint(AnugaDomain *dom, int comm_fint);

/* Set up the MPI halo (ghost) exchange. Mirrors build_halo_from_dicts in the
 * Cython bridge. Call after set_mpi_comm and BEFORE map_to_device. The flat
 * arrays are copied by the library. Index arrays are int32.
 *   neighbor_ranks[num_neighbors]   ranks we exchange with
 *   send_counts/recv_counts[num_neighbors]
 *   flat_send_indices[sum(send_counts)]   local cell ids to pack per neighbor
 *   flat_recv_indices[sum(recv_counts)]   local ghost cell ids to unpack into  */
void anuga_init_halo(AnugaDomain *dom, int num_neighbors,
                     const int *neighbor_ranks,
                     const int *send_counts, const int *recv_counts,
                     const int *flat_send_indices, const int *flat_recv_indices);

/* Exchange ghost cells between ranks (also done inside the evolve steps when
 * nprocs > 1; exposed for explicit use/testing). No-op single-process. */
void anuga_exchange_ghosts(AnugaDomain *dom);

/* Current rank / number of ranks as seen by the library. */
int anuga_rank(AnugaDomain *dom);
int anuga_nprocs(AnugaDomain *dom);
/* GPU device this rank is pinned to (-1 if no device / CPU). */
int anuga_device_id(AnugaDomain *dom);

/* Map host arrays to the device (no-op in CPU_ONLY_MODE). Returns 1 on success,
 * 0 if device memory is insufficient. */
int anuga_domain_map_to_device(AnugaDomain *dom);

void anuga_domain_destroy(AnugaDomain *dom);

/* ============================ host<->device sync ======================== */
void anuga_sync_to_device(AnugaDomain *dom);        /* centroid H2D            */
void anuga_sync_from_device(AnugaDomain *dom);      /* centroid D2H            */
void anuga_sync_all_from_device(AnugaDomain *dom);  /* everything D2H (debug)  */

/* ============================ boundaries ================================= */
/* For each, num_edges segment edges with parallel index arrays:
 *   boundary_indices[k] -> slot in *_boundary_values
 *   vol_ids[k]          -> interior cell id
 *   edge_ids[k]         -> which edge (0,1,2)
 * Index arrays are int32 (np.intc); the library copies them, so they need not
 * outlive the call. */
void anuga_init_reflective(AnugaDomain *dom, int num_edges,
                           const int *boundary_indices,
                           const int *vol_ids, const int *edge_ids);
void anuga_init_dirichlet(AnugaDomain *dom, int num_edges,
                          const int *boundary_indices,
                          const int *vol_ids, const int *edge_ids,
                          double stage_value, double xmom_value, double ymom_value);
void anuga_init_transmissive(AnugaDomain *dom, int num_edges,
                             const int *boundary_indices,
                             const int *vol_ids, const int *edge_ids,
                             int use_centroid);

/* Evaluate all initialised boundaries on the device (normally called inside the
 * evolve steps; exposed for kernel-level testing/parity checks). */
void anuga_evaluate_boundaries(AnugaDomain *dom);

/* ============================ evolve ==================================== */
/* Each returns the timestep actually taken. apply_forcing applies Manning
 * friction inside the step (1) or not (0). */
double anuga_evolve_one_euler_step(AnugaDomain *dom, double max_timestep, int apply_forcing);
double anuga_evolve_one_rk2_step(AnugaDomain *dom, double max_timestep, int apply_forcing);
double anuga_evolve_one_rk3_step(AnugaDomain *dom, double max_timestep, int apply_forcing);
double anuga_evolve_one_ader2_step(AnugaDomain *dom, double max_timestep,
                                   int apply_forcing, double prev_dt);

/* ============================ individual kernels (parity testing) ======== */
void   anuga_extrapolate_second_order(AnugaDomain *dom);
double anuga_compute_fluxes(AnugaDomain *dom);
void   anuga_update_conserved_quantities(AnugaDomain *dom, double timestep);
double anuga_protect(AnugaDomain *dom);
void   anuga_manning_friction(AnugaDomain *dom);
void   anuga_backup_conserved_quantities(AnugaDomain *dom);
void   anuga_saxpy_conserved_quantities(AnugaDomain *dom, double a, double b);

/* ============================ diagnostics =============================== */
double anuga_compute_water_volume(AnugaDomain *dom);

#ifdef __cplusplus
}
#endif

#endif /* ANUGA_SW_H */

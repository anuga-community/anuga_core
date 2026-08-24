// Argument block for the CUDA reconstruction-kernel experiment.
// All pointers are DEVICE pointers (the caller resolves them with
// omp_get_mapped_ptr); the .so is pure CUDA and never touches OpenMP.
#ifndef CUDA_EXTRAP_H
#define CUDA_EXTRAP_H

struct extrap_args {
    long long n;
    double minimum_allowed_height;
    long long extrapolate_velocity_second_order;
    double g;
    double beta_w, beta_w_dry, beta_uh, beta_uh_dry, beta_vh, beta_vh_dry;
    double predictor_dt;

    double *stage_cv, *xmom_cv, *ymom_cv, *bed_cv, *height_cv;
    double *stage_ev, *xmom_ev, *ymom_ev, *bed_ev, *height_ev;
    double *centroid_coords, *edge_coords;
    long long *surrogate_neighbours, *number_of_boundaries;
    double *x_centroid_work, *y_centroid_work;
};

// Launches the kernel and synchronizes.  Returns 0 on success.
typedef int (*cuda_extrap_fn)(struct extrap_args a, int tpb);

#endif

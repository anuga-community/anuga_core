// GPU boundary implementations
//
// All boundary types: Reflective, Dirichlet, Transmissive,
// Transmissive_n_zero_t, Time_boundary

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "gpu_domain.h"
#include "gpu_omp_macros.h"

// ============================================================================
// Reflective Boundary Setup
// ============================================================================

int gpu_reflective_init(struct gpu_domain *GD, int num_edges,
                        int *boundary_indices, int *vol_ids, int *edge_ids) {
    struct reflective_boundary *R = &GD->reflective;

    R->num_edges = num_edges;
    R->mapped = 0;  // Will be mapped later in gpu_domain_map_arrays

    if (num_edges == 0) {
        R->boundary_indices = NULL;
        R->vol_ids = NULL;
        R->edge_ids = NULL;
        return 0;
    }

    // Allocate and copy arrays (will be mapped to GPU in gpu_domain_map_arrays)
    R->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    R->vol_ids = (int*)malloc(num_edges * sizeof(int));
    R->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!R->boundary_indices || !R->vol_ids || !R->edge_ids) {
        fprintf(stderr, "Failed to allocate reflective boundary arrays\n");
        return -1;
    }

    memcpy(R->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(R->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(R->edge_ids, edge_ids, num_edges * sizeof(int));


    return 0;
}

void gpu_reflective_finalize(struct gpu_domain *GD) {
    struct reflective_boundary *R = &GD->reflective;

    if (R->mapped && R->num_edges > 0) {
        int *b_idx = R->boundary_indices;
        int *v_ids = R->vol_ids;
        int *e_ids = R->edge_ids;
        int ne = R->num_edges;

        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (R->boundary_indices) free(R->boundary_indices);
    if (R->vol_ids) free(R->vol_ids);
    if (R->edge_ids) free(R->edge_ids);

    R->num_edges = 0;
    R->boundary_indices = NULL;
    R->vol_ids = NULL;
    R->edge_ids = NULL;
    R->mapped = 0;
}

void gpu_evaluate_reflective_boundary(struct gpu_domain *GD) {
    // Evaluate reflective boundary entirely on GPU
    // No data transfer needed - reads edge values, writes boundary values

    struct reflective_boundary *R = &GD->reflective;
    if (R->num_edges == 0) return;

    // Only run GPU kernel if arrays are mapped
    if (!R->mapped) {
        // Arrays not on GPU - caller should use CPU fallback
        return;
    }

    int num_edges = R->num_edges;
    int *boundary_indices = R->boundary_indices;
    int *vol_ids = R->vol_ids;
    int *edge_ids = R->edge_ids;

    // Edge values (read)
    double *stage_ev = GD->D.stage_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *normals = GD->D.normals;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    // All arrays already mapped via target enter data
    OMP_PARALLEL_LOOP
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Copy conserved quantities from edge to boundary
        stage_bv[bid] = stage_ev[3 * vid + eid];
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];

        // Get normal vector for this edge
        double n1 = normals[vid * 6 + 2 * eid];
        double n2 = normals[vid * 6 + 2 * eid + 1];

        // Get interior momentum
        double q1 = xmom_ev[3 * vid + eid];
        double q2 = ymom_ev[3 * vid + eid];

        // Reflect momentum: negate normal component, keep tangential
        // r = q - 2*(q.n)*n  but we compute it via rotation
        double r1 = -q1 * n1 - q2 * n2;  // -(q dot n)
        double r2 = -q1 * n2 + q2 * n1;  // tangential component

        // Rotate back
        xmom_bv[bid] = n1 * r1 - n2 * r2;
        ymom_bv[bid] = n2 * r1 + n1 * r2;
    }
}

// ============================================================================
// Dirichlet Boundary Setup and Evaluation
// ============================================================================

int gpu_dirichlet_init(struct gpu_domain *GD, int num_edges,
                       int *boundary_indices, int *vol_ids, int *edge_ids,
                       double stage_value, double xmom_value, double ymom_value) {
    struct dirichlet_boundary *D = &GD->dirichlet;

    D->num_edges = num_edges;
    D->mapped = 0;
    D->stage_value = stage_value;
    D->xmom_value = xmom_value;
    D->ymom_value = ymom_value;

    if (num_edges == 0) {
        D->boundary_indices = NULL;
        D->vol_ids = NULL;
        D->edge_ids = NULL;
        return 0;
    }

    D->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    D->vol_ids = (int*)malloc(num_edges * sizeof(int));
    D->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!D->boundary_indices || !D->vol_ids || !D->edge_ids) {
        fprintf(stderr, "Failed to allocate Dirichlet boundary arrays\n");
        return -1;
    }

    memcpy(D->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(D->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(D->edge_ids, edge_ids, num_edges * sizeof(int));


    return 0;
}

void gpu_dirichlet_finalize(struct gpu_domain *GD) {
    struct dirichlet_boundary *D = &GD->dirichlet;

    if (D->mapped && D->num_edges > 0) {
        int ne = D->num_edges;
        int *b_idx = D->boundary_indices;
        int *v_ids = D->vol_ids;
        int *e_ids = D->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (D->boundary_indices) free(D->boundary_indices);
    if (D->vol_ids) free(D->vol_ids);
    if (D->edge_ids) free(D->edge_ids);

    D->num_edges = 0;
    D->boundary_indices = NULL;
    D->vol_ids = NULL;
    D->edge_ids = NULL;
    D->mapped = 0;
}

void gpu_evaluate_dirichlet_boundary(struct gpu_domain *GD) {
    struct dirichlet_boundary *D = &GD->dirichlet;
    if (D->num_edges == 0) return;
    if (!D->mapped) return;

    int num_edges = D->num_edges;
    int *boundary_indices = D->boundary_indices;
    int *vol_ids = D->vol_ids;
    int *edge_ids = D->edge_ids;
    double stage_val = D->stage_value;
    double xmom_val = D->xmom_value;
    double ymom_val = D->ymom_value;

    // Edge values (read for bed/height)
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Set constant Dirichlet values
        stage_bv[bid] = stage_val;
        xmom_bv[bid] = xmom_val;
        ymom_bv[bid] = ymom_val;

        // Copy bed/height from interior edge
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];
    }
}

// ============================================================================
// Transmissive Boundary Setup and Evaluation
// ============================================================================

int gpu_transmissive_init(struct gpu_domain *GD, int num_edges,
                          int *boundary_indices, int *vol_ids, int *edge_ids,
                          int use_centroid) {
    struct transmissive_boundary *T = &GD->transmissive;

    T->num_edges = num_edges;
    T->mapped = 0;
    T->use_centroid = use_centroid;

    if (num_edges == 0) {
        T->boundary_indices = NULL;
        T->vol_ids = NULL;
        T->edge_ids = NULL;
        return 0;
    }

    T->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    T->vol_ids = (int*)malloc(num_edges * sizeof(int));
    T->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!T->boundary_indices || !T->vol_ids || !T->edge_ids) {
        fprintf(stderr, "Failed to allocate Transmissive boundary arrays\n");
        return -1;
    }

    memcpy(T->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(T->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(T->edge_ids, edge_ids, num_edges * sizeof(int));


    return 0;
}

void gpu_transmissive_finalize(struct gpu_domain *GD) {
    struct transmissive_boundary *T = &GD->transmissive;

    if (T->mapped && T->num_edges > 0) {
        int ne = T->num_edges;
        int *b_idx = T->boundary_indices;
        int *v_ids = T->vol_ids;
        int *e_ids = T->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (T->boundary_indices) free(T->boundary_indices);
    if (T->vol_ids) free(T->vol_ids);
    if (T->edge_ids) free(T->edge_ids);

    T->num_edges = 0;
    T->boundary_indices = NULL;
    T->vol_ids = NULL;
    T->edge_ids = NULL;
    T->mapped = 0;
}

void gpu_evaluate_transmissive_boundary(struct gpu_domain *GD) {
    struct transmissive_boundary *T = &GD->transmissive;
    if (T->num_edges == 0) return;
    if (!T->mapped) return;

    int num_edges = T->num_edges;
    int *boundary_indices = T->boundary_indices;
    int *vol_ids = T->vol_ids;
    int *edge_ids = T->edge_ids;
    int use_centroid = T->use_centroid;

    // Centroid values (for use_centroid mode)
    double *stage_cv = GD->D.stage_centroid_values;
    double *xmom_cv = GD->D.xmom_centroid_values;
    double *ymom_cv = GD->D.ymom_centroid_values;
    double *bed_cv = GD->D.bed_centroid_values;
    double *height_cv = GD->D.height_centroid_values;

    // Edge values
    double *stage_ev = GD->D.stage_edge_values;
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        if (use_centroid) {
            // Copy from centroid values
            stage_bv[bid] = stage_cv[vid];
            xmom_bv[bid] = xmom_cv[vid];
            ymom_bv[bid] = ymom_cv[vid];
            bed_bv[bid] = bed_cv[vid];
            height_bv[bid] = height_cv[vid];
        } else {
            // Copy from edge values
            stage_bv[bid] = stage_ev[3 * vid + eid];
            xmom_bv[bid] = xmom_ev[3 * vid + eid];
            ymom_bv[bid] = ymom_ev[3 * vid + eid];
            bed_bv[bid] = bed_ev[3 * vid + eid];
            height_bv[bid] = height_ev[3 * vid + eid];
        }
    }
}

// ============================================================================
// Transmissive_n_momentum_zero_t_momentum_set_stage Boundary
// ============================================================================

int gpu_transmissive_n_zero_t_init(struct gpu_domain *GD, int num_edges,
                                   int *boundary_indices, int *vol_ids, int *edge_ids) {
    struct transmissive_n_zero_t_boundary *B = &GD->transmissive_n_zero_t;

    B->num_edges = num_edges;
    B->mapped = 0;
    B->stage_value = 0.0;

    if (num_edges == 0) {
        B->boundary_indices = NULL;
        B->vol_ids = NULL;
        B->edge_ids = NULL;
        return 0;
    }

    B->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    B->vol_ids = (int*)malloc(num_edges * sizeof(int));
    B->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!B->boundary_indices || !B->vol_ids || !B->edge_ids) {
        fprintf(stderr, "Failed to allocate transmissive_n_zero_t boundary arrays\n");
        return -1;
    }

    memcpy(B->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(B->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(B->edge_ids, edge_ids, num_edges * sizeof(int));


    return 0;
}

void gpu_transmissive_n_zero_t_finalize(struct gpu_domain *GD) {
    struct transmissive_n_zero_t_boundary *B = &GD->transmissive_n_zero_t;

    if (B->mapped && B->num_edges > 0) {
        int ne = B->num_edges;
        int *b_idx = B->boundary_indices;
        int *v_ids = B->vol_ids;
        int *e_ids = B->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (B->boundary_indices) free(B->boundary_indices);
    if (B->vol_ids) free(B->vol_ids);
    if (B->edge_ids) free(B->edge_ids);

    B->num_edges = 0;
    B->boundary_indices = NULL;
    B->vol_ids = NULL;
    B->edge_ids = NULL;
    B->mapped = 0;
}

void gpu_transmissive_n_zero_t_set_stage(struct gpu_domain *GD, double stage_value) {
    // Update stage value - called from Python each timestep before evaluate
    GD->transmissive_n_zero_t.stage_value = stage_value;
}

void gpu_evaluate_transmissive_n_zero_t_boundary(struct gpu_domain *GD) {
    // Transmissive normal momentum, zero tangential momentum, set stage
    struct transmissive_n_zero_t_boundary *B = &GD->transmissive_n_zero_t;
    if (B->num_edges == 0) return;
    if (!B->mapped) return;

    int num_edges = B->num_edges;
    int *boundary_indices = B->boundary_indices;
    int *vol_ids = B->vol_ids;
    int *edge_ids = B->edge_ids;
    double stage_val = B->stage_value;

    // Edge values (read)
    double *xmom_ev = GD->D.xmom_edge_values;
    double *ymom_ev = GD->D.ymom_edge_values;
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;
    double *normals = GD->D.normals;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Set stage from external value
        stage_bv[bid] = stage_val;

        // Copy bed/height from interior edge
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];

        // Get normal vector for this edge
        double n1 = normals[vid * 6 + 2 * eid];
        double n2 = normals[vid * 6 + 2 * eid + 1];

        // Get interior momentum
        double q1 = xmom_ev[3 * vid + eid];
        double q2 = ymom_ev[3 * vid + eid];

        // Compute normal component of momentum (dot product with normal)
        double ndotq = n1 * q1 + n2 * q2;

        // Set boundary momentum to just the normal component (zero tangential)
        xmom_bv[bid] = ndotq * n1;
        ymom_bv[bid] = ndotq * n2;
    }
}

// ============================================================================
// Time_boundary - time-dependent Dirichlet values
// ============================================================================

int gpu_time_boundary_init(struct gpu_domain *GD, int num_edges,
                           int *boundary_indices, int *vol_ids, int *edge_ids) {
    struct time_boundary *B = &GD->time_bdry;

    B->num_edges = num_edges;
    B->mapped = 0;
    B->stage_value = 0.0;
    B->xmom_value = 0.0;
    B->ymom_value = 0.0;

    if (num_edges == 0) {
        B->boundary_indices = NULL;
        B->vol_ids = NULL;
        B->edge_ids = NULL;
        return 0;
    }

    B->boundary_indices = (int*)malloc(num_edges * sizeof(int));
    B->vol_ids = (int*)malloc(num_edges * sizeof(int));
    B->edge_ids = (int*)malloc(num_edges * sizeof(int));

    if (!B->boundary_indices || !B->vol_ids || !B->edge_ids) {
        fprintf(stderr, "Failed to allocate time_boundary arrays\n");
        return -1;
    }

    memcpy(B->boundary_indices, boundary_indices, num_edges * sizeof(int));
    memcpy(B->vol_ids, vol_ids, num_edges * sizeof(int));
    memcpy(B->edge_ids, edge_ids, num_edges * sizeof(int));


    return 0;
}

void gpu_time_boundary_finalize(struct gpu_domain *GD) {
    struct time_boundary *B = &GD->time_bdry;

    if (B->mapped && B->num_edges > 0) {
        int ne = B->num_edges;
        int *b_idx = B->boundary_indices;
        int *v_ids = B->vol_ids;
        int *e_ids = B->edge_ids;
        #pragma omp target exit data map(delete: b_idx[0:ne], v_ids[0:ne], e_ids[0:ne])
    }

    if (B->boundary_indices) free(B->boundary_indices);
    if (B->vol_ids) free(B->vol_ids);
    if (B->edge_ids) free(B->edge_ids);

    B->num_edges = 0;
    B->boundary_indices = NULL;
    B->vol_ids = NULL;
    B->edge_ids = NULL;
    B->mapped = 0;
}

void gpu_time_boundary_set_values(struct gpu_domain *GD, double stage, double xmom, double ymom) {
    // Update values - called from Python each timestep before evaluate
    GD->time_bdry.stage_value = stage;
    GD->time_bdry.xmom_value = xmom;
    GD->time_bdry.ymom_value = ymom;
}

void gpu_evaluate_time_boundary(struct gpu_domain *GD) {
    // Time-dependent Dirichlet boundary - sets constant values (that vary with time)
    struct time_boundary *B = &GD->time_bdry;
    if (B->num_edges == 0) return;
    if (!B->mapped) return;

    int num_edges = B->num_edges;
    int *boundary_indices = B->boundary_indices;
    int *vol_ids = B->vol_ids;
    int *edge_ids = B->edge_ids;
    double stage_val = B->stage_value;
    double xmom_val = B->xmom_value;
    double ymom_val = B->ymom_value;

    // Edge values (read for bed/height)
    double *bed_ev = GD->D.bed_edge_values;
    double *height_ev = GD->D.height_edge_values;

    // Boundary values (write)
    double *stage_bv = GD->D.stage_boundary_values;
    double *bed_bv = GD->D.bed_boundary_values;
    double *height_bv = GD->D.height_boundary_values;
    double *xmom_bv = GD->D.xmom_boundary_values;
    double *ymom_bv = GD->D.ymom_boundary_values;

    OMP_PARALLEL_LOOP
    for (int k = 0; k < num_edges; k++) {
        int bid = boundary_indices[k];
        int vid = vol_ids[k];
        int eid = edge_ids[k];

        // Set stage and momentum from time-dependent values
        stage_bv[bid] = stage_val;
        xmom_bv[bid] = xmom_val;
        ymom_bv[bid] = ymom_val;

        // Copy bed/height from interior edge
        bed_bv[bid] = bed_ev[3 * vid + eid];
        height_bv[bid] = height_ev[3 * vid + eid];
    }
}

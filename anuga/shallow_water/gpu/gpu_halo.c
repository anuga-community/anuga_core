// GPU-accelerated shallow water solver
// Split from sw_domain_gpu.c for maintainability

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include "gpu_domain.h"

// Halo exchange setup and MPI ghost exchange

// ============================================================================
// Halo Exchange Setup
// ============================================================================

int gpu_halo_init(struct gpu_domain *GD,
                  int num_neighbors,
                  int *neighbor_ranks,
                  int *send_counts,
                  int *recv_counts,
                  int *flat_send_indices,
                  int *flat_recv_indices) {
    struct halo_exchange *H = &GD->halo;

    H->num_neighbors = num_neighbors;

    if (num_neighbors == 0) {
        // No communication needed
        return 0;
    }

    // Allocate and copy neighbor info
    H->neighbor_ranks = (int *)malloc(num_neighbors * sizeof(int));
    H->send_counts = (int *)malloc(num_neighbors * sizeof(int));
    H->recv_counts = (int *)malloc(num_neighbors * sizeof(int));
    H->send_offsets = (int *)malloc((num_neighbors + 1) * sizeof(int));
    H->recv_offsets = (int *)malloc((num_neighbors + 1) * sizeof(int));

    memcpy(H->neighbor_ranks, neighbor_ranks, num_neighbors * sizeof(int));
    memcpy(H->send_counts, send_counts, num_neighbors * sizeof(int));
    memcpy(H->recv_counts, recv_counts, num_neighbors * sizeof(int));

    // Compute total sizes and offsets
    H->total_send_size = 0;
    H->total_recv_size = 0;
    H->send_offsets[0] = 0;
    H->recv_offsets[0] = 0;

    for (int ni = 0; ni < num_neighbors; ni++) {
        H->total_send_size += send_counts[ni];
        H->total_recv_size += recv_counts[ni];
        H->send_offsets[ni + 1] = H->total_send_size;
        H->recv_offsets[ni + 1] = H->total_recv_size;
    }

    // Allocate and copy flattened index arrays
    H->flat_send_indices = (int *)malloc(H->total_send_size * sizeof(int));
    H->flat_recv_indices = (int *)malloc(H->total_recv_size * sizeof(int));

    memcpy(H->flat_send_indices, flat_send_indices, H->total_send_size * sizeof(int));
    memcpy(H->flat_recv_indices, flat_recv_indices, H->total_recv_size * sizeof(int));

    // Allocate communication buffers
    // 3 quantities per element: stage, xmom, ymom centroid values
    H->send_buffer = (double *)malloc(3 * H->total_send_size * sizeof(double));
    H->recv_buffer = (double *)malloc(3 * H->total_recv_size * sizeof(double));

    // Allocate MPI request array
    H->requests = (MPI_Request *)malloc(2 * num_neighbors * sizeof(MPI_Request));

    if (GD->rank == 0) {
        printf("GPU halo exchange initialized:\n");
        printf("  Neighbors: %d\n", num_neighbors);
        printf("  Total send: %d elements\n", H->total_send_size);
        printf("  Total recv: %d elements\n", H->total_recv_size);
    }

    return 0;
}

void gpu_halo_finalize(struct gpu_domain *GD) {
    struct halo_exchange *H = &GD->halo;

    if (H->neighbor_ranks) free(H->neighbor_ranks);
    if (H->send_counts) free(H->send_counts);
    if (H->recv_counts) free(H->recv_counts);
    if (H->send_offsets) free(H->send_offsets);
    if (H->recv_offsets) free(H->recv_offsets);
    if (H->flat_send_indices) free(H->flat_send_indices);
    if (H->flat_recv_indices) free(H->flat_recv_indices);
    if (H->send_buffer) free(H->send_buffer);
    if (H->recv_buffer) free(H->recv_buffer);
    if (H->requests) free(H->requests);

    H->num_neighbors = 0;
    H->neighbor_ranks = NULL;
    H->send_counts = NULL;
    H->recv_counts = NULL;
    H->send_offsets = NULL;
    H->recv_offsets = NULL;
    H->flat_send_indices = NULL;
    H->flat_recv_indices = NULL;
    H->send_buffer = NULL;
    H->recv_buffer = NULL;
    H->requests = NULL;
}


// ============================================================================
// Ghost Exchange - Key MPI Function
// ============================================================================

// Exchange ghost cell data between MPI ranks
// Adapted from miniapp_mpi.c exchange_halo()
void gpu_exchange_ghosts(struct gpu_domain *GD) {
    struct halo_exchange *H = &GD->halo;

    if (H->num_neighbors == 0) return;

    int send_size = H->total_send_size;
    int recv_size = H->total_recv_size;

    double *stage = GD->D.stage_centroid_values;
    double *xmom = GD->D.xmom_centroid_values;
    double *ymom = GD->D.ymom_centroid_values;
    double *send_buf = H->send_buffer;
    double *recv_buf = H->recv_buffer;
    int *flat_send = H->flat_send_indices;
    int *flat_recv = H->flat_recv_indices;

    // Pack send buffer on GPU
    #pragma omp target teams distribute parallel for
    for (int idx = 0; idx < send_size; idx++) {
        int k = flat_send[idx];  // Local element index
        send_buf[3*idx + 0] = stage[k];
        send_buf[3*idx + 1] = xmom[k];
        send_buf[3*idx + 2] = ymom[k];
    }

#ifdef GPU_AWARE_MPI
    // GPU-aware MPI path: pass device pointers directly to MPI
    #pragma omp target data use_device_addr(send_buf, recv_buf)
    {
        int req_count = 0;
        int send_offset = 0, recv_offset = 0;

        // Post all receives first
        for (int ni = 0; ni < H->num_neighbors; ni++) {
            int partner = H->neighbor_ranks[ni];
            int count = H->recv_counts[ni];
            MPI_Irecv(&recv_buf[3*recv_offset], 3*count, MPI_DOUBLE,
                      partner, 0, GD->comm, &H->requests[req_count++]);
            recv_offset += count;
        }

        // Post all sends
        for (int ni = 0; ni < H->num_neighbors; ni++) {
            int partner = H->neighbor_ranks[ni];
            int count = H->send_counts[ni];
            MPI_Isend(&send_buf[3*send_offset], 3*count, MPI_DOUBLE,
                      partner, 0, GD->comm, &H->requests[req_count++]);
            send_offset += count;
        }

        // Wait for all communication to complete
        MPI_Waitall(req_count, H->requests, MPI_STATUSES_IGNORE);
    }

#else
    // Non-GPU-aware MPI path: transfer halo buffers through host
    // This is still efficient because halo is much smaller than full domain

    // Copy packed send buffer from GPU to host
    #pragma omp target update from(send_buf[0:3*send_size])

    // MPI communication on host
    int req_count = 0;
    int send_offset = 0, recv_offset = 0;

    // Post all receives first
    for (int ni = 0; ni < H->num_neighbors; ni++) {
        int partner = H->neighbor_ranks[ni];
        int count = H->recv_counts[ni];
        MPI_Irecv(&recv_buf[3*recv_offset], 3*count, MPI_DOUBLE,
                  partner, 0, GD->comm, &H->requests[req_count++]);
        recv_offset += count;
    }

    // Post all sends
    for (int ni = 0; ni < H->num_neighbors; ni++) {
        int partner = H->neighbor_ranks[ni];
        int count = H->send_counts[ni];
        MPI_Isend(&send_buf[3*send_offset], 3*count, MPI_DOUBLE,
                  partner, 0, GD->comm, &H->requests[req_count++]);
        send_offset += count;
    }

    // Wait for all communication to complete
    MPI_Waitall(req_count, H->requests, MPI_STATUSES_IGNORE);

    // Copy received halo data from host to GPU
    #pragma omp target update to(recv_buf[0:3*recv_size])
#endif

    // Unpack receive buffer on GPU
    #pragma omp target teams distribute parallel for
    for (int idx = 0; idx < recv_size; idx++) {
        int k = flat_recv[idx];  // Local ghost element index
        stage[k] = recv_buf[3*idx + 0];
        xmom[k] = recv_buf[3*idx + 1];
        ymom[k] = recv_buf[3*idx + 2];
    }
}


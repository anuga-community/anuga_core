// Native HIP STREAM, the hardware ceiling for one GCD.
//   hipcc -O3 --offload-arch=gfx90a stream_hip.cpp -o stream_hip
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void copy(const double *a, double *c, long n)  { for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) c[i] = a[i]; }
__global__ void scale(double *b, const double *c, double s, long n) { for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) b[i] = s * c[i]; }
__global__ void add(const double *a, const double *b, double *c, long n) { for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) c[i] = a[i] + b[i]; }
__global__ void triad(double *a, const double *b, const double *c, double s, long n) { for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) a[i] = b[i] + s * c[i]; }
__global__ void init(double *a, double *b, double *c, long n) { for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x) { a[i] = 1; b[i] = 2; c[i] = 0; } }

int main(int argc, char **argv) {
    long n = 1L << (argc > 1 ? atoi(argv[1]) : 28);
    double *a, *b, *c, s = 3.0;
    hipMalloc(&a, n * 8); hipMalloc(&b, n * 8); hipMalloc(&c, n * 8);
    int bs = 256; long nb = (n + bs - 1) / bs;
    int grids[2] = {(int)(nb < 2147483647 ? nb : 2147483647), 110 * 4 * 8};  // full grid, and CUs x 32 blocks (grid-stride)
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    init<<<grids[0], bs>>>(a, b, c, n); hipDeviceSynchronize();
    const char *name[4] = {"copy", "scale", "add", "triad"};
    const double words[4] = {2, 2, 3, 3};
    for (int g = 0; g < 2; g++) {
        double best[4] = {1e30, 1e30, 1e30, 1e30};
        for (int r = 0; r < 20; r++)
            for (int k = 0; k < 4; k++) {
                hipEventRecord(e0);
                switch (k) {
                case 0: copy<<<grids[g], bs>>>(a, c, n); break;
                case 1: scale<<<grids[g], bs>>>(b, c, s, n); break;
                case 2: add<<<grids[g], bs>>>(a, b, c, n); break;
                case 3: triad<<<grids[g], bs>>>(a, b, c, s, n); break;
                }
                hipEventRecord(e1); hipEventSynchronize(e1);
                float ms; hipEventElapsedTime(&ms, e0, e1);
                if (r > 0 && ms * 1e-3 < best[k]) best[k] = ms * 1e-3;
            }
        printf("hip  n=%ld grid=%d\n", n, grids[g]);
        for (int k = 0; k < 4; k++)
            printf("  %-6s %8.1f GB/s\n", name[k], words[k] * n * 8 / best[k] / 1e9);
    }
    return 0;
}

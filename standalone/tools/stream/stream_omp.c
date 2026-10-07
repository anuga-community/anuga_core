/* STREAM on one GPU via OpenMP target offload, same loop form as the miniapp
 * (teams distribute parallel for). Arrays live on the device (omp_target_alloc).
 *   amdclang -O3 -fopenmp --offload-arch=gfx90a stream_omp.c -o stream_omp
 *   ./stream_omp [log2 N]          (default 2^28 doubles = 2 GiB per array) */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    long n = 1L << (argc > 1 ? atoi(argv[1]) : 28);
    int dev = omp_get_default_device(), reps = 20;
    double *a = omp_target_alloc(n * sizeof(double), dev);
    double *b = omp_target_alloc(n * sizeof(double), dev);
    double *c = omp_target_alloc(n * sizeof(double), dev);
    double s = 3.0;
    #pragma omp target teams distribute parallel for is_device_ptr(a, b, c)
    for (long i = 0; i < n; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    const char *name[4] = {"copy", "scale", "add", "triad"};
    const double words[4] = {2, 2, 3, 3};
    double best[4] = {1e30, 1e30, 1e30, 1e30};
    for (int r = 0; r < reps; r++) {
        for (int k = 0; k < 4; k++) {
            double t = omp_get_wtime();
            switch (k) {
            case 0:
                #pragma omp target teams distribute parallel for is_device_ptr(a, c)
                for (long i = 0; i < n; i++) c[i] = a[i];
                break;
            case 1:
                #pragma omp target teams distribute parallel for is_device_ptr(b, c)
                for (long i = 0; i < n; i++) b[i] = s * c[i];
                break;
            case 2:
                #pragma omp target teams distribute parallel for is_device_ptr(a, b, c)
                for (long i = 0; i < n; i++) c[i] = a[i] + b[i];
                break;
            case 3:
                #pragma omp target teams distribute parallel for is_device_ptr(a, b, c)
                for (long i = 0; i < n; i++) a[i] = b[i] + s * c[i];
                break;
            }
            t = omp_get_wtime() - t;
            if (r > 0 && t < best[k]) best[k] = t;      /* first rep is warmup */
        }
    }
    printf("omp  n=%ld (%.2f GiB/array)\n", n, n * 8.0 / (1 << 30));
    for (int k = 0; k < 4; k++)
        printf("  %-6s %8.1f GB/s\n", name[k], words[k] * n * 8 / best[k] / 1e9);
    omp_target_free(a, dev); omp_target_free(b, dev); omp_target_free(c, dev);
    return 0;
}

#include "snapshot.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *field_names[SNAP_NFIELDS] = {
    "stage", "xmomentum", "ymomentum", "height", "elevation"
};

static const double *field_ptr(const struct gpu_domain *GD, int f) {
    switch (f) {
        case SNAP_STAGE:  return GD->D.stage_centroid_values;
        case SNAP_XMOM:   return GD->D.xmom_centroid_values;
        case SNAP_YMOM:   return GD->D.ymom_centroid_values;
        case SNAP_HEIGHT: return GD->D.height_centroid_values;
        default:          return GD->D.bed_centroid_values;
    }
}

int snapshot_save(const char *path, const struct gpu_domain *GD,
                  const int64_t *orig_id,
                  int64_t nx, int64_t ny, int which_case,
                  int64_t total_steps, double t, double last_dt) {
    FILE *fp = fopen(path, "wb");
    if (!fp) { perror(path); return 1; }

    snap_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, SNAP_MAGIC, 8);
    h.version     = SNAP_VERSION;
    h.which_case  = which_case;
    h.n           = GD->D.number_of_elements;
    h.nb          = GD->D.boundary_length;
    h.nx          = nx;
    h.ny          = ny;
    h.total_steps = total_steps;
    h.t           = t;
    h.last_dt     = last_dt;

    if (fwrite(&h, sizeof(h), 1, fp) != 1) { perror(path); fclose(fp); return 1; }

    // Scatter into canonical order when the run is permuted
    double *canon = NULL;
    if (orig_id) {
        canon = (double *)malloc((size_t)h.n * sizeof(double));
        if (!canon) { fclose(fp); return 1; }
    }
    for (int f = 0; f < SNAP_NFIELDS; f++) {
        const double *src = field_ptr(GD, f);
        if (orig_id) {
            for (int64_t k = 0; k < h.n; k++) canon[orig_id[k]] = src[k];
            src = canon;
        }
        if (fwrite(src, sizeof(double), (size_t)h.n, fp) != (size_t)h.n) {
            perror(path); free(canon); fclose(fp); return 1;
        }
    }
    free(canon);
    fclose(fp);
    printf("saved snapshot -> %s  (%lld triangles, %lld steps, t = %.9g)\n",
           path, (long long)h.n, (long long)total_steps, t);
    return 0;
}

int snapshot_check(const char *path, const struct gpu_domain *GD,
                   const int64_t *orig_id,
                   double rtol, double atol) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 1; }

    snap_header h;
    if (fread(&h, sizeof(h), 1, fp) != 1) { perror(path); fclose(fp); return 1; }
    if (memcmp(h.magic, SNAP_MAGIC, 8) != 0 || h.version != SNAP_VERSION) {
        fprintf(stderr, "bench: %s is not a v%d snapshot\n", path, SNAP_VERSION);
        fclose(fp); return 1;
    }
    if (h.n != GD->D.number_of_elements) {
        fprintf(stderr, "bench: snapshot has %lld triangles, this run has %lld\n",
                (long long)h.n, (long long)GD->D.number_of_elements);
        fclose(fp); return 1;
    }

    double *ref = (double *)malloc((size_t)h.n * sizeof(double));
    if (!ref) { fclose(fp); return 1; }

    printf("\nverification against %s (%lld steps, t = %.9g)\n",
           path, (long long)h.total_steps, h.t);
    printf("  %-10s %14s %14s %14s\n", "field", "max abs diff", "rel to scale", "rms diff");

    int failed = 0;
    for (int f = 0; f < SNAP_NFIELDS; f++) {
        if (fread(ref, sizeof(double), (size_t)h.n, fp) != (size_t)h.n) {
            perror(path); free(ref); fclose(fp); return 1;
        }
        const double *cur = field_ptr(GD, f);

        double max_abs = 0.0, sumsq = 0.0, field_scale = 0.0;
        int64_t worst = -1;
        int bad = 0;
        for (int64_t k = 0; k < h.n; k++) {
            const double r = ref[orig_id ? orig_id[k] : k];   // file is canonical order
            const double d = fabs(cur[k] - r);
            const double scale = fabs(r);
            sumsq += d * d;
            if (d > max_abs) { max_abs = d; worst = k; }
            if (scale > field_scale) field_scale = scale;
            if (d > atol + rtol * scale) bad = 1;
        }
        const double rms = sqrt(sumsq / (double)h.n);
        // Pointwise relative error is meaningless where the reference is ~0
        // (momentum in still water), so report the error against the largest
        // value in the field instead.
        const double max_rel = field_scale > 0.0 ? max_abs / field_scale : 0.0;

        printf("  %-10s %14.6e %14.6e %14.6e%s\n",
               field_names[f], max_abs, max_rel, rms, bad ? "   FAIL" : "");
        if (bad) {
            failed = 1;
            if (worst >= 0)
                printf("             worst at triangle %lld: got %.17g, expected %.17g\n",
                       (long long)worst, cur[worst],
                       ref[orig_id ? orig_id[worst] : worst]);
        }
    }

    free(ref);
    fclose(fp);
    printf("  tolerance: atol %g, rtol %g  ->  %s\n\n",
           atol, rtol, failed ? "MISMATCH" : "OK");
    return failed;
}

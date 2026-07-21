#include "voxtral.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <time.h>

extern int vox_verbose;

static double elapsed_ms(struct timespec start, struct timespec end) {
    return (double)(end.tv_sec - start.tv_sec) * 1000.0 +
           (double)(end.tv_nsec - start.tv_nsec) / 1000000.0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: voxtral-c-mps-lifecycle MODEL_DIR\n");
        return 2;
    }

    vox_verbose = 0;
    struct timespec load_start;
    struct timespec load_end;
    struct timespec destroy_end;
    clock_gettime(CLOCK_MONOTONIC, &load_start);
    vox_ctx_t *context = vox_load(argv[1]);
    clock_gettime(CLOCK_MONOTONIC, &load_end);
    if (context == NULL) {
        fprintf(stderr, "vox_load failed\n");
        return 1;
    }

    vox_free(context);
    clock_gettime(CLOCK_MONOTONIC, &destroy_end);

    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        perror("getrusage");
        return 1;
    }

    printf(
        "{\"schema_version\":\"1.0.0\",\"status\":\"ok\","
        "\"boundary\":\"antirez-voxtral-c-mps-control\","
        "\"load_ms\":%.6f,\"destroy_ms\":%.6f,"
        "\"peak_rss_bytes\":%ld}\n",
        elapsed_ms(load_start, load_end),
        elapsed_ms(load_end, destroy_end),
        usage.ru_maxrss);
    return 0;
}

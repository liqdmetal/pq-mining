// Chitin-PoW CPU baseline: the SAME two phases, same parameters.
//
// This exists so the CUDA numbers mean something. Comparing a hand-tuned
// CUDA kernel against pure-Python would be dishonest -- it would measure
// Python, not the algorithm. Both sides here are -O3 C with identical
// arithmetic, identical mixer, identical table layout.
//
// Compile:
//   gcc -O3 -march=native -fopenmp -o cpu_bench cpu_bench.c
//   (or without -fopenmp for single-threaded)

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define N_ROWS    4
#define M_COLS    64
#define Q         97
#define SPARSITY  8

static inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

/* ---------------------------------------------------------------- PHASE 1
 * Batch syndrome generation -- identical arithmetic to CUDA kernel 1. */
static double bench_batch(const uint32_t* matrix, uint32_t candidates,
                          int threads) {
    uint32_t* out = (uint32_t*)malloc((size_t)candidates * N_ROWS * 4);
    if (!out) { printf("alloc failed\n"); exit(1); }

    double start = now_seconds();
#ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (long long idx = 0; idx < (long long)candidates; ++idx) {
        int32_t vec[M_COLS];
        memset(vec, 0, sizeof(vec));

        uint32_t h = mix32((uint32_t)idx);
        for (int s = 0; s < SPARSITY; ++s) {
            h = mix32(h + (uint32_t)s);
            uint32_t pos = h % M_COLS;
            int32_t sign = (h & 0x10000U) ? 1 : -1;
            vec[pos] += sign;
        }
        for (int r = 0; r < N_ROWS; ++r) {
            uint64_t acc = 0;
            const uint32_t* row = matrix + (size_t)r * M_COLS;
            for (int c = 0; c < M_COLS; ++c) {
                int32_t v = vec[c];
                if (v) acc += (uint64_t)row[c] * (uint64_t)((v % Q + Q) % Q);
            }
            out[(size_t)idx * N_ROWS + r] = (uint32_t)(acc % Q);
        }
    }
    double elapsed = now_seconds() - start;
    free(out);
    return candidates / elapsed;
}

/* ---------------------------------------------------------------- PHASE 2
 * Data-dependent random walk -- identical to CUDA kernel 2. */
static double bench_walk(const uint32_t* matrix, uint32_t* table,
                         uint32_t table_mask, uint32_t walkers,
                         uint32_t steps, int threads) {
    uint32_t* hits = (uint32_t*)calloc(walkers, 4);
    if (!hits) { printf("alloc failed\n"); exit(1); }

    double start = now_seconds();
#ifdef _OPENMP
    #pragma omp parallel for num_threads(threads) schedule(static)
#endif
    for (long long w = 0; w < (long long)walkers; ++w) {
        int32_t vec[M_COLS];
        memset(vec, 0, sizeof(vec));
        uint32_t synd[N_ROWS];
        memset(synd, 0, sizeof(synd));

        uint32_t position = mix32(12345u + (uint32_t)w) % M_COLS;
        uint32_t local_hits = 0;

        for (uint32_t step = 0; step < steps; ++step) {
            int32_t sign = (step & 1) ? -1 : 1;
            for (int r = 0; r < N_ROWS; ++r) {
                uint32_t col = matrix[(size_t)r * M_COLS + position];
                if (sign > 0) synd[r] = (synd[r] + col) % Q;
                else          synd[r] = (synd[r] + Q - col) % Q;
            }
            vec[position] += sign;

            uint32_t key = 0;
            for (int r = 0; r < N_ROWS; ++r) key = key * Q + synd[r];
            uint32_t slot = mix32(key) & table_mask;

            uint32_t previous = table[slot];
            uint32_t packed = (key << 8) | ((uint32_t)w & 0xFF);
            if (previous == packed) local_hits++;
            table[slot] = packed;

            position = (position + 1u + (previous & 0x1F)
                        + (synd[0] % M_COLS)) % M_COLS;
        }
        hits[w] = local_hits;
    }
    double elapsed = now_seconds() - start;
    free(hits);
    return ((double)walkers * (double)steps) / elapsed;
}

int main(int argc, char** argv) {
    uint32_t candidates = (argc > 1) ? (uint32_t)atoi(argv[1]) : 4194304u;
    uint32_t walkers    = (argc > 2) ? (uint32_t)atoi(argv[2]) : 65536u;
    uint32_t steps      = (argc > 3) ? (uint32_t)atoi(argv[3]) : 512u;
    int table_bits      = (argc > 4) ? atoi(argv[4]) : 26;
    int threads         = (argc > 5) ? atoi(argv[5]) : 12;

    uint32_t matrix[N_ROWS * M_COLS];
    for (int r = 0; r < N_ROWS; ++r)
        for (int c = 0; c < M_COLS; ++c)
            matrix[r * M_COLS + c] = mix32((uint32_t)(r * 1000 + c)) % Q;

    size_t entries = (size_t)1 << table_bits;
    uint32_t* table = (uint32_t*)calloc(entries, 4);
    if (!table) { printf("table alloc failed\n"); return 1; }

#ifdef _OPENMP
    printf("openmp            : yes, %d threads requested\n", threads);
#else
    printf("openmp            : no (single-threaded)\n");
    threads = 1;
#endif
    printf("table             : 2^%d entries = %.0f MB\n", table_bits,
           entries * 4.0 / (1024.0 * 1024.0));
    printf("\n");

    /* warm up */
    bench_batch(matrix, 65536u, threads);
    double batch_rate = bench_batch(matrix, candidates, threads);
    printf("PHASE 1  batch syndrome generation\n");
    printf("  candidates      : %u\n", candidates);
    printf("  rate            : %.3e candidates/s\n", batch_rate);
    printf("\n");

    bench_walk(matrix, table, (uint32_t)(entries - 1), 1024u, 8u, threads);
    double walk_rate = bench_walk(matrix, table, (uint32_t)(entries - 1),
                                  walkers, steps, threads);
    printf("PHASE 2  data-dependent random walk\n");
    printf("  walkers x steps : %u x %u = %.3e steps\n", walkers, steps,
           (double)walkers * (double)steps);
    printf("  rate            : %.3e steps/s\n", walk_rate);
    printf("\n");

    printf("MACHINE-READABLE\n");
    printf("cpu_batch_rate=%.6e\n", batch_rate);
    printf("cpu_walk_rate=%.6e\n", walk_rate);

    free(table);
    return 0;
}

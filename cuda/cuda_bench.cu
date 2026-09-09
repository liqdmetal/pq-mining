// Chitin-PoW CUDA benchmark: does a GPU beat a CPU on each phase?
//
// THE EXPERIMENT
// --------------
// Two kernels, deliberately chosen to answer two different questions.
//
// KERNEL 1 -- batch syndrome generation (the naive miner's Phase A).
//   Each thread takes an independent candidate index, derives a sparse
//   vector, and computes A*z mod q. No shared state, no branches that
//   diverge, perfectly coalesced-ish. This is the shape a GPU is built for,
//   so if the GPU does NOT win here something is wrong with the harness.
//
// KERNEL 2 -- data-dependent random walk (the balanced miner's inner loop).
//   Each thread walks its own trajectory through a large table: incremental
//   syndrome update, hash-table probe, and the NEXT flip position depends on
//   what the probe returned. Random global-memory access, divergent
//   branches, serial dependence within each walk. This is the shape that is
//   supposed to resist fixed-function hardware.
//
// The comparison that matters is KERNEL 2's speedup vs KERNEL 1's. If
// kernel 1 gives (say) 100x and kernel 2 gives 3x, the redesign genuinely
// moved the work into a regime where the GPU's advantage collapses. If both
// give 100x, the memory-hardness claim is false and I should say so.
//
// Compile:
//   nvcc -O3 -arch=sm_86 -o cuda_bench cuda_bench.cu
//
// NOTE ON HONESTY: this measures MY implementations. A better GPU
// programmer could improve both kernels; the ratio between them is the
// robust signal, not the absolute numbers.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do {                                                  \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
        printf("CUDA error %s at line %d\n", cudaGetErrorString(err),     \
               __LINE__);                                                 \
        exit(1);                                                          \
    }                                                                     \
} while (0)

// Toy parameters matching pqmine's benchmark settings.
#define N_ROWS    4
#define M_COLS    64
#define Q         97
#define SPARSITY  8

// A cheap deterministic mixer standing in for SHAKE256 candidate
// derivation. Using the real XOF on-device would dominate both sides
// equally and obscure the memory behaviour we are measuring.
__device__ __host__ inline uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// ---------------------------------------------------------------- KERNEL 1
// Independent batch syndrome generation: the GPU-favourable shape.
__global__ void kernel_batch_syndromes(const uint32_t* __restrict__ matrix,
                                       uint32_t* __restrict__ out,
                                       uint32_t base_index,
                                       uint32_t count) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    uint32_t index = base_index + tid;
    int32_t vec[M_COLS];
    for (int i = 0; i < M_COLS; ++i) vec[i] = 0;

    // Sparse candidate: SPARSITY signed unit bumps at mixed positions.
    uint32_t h = mix32(index);
    for (int s = 0; s < SPARSITY; ++s) {
        h = mix32(h + s);
        uint32_t pos = h % M_COLS;
        int32_t sign = (h & 0x10000U) ? 1 : -1;
        vec[pos] += sign;
    }

    // Full mat-vec: N_ROWS * M_COLS modular multiplies.
    for (int r = 0; r < N_ROWS; ++r) {
        uint64_t acc = 0;
        const uint32_t* row = matrix + (size_t)r * M_COLS;
        for (int c = 0; c < M_COLS; ++c) {
            int32_t v = vec[c];
            if (v) acc += (uint64_t)row[c] * (uint64_t)((v % Q + Q) % Q);
        }
        out[(size_t)tid * N_ROWS + r] = (uint32_t)(acc % Q);
    }
}

// ---------------------------------------------------------------- KERNEL 2
// Data-dependent random walk over a large table: the resistant shape.
__global__ void kernel_dependent_walk(const uint32_t* __restrict__ matrix,
                                      uint32_t* __restrict__ table,
                                      uint32_t table_mask,
                                      uint32_t steps,
                                      uint32_t seed_base,
                                      uint32_t* __restrict__ hits) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t vec[M_COLS];
    for (int i = 0; i < M_COLS; ++i) vec[i] = 0;

    uint32_t synd[N_ROWS];
    for (int r = 0; r < N_ROWS; ++r) synd[r] = 0;

    uint32_t position = mix32(seed_base + tid) % M_COLS;
    uint32_t local_hits = 0;

    for (uint32_t step = 0; step < steps; ++step) {
        // --- incremental syndrome update: s' = s +/- A[:, position]
        int32_t sign = (step & 1) ? -1 : 1;
        for (int r = 0; r < N_ROWS; ++r) {
            uint32_t col = matrix[(size_t)r * M_COLS + position];
            if (sign > 0) synd[r] = (synd[r] + col) % Q;
            else          synd[r] = (synd[r] + Q - col) % Q;
        }
        vec[position] += sign;

        // --- table probe: RANDOM ACCESS, this is the bottleneck by design
        uint32_t key = 0;
        for (int r = 0; r < N_ROWS; ++r) key = key * Q + synd[r];
        uint32_t slot = mix32(key) & table_mask;

        uint32_t previous = table[slot];       // dependent global load
        uint32_t packed = (key << 8) | (tid & 0xFF);
        if (previous == packed) {
            local_hits++;                      // divergent branch
        }
        table[slot] = packed;                  // dependent global store

        // --- DATA DEPENDENCE: next position depends on the loaded value,
        // so the walk cannot be precomputed or pipelined ahead.
        position = (position + 1u + (previous & 0x1F) + (synd[0] % M_COLS))
                   % M_COLS;
    }
    hits[tid] = local_hits;
}

int main(int argc, char** argv) {
    uint32_t candidates = (argc > 1) ? (uint32_t)atoi(argv[1]) : 4194304u;
    uint32_t walk_threads = (argc > 2) ? (uint32_t)atoi(argv[2]) : 65536u;
    uint32_t walk_steps = (argc > 3) ? (uint32_t)atoi(argv[3]) : 512u;
    int table_bits = (argc > 4) ? atoi(argv[4]) : 26;   // 2^26 * 4B = 256 MB

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device            : %s\n", prop.name);
    printf("SMs               : %d\n", prop.multiProcessorCount);
    printf("mem clock / bus   : %d kHz / %d-bit\n",
           prop.memoryClockRate, prop.memoryBusWidth);
    double bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6;
    printf("peak bandwidth    : %.0f GB/s\n", bw);
    printf("\n");

    // Host matrix, deterministic so the CPU side can match it exactly.
    uint32_t h_matrix[N_ROWS * M_COLS];
    for (int r = 0; r < N_ROWS; ++r)
        for (int c = 0; c < M_COLS; ++c)
            h_matrix[r * M_COLS + c] = mix32((uint32_t)(r * 1000 + c)) % Q;

    uint32_t* d_matrix = nullptr;
    CHECK(cudaMalloc(&d_matrix, sizeof(h_matrix)));
    CHECK(cudaMemcpy(d_matrix, h_matrix, sizeof(h_matrix),
                     cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    // ------------------------------------------------ kernel 1 measurement
    uint32_t* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, (size_t)candidates * N_ROWS * sizeof(uint32_t)));

    int threads = 256;
    int blocks = (int)((candidates + threads - 1) / threads);

    // warm up
    kernel_batch_syndromes<<<blocks, threads>>>(d_matrix, d_out, 0, candidates);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    kernel_batch_syndromes<<<blocks, threads>>>(d_matrix, d_out, 0, candidates);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    double gpu_batch_rate = candidates / (ms / 1000.0);
    printf("KERNEL 1  batch syndrome generation (GPU-favourable shape)\n");
    printf("  candidates      : %u\n", candidates);
    printf("  time            : %.3f ms\n", ms);
    printf("  rate            : %.3e candidates/s\n", gpu_batch_rate);
    printf("\n");

    // ------------------------------------------------ kernel 2 measurement
    size_t table_entries = (size_t)1 << table_bits;
    uint32_t* d_table = nullptr;
    CHECK(cudaMalloc(&d_table, table_entries * sizeof(uint32_t)));
    CHECK(cudaMemset(d_table, 0, table_entries * sizeof(uint32_t)));

    uint32_t* d_hits = nullptr;
    CHECK(cudaMalloc(&d_hits, (size_t)walk_threads * sizeof(uint32_t)));

    int wblocks = (int)((walk_threads + threads - 1) / threads);

    kernel_dependent_walk<<<wblocks, threads>>>(
        d_matrix, d_table, (uint32_t)(table_entries - 1), 8u, 1u, d_hits);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    kernel_dependent_walk<<<wblocks, threads>>>(
        d_matrix, d_table, (uint32_t)(table_entries - 1),
        walk_steps, 12345u, d_hits);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    double walk_ops = (double)walk_threads * (double)walk_steps;
    double gpu_walk_rate = walk_ops / (ms / 1000.0);
    printf("KERNEL 2  data-dependent random walk (resistant shape)\n");
    printf("  table           : 2^%d entries = %.0f MB\n", table_bits,
           table_entries * 4.0 / (1024.0 * 1024.0));
    printf("  threads x steps : %u x %u = %.3e steps\n",
           walk_threads, walk_steps, walk_ops);
    printf("  time            : %.3f ms\n", ms);
    printf("  rate            : %.3e steps/s\n", gpu_walk_rate);
    printf("\n");

    printf("MACHINE-READABLE\n");
    printf("gpu_batch_rate=%.6e\n", gpu_batch_rate);
    printf("gpu_walk_rate=%.6e\n", gpu_walk_rate);

    CHECK(cudaFree(d_matrix));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_table));
    CHECK(cudaFree(d_hits));
    return 0;
}

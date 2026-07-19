// MoEx: GPU bottleneck diagnostic. No perf-counter permission needed (ncu
// on this machine returns ERR_NVGPUCTRPERM without admin). Three checks:
//   1. cudaOccupancyMaxActiveBlocksPerMultiprocessor + cudaFuncGetAttributes
//      on the real production kernels — exact register/shared-mem-limited
//      occupancy, no guessing.
//   2. A raw sequential-read bandwidth microbenchmark — this machine's real
//      achievable HBM bandwidth, not the vendor spec sheet number.
//   3. A microbenchmark reproducing row_dot's exact Q4_K lane-access pattern
//      on a synthetic buffer — isolates whether the access pattern itself
//      (vs. raw hardware) is leaving bandwidth on the table.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda/gemv.cuh"
#include "cuda/kernels.cuh"

using namespace moex;

static void occ(const char* name, const void* kfunc, int block_size, size_t dyn_smem) {
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, kfunc);
  int max_blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kfunc, block_size, dyn_smem);
  int dev;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  int warps_per_block = block_size / 32;
  int active_warps = max_blocks * warps_per_block;
  int max_warps_per_sm = prop.maxThreadsPerMultiProcessor / 32;
  double occ_pct = 100.0 * active_warps / max_warps_per_sm;
  std::printf("%-32s regs=%3d smem_static=%4zu smem_dyn=%6zu -> %d blocks/SM, "
              "%d/%d warps active (%.1f%% occupancy)\n",
              name, attr.numRegs, attr.sharedSizeBytes, dyn_smem, max_blocks,
              active_warps, max_warps_per_sm, occ_pct);
}

// ---- raw bandwidth probes --------------------------------------------------
__global__ void bw_read_kernel(const float4* buf, size_t n4, float* out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 acc = {0, 0, 0, 0};
  for (; i < n4; i += stride) {
    float4 v = buf[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (acc.x + acc.y + acc.z + acc.w == 12345.6789f)  // never true; prevents DCE
    out[blockIdx.x] = acc.x;
}

// Reproduces row_dot's Q4_K access exactly: 32 lanes x 8 groups, each lane
// reads q[(g>>1)*32+lane] -- one byte per lane per group, fully coalesced
// within a group (consecutive lanes -> consecutive addresses). Times the
// SAME total-bytes-read workload as bw_read_kernel for a fair GB/s compare.
__global__ void bw_q4k_pattern_kernel(const uint8_t* blocks, size_t n_blocks,
                                      float* out) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int nwarps = (gridDim.x * blockDim.x) / 32;
  float acc = 0.0f;
  for (size_t b = warp; b < n_blocks; b += nwarps) {
    const uint8_t* p = blocks + b * BQ4_K;
    const float d = rd_h(p);
    const uint8_t* q = p + 16;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const uint8_t byte = q[(g >> 1) * 32 + lane];
      acc += d * (float)(byte & 0x0F);
    }
  }
  if (acc == -999999.0f) out[warp % 1024] = acc;  // prevent DCE
}

int main() {
  cudaDeviceProp prop;
  int dev;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  int mem_khz = 0, bus_bits = 0;
  cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
  std::printf("=== GPU: %s | %d SMs | mem_clock=%d kHz bus=%d bit -> %.0f GB/s theoretical (spec) | maxThreads/SM=%d ===\n",
              prop.name, prop.multiProcessorCount, mem_khz, bus_bits,
              2.0 * mem_khz * (bus_bits / 8) / 1e6,
              prop.maxThreadsPerMultiProcessor);

  std::printf("\n=== 1. Occupancy of production kernels (exact, via CUDA runtime API) ===\n");
  const int D = 2048, DFF = 768, NEXP = 8;
  occ("expert_group_gate_up_silu", (void*)expert_group_gate_up_silu_kernel, 256,
      D * sizeof(float));
  occ("expert_group_down", (void*)expert_group_down_kernel, 256, DFF * sizeof(float));
  occ("qkv_q_kernel", (void*)qkv_q_kernel, 256, D * sizeof(float));
  occ("gemv_q_residual", (void*)gemv_q_residual, 256, D * sizeof(float));
  occ("router_topk8_kernel", (void*)router_topk8_kernel, 128, 0);
  occ("argmax_stage1_kernel", (void*)argmax_stage1_kernel, 256, 0);
  (void)NEXP;

  std::printf("\n=== 2. Raw achievable bandwidth on THIS card (not spec sheet) ===\n");
  const size_t buf_bytes = 512ull << 20;  // 512 MB
  float4* d_buf;
  float* d_out;
  cudaMalloc(&d_buf, buf_bytes);
  cudaMalloc(&d_out, 4096 * sizeof(float));
  cudaMemset(d_buf, 1, buf_bytes);
  const size_t n4 = buf_bytes / sizeof(float4);
  cudaEvent_t t0, t1;
  cudaEventCreate(&t0);
  cudaEventCreate(&t1);
  const int reps = 20;
  int grid = prop.multiProcessorCount * 32;
  bw_read_kernel<<<grid, 256>>>(d_buf, n4, d_out);  // warmup
  cudaDeviceSynchronize();
  cudaEventRecord(t0);
  for (int i = 0; i < reps; ++i) bw_read_kernel<<<grid, 256>>>(d_buf, n4, d_out);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  float ms = 0;
  cudaEventElapsedTime(&ms, t0, t1);
  double raw_gbps = (double)buf_bytes * reps / (ms / 1000.0) / 1e9;
  std::printf("%-32s %8.2f GB/s  (%.3f ms/rep, fully coalesced float4 stream)\n",
              "raw_sequential_read", raw_gbps, ms / reps);

  std::printf("\n=== 3. row_dot's actual Q4_K access pattern, same total bytes ===\n");
  const size_t n_blocks = buf_bytes / BQ4_K;
  const uint8_t* d_blocks = (const uint8_t*)d_buf;
  const int block_threads = 256;
  const int nwarps_total = grid * (block_threads / 32);
  bw_q4k_pattern_kernel<<<grid, block_threads>>>(d_blocks, n_blocks, d_out);
  cudaDeviceSynchronize();
  cudaEventRecord(t0);
  for (int i = 0; i < reps; ++i)
    bw_q4k_pattern_kernel<<<grid, block_threads>>>(d_blocks, n_blocks, d_out);
  cudaEventRecord(t1);
  cudaEventSynchronize(t1);
  cudaEventElapsedTime(&ms, t0, t1);
  // Bytes actually touched per block: 2 (d) + 8*32 (nibble bytes, some double
  // counted between groups sharing byte pairs -- q4k reads 128B of the 144B
  // block per warp pass) ~= 130B/block touched by the read pattern.
  double bytes_touched = (double)n_blocks * 130.0;
  double q4k_gbps = bytes_touched * reps / (ms / 1000.0) / 1e9;
  std::printf("%-32s %8.2f GB/s  (%.3f ms/rep, %d warps, lane-strided q4k reads)\n",
              "row_dot_q4k_pattern", q4k_gbps, ms / reps, nwarps_total);
  std::printf("\nefficiency vs raw achievable: %.1f%%\n", 100.0 * q4k_gbps / raw_gbps);

  std::printf("\n=== 4. Scattered arena reads via the REAL production kernels ===\n");
  std::printf("(isolates the theory: contiguous-buffer probes above don't\n");
  std::printf(" capture the real cost of 8 experts scattered across a live,\n");
  std::printf(" LRU-churned ~3 GiB arena -- this test uses the exact same\n");
  std::printf(" launch_expert_group_gate_up_silu/_down as production Forward::step)\n");
  {
    const uint64_t gate_bytes = 768ull * 8 * BQ4_K;   // d_ff=768, nsb=8, Q4_K
    const uint64_t up_bytes = gate_bytes;
    const uint64_t down_bytes = 2048ull * 3 * BQ6_K;  // D=2048, nsb=3, Q6_K
    auto align256 = [](uint64_t x) { return (x + 255) & ~255ull; };
    const uint64_t gate_off = 0;
    const uint64_t up_off = align256(gate_bytes);
    const uint64_t down_off = up_off + align256(up_bytes);
    const uint64_t slot_stride = down_off + align256(down_bytes);

    const uint32_t n_slots = 400;  // ~1.17 GiB arena -- big enough to spread
    void* arena;
    cudaMalloc(&arena, (uint64_t)n_slots * slot_stride);
    cudaMemset(arena, 0x11, (uint64_t)n_slots * slot_stride);
    std::printf("arena: %u slots x %.3f MiB = %.2f GiB (matches real bundle geometry)\n",
                n_slots, slot_stride / 1048576.0,
                n_slots * slot_stride / 1073741824.0);

    float *x, *gates, *outs;
    cudaMalloc(&x, D * sizeof(float));
    cudaMalloc(&gates, (size_t)8 * DFF * sizeof(float));
    cudaMalloc(&outs, (size_t)8 * D * sizeof(float));
    cudaMemset(x, 0x11, D * sizeof(float));

    ExpertDispatch h_disp;
    for (int j = 0; j < 8; ++j) h_disp.weight[j] = 1.0f / 8;

    auto build_dispatch = [&](bool scattered) {
      int slot_ids[8];
      if (scattered) {
        // Emulate post-warmup LRU churn: 8 experts at essentially random
        // slots across the whole arena (worst realistic case).
        for (int j = 0; j < 8; ++j) slot_ids[j] = rand() % n_slots;
      } else {
        for (int j = 0; j < 8; ++j) slot_ids[j] = j;  // best case: contiguous
      }
      for (int j = 0; j < 8; ++j) {
        uint8_t* base = (uint8_t*)arena + (uint64_t)slot_ids[j] * slot_stride;
        h_disp.gate[j] = base + gate_off;
        h_disp.up[j] = base + up_off;
        h_disp.down[j] = base + down_off;
      }
    };

    auto run = [&](const char* name, bool scattered) {
      const int reps = 30;
      cudaEvent_t t0, t1;
      cudaEventCreate(&t0);
      cudaEventCreate(&t1);
      build_dispatch(scattered);
      launch_expert_group_gate_up_silu(GT_Q4_K, h_disp, x, gates, 8, DFF, D,
                                       (int)(gate_bytes / DFF));
      launch_expert_group_down(GT_Q6_K, h_disp, gates, outs, 8, D, DFF,
                               (int)(down_bytes / D));
      cudaDeviceSynchronize();
      cudaEventRecord(t0);
      for (int i = 0; i < reps; ++i) {
        build_dispatch(scattered);  // fresh random slots each rep, like real tokens
        launch_expert_group_gate_up_silu(GT_Q4_K, h_disp, x, gates, 8, DFF, D,
                                         (int)(gate_bytes / DFF));
        launch_expert_group_down(GT_Q6_K, h_disp, gates, outs, 8, D, DFF,
                                 (int)(down_bytes / D));
      }
      cudaEventRecord(t1);
      cudaEventSynchronize(t1);
      float ms = 0;
      cudaEventElapsedTime(&ms, t0, t1);
      const double bytes_per_rep = 8.0 * (gate_bytes + up_bytes + down_bytes);
      const double gbps = bytes_per_rep * reps / (ms / 1000.0) / 1e9;
      std::printf("%-32s %8.2f GB/s  (%.3f ms/rep = one layer's 8 experts, %.2f MB/rep)\n",
                  name, gbps, ms / reps, bytes_per_rep / 1e6);
      return gbps;
    };

    double seq_gbps = run("8_experts_sequential_slots", false);
    double scat_gbps = run("8_experts_scattered_slots", true);
    std::printf("scattered/sequential ratio: %.1f%%\n", 100.0 * scat_gbps / seq_gbps);
    std::printf("scattered vs raw achievable (%.1f GB/s): %.1f%%\n", raw_gbps,
                100.0 * scat_gbps / raw_gbps);

    cudaFree(arena);
    cudaFree(x);
    cudaFree(gates);
    cudaFree(outs);
  }

  cudaFree(d_buf);
  cudaFree(d_out);
  return 0;
}

# joi LLM Server — Configuration & Optimization Notes

## Hardware

AMD Strix Halo (Ryzen AI Max 395), gfx1151, 128GB unified memory (125GB usable).
UMA architecture — GPU and CPU share the same physical RAM pool. The ~117GB "VRAM"
is a BIOS-configured slice of the same 125GB system RAM, not separate memory.

## Current Models (as of 2026-03-31)

| Model | Quant | Size on Disk | RSS | ctx-size | parallel | Port |
|-------|-------|-------------|-----|----------|----------|------|
| Qwen3.5-122B-A10B | Q4_K_M (3 shards) | ~73GB | ~90GB | 131072 | 2 | 3101 |
| Qwen3.5-9B | Q4_K_M | ~6GB | ~18GB | 8192 | 2 | 3102 |
| Qwen3-Embedding-8B | Q4_K_M | ~4.4GB | ~6.6GB | 8192 | 2 | 3103 |
| **Total** | | | **~115GB** | | | |

The mmproj-F16.gguf (867MB) is loaded on the 122B and 9B models for vision support.

## Current llama.cpp Version

Version 1, commit `f20469d`, built with Clang 22, ROCm, gfx1151.

## Memory Pressure

With all 3 models loaded: ~117GB / 125GB used, ~441MB in swap. This is tight but
functional. The 122B model's `--no-mmap` flag pins all weights in RAM (~73GB model
+ ~3GB KV cache + runtime overhead = ~90GB RSS).

### Actual Token Usage (spin-cycle project)

From llama-server logs (122B model, slot size = 65,536 tokens):

| Pipeline Step | Typical Input Tokens | Peak Total (in+out) | % of Slot |
|---------------|---------------------|---------------------|-----------|
| Extraction chunks | 14K-22K | ~22K | 34% |
| Research agent (iterative) | 5K-19K (grows) | ~19K | 29% |
| Judge | 5K-12K | ~12K | 18% |
| Synthesize verdict | 5K-9K | ~9K | 14% |
| Classify (batched) | 3K-5K | ~8K | 12% |

**Peak observed: 22,143 tokens (34% of 65K slot).** Current 131072 ctx-size
(65K per slot) has ~2x headroom over the worst case. Reducing to 98304 (48K/slot)
or 81920 (40K/slot) would save KV cache memory while maintaining safe headroom.

### KV Cache Size Estimate

Qwen3.5-122B-A10B hybrid architecture: 2 KV heads, 12 attention layers of 48 total.
Each checkpoint: ~149MB (from server logs). Total cache budget: ~8GB (from server
`cache state: 8192.000 MiB`).

## Optimization Opportunities

### 1. `rocm-wmma-tune` Branch (High Impact)

**Source:** https://github.com/lhl/llama.cpp/tree/rocm-wmma-tune
**PR:** https://github.com/ggml-org/llama.cpp/pull/16827 (rejected upstream, pending full WMMA rewrite)
**Companion benchmarks:** https://github.com/lhl/strix-halo-testing

Retunes rocWMMA FlashAttention kernels for RDNA3/3.5 (gfx1151):
- +29% prefill at 4K context
- +53% prefill at 16K context
- +66% prefill at 65K context
- Fixes -58% decode regression in stock WMMA path

Changes: increased block residency via `__launch_bounds__`, adaptive KQ stride,
WMMA restricted to prefill only (decode falls through to VEC/TILE paths).

**Pre-built Docker option:** `kyuz0/amd-strix-halo-toolboxes` containers tagged
`-rocwmma-improved` include this patch baked in.

### 2. Runtime Environment Variables

- `ROCBLAS_USE_HIPBLASLT=1` — switches to hipBLASLt kernels, ~2.2x pp improvement
- Check if this is set in current Docker compose

### 3. Build Flags (for next update)

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx1151" \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DCMAKE_BUILD_TYPE=Release
```

ROCm 7.x compiler regression workaround:
```bash
-DCMAKE_HIP_FLAGS="--rocm-path=/opt/rocm -mllvm --amdgpu-unroll-threshold-local=600"
```

### 4. Known Bugs Affecting Strix Halo

1. **UMA detection bug** ([#18159](https://github.com/ggml-org/llama.cpp/issues/18159)):
   Misreports available memory using `MemAvailable` instead of `hipMemGetInfo()`.
   Fix: [PR #20472](https://github.com/ggml-org/llama.cpp/pull/20472).

2. **ROCm 7.x compiler regression** ([#19984](https://github.com/ggml-org/llama.cpp/issues/19984)):
   Up to 3x slower than ROCm 6.4.4. Workaround: `-mllvm --amdgpu-unroll-threshold-local=600`.

3. **Slow loading past 64GB** ([#15018](https://github.com/ggml-org/llama.cpp/issues/15018)):
   Model weight loading extremely slow past 64GB mark on ROCm. `--no-mmap` helps.

4. **KV cache shared memory** ([#18011](https://github.com/ggml-org/llama.cpp/issues/18011)):
   KV cache always dumps to shared memory on UMA systems.

### 5. TurboQuant (Future — Not Ready)

KV cache compression to 3-4 bits (Google Research, ICLR 2026). Would give 4-5x
KV cache memory reduction. Community forks exist but have known HIP NaN bugs.
Upstream merge expected ~Q3 2026.

- Discussion: https://github.com/ggml-org/llama.cpp/discussions/20969
- Feature request: https://github.com/ggml-org/llama.cpp/issues/20977

### 6. Vulkan vs ROCm on Strix Halo

Vulkan RADV is 16-33% faster than ROCm for MoE at standard context. But at
long context (130K+), ROCm with Flash Attention is 3-4x faster. Since we run
`--ctx-size 131072`, ROCm is the right backend.

### 7. ik_llama.cpp

Advanced fork with SOTA quantization, better CPU perf, FlashMLA, fused MoE ops.
**Does NOT support ROCm/AMD GPUs.** CPU and CUDA only. Not usable for our setup.

## Reference Links

- [Strix Halo Wiki - llama.cpp with ROCm](https://strixhalo.wiki/AI/llamacpp-with-ROCm)
- [Strix Halo Wiki - llama.cpp Performance](https://strixhalo.wiki/AI/llamacpp-performance)
- [AMD ROCm Blog - Accelerating llama.cpp (Oct 2025)](https://rocm.blogs.amd.com/ecosystems-and-partners/llama-cpp-oct2025/README.html)
- [Discussion #15021 - llama.cpp on AMD ROCm](https://github.com/ggml-org/llama.cpp/discussions/15021)
- [kyuz0 Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
- [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)

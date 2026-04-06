# joi LLM Server — Configuration & Optimization Notes

## Hardware

AMD Strix Halo (Ryzen AI Max 395), gfx1151, 128GB unified memory (125GB usable).
UMA architecture — GPU and CPU share the same physical RAM pool. The ~117GB "VRAM"
is a BIOS-configured slice of the same 125GB system RAM, not separate memory.

## Current Models (as of 2026-04-05)

| Model | Quant | Size on Disk | Vulkan Buffer | ctx-size | parallel | Port |
|-------|-------|-------------|--------------|----------|----------|------|
| Qwen3.5-122B-A10B | Q4_K_M (3 shards) | ~73GB | 72.2GB + 773MB host | 65536 | 2 | 3101 |
| Qwen3.5-9B | Q4_K_M | ~6GB | ~6GB | 8192 | 2 | 3102 |
| Qwen3-Embedding-8B | Q4_K_M | ~4.4GB | ~5GB | 8192 | 2 | 3103 |
| **Total** | | | **~101GB** | | | |

The mmproj-F16.gguf (867MB) is loaded on the 122B and 9B models for vision support.

## Current llama.cpp Version

Build **b8671** (2026-04-06), pre-built `ubuntu-vulkan-x64` binary, Vulkan RADV (Mesa 25.2.8).

Previous: version 1, commit `f20469d`, ROCm. Backup at `docker-compose.yml.rocm-backup`.

## Backend: Vulkan RADV

Switched from ROCm to Vulkan RADV on 2026-04-05. Reasons:
- **Memory leak**: ROCm VRAM usage crept up during long inference sessions (#19979),
  causing 117GB/125GB pressure with 2.1GB swap. Vulkan stays stable at ~101GB.
- **Performance**: Vulkan RADV is 15-33% faster than ROCm for MoE at our working
  context lengths (peak 22K tokens). ROCm only wins at 130K+ which we never hit.
- **Fewer bugs**: ROCm had UMA detection, compiler regression, slow loading past 64GB,
  KV cache shared memory issues — all absent with Vulkan.
- **Official guidance**: llama.cpp devs closed #20934 saying ROCm is "not expected to
  be faster than Vulkan" on RDNA (Wave64 available through Vulkan but not HIP).

Key Vulkan flags:
- `--ubatch-size 512` — >512 causes ~44% pp regression on Strix Halo (#18725)
- `--no-direct-io` — prevents model loading failure (#18741)
- `--flash-attn on` — Vulkan scalar FA landed May 2025 (#13324), refined 2026 (#19625)

## Memory Profile

With all 3 models loaded (Vulkan):
- **Used**: ~101GB / 125GB
- **Available**: ~23GB
- **Swap**: ~243MB
- **Headroom**: 45GB free device memory reported by Vulkan

Previous (ROCm): 117GB used, 7.3GB available, 2.1GB swap.

### Actual Token Usage (spin-cycle project)

From llama-server logs (122B model, slot size = 32,768 tokens):

| Pipeline Step | Typical Input Tokens | Peak Total (in+out) | % of Slot |
|---------------|---------------------|---------------------|-----------|
| Extraction chunks | 14K-22K | ~22K | 67% |
| Research agent (iterative) | 5K-19K (grows) | ~19K | 58% |
| Judge | 5K-12K | ~12K | 37% |
| Synthesize verdict | 5K-9K | ~9K | 27% |
| Classify (batched) | 3K-5K | ~8K | 24% |

**Peak observed: 22,143 tokens (67% of 32K slot).** Current 65536 ctx-size
(32K per slot with --parallel 2) provides ~1.5x headroom over worst case.

### Performance (Vulkan RADV, short context)

- Prompt processing: ~63 tokens/sec
- Token generation: ~23 tokens/sec

## Optimization Opportunities

### 1. TurboQuant (Future — Not Ready)

KV cache compression to 3-4 bits (Google Research, ICLR 2026). Would give 4-5x
KV cache memory reduction. Community forks exist but have known HIP NaN bugs.
Upstream merge expected ~Q3 2026.

- Discussion: https://github.com/ggml-org/llama.cpp/discussions/20969
- Feature request: https://github.com/ggml-org/llama.cpp/issues/20977

### 2. ik_llama.cpp

Advanced fork with SOTA quantization, better CPU perf, FlashMLA, fused MoE ops.
**Does NOT support AMD GPUs.** CPU and CUDA only. Not usable for our setup.

### 3. Known Vulkan Issues on Strix Halo

1. **Ubatch regression** ([#18725](https://github.com/ggml-org/llama.cpp/issues/18725)):
   ~44% pp loss when ubatch >512. Mitigated with `--ubatch-size 512`.

2. **Model loading with direct I/O** ([#18741](https://github.com/ggml-org/llama.cpp/issues/18741)):
   "Unexpectedly reached end of file." Mitigated with `--no-direct-io`.

3. **AMDVLK 2 GiB buffer limit** ([#15054](https://github.com/ggml-org/llama.cpp/issues/15054)):
   Only affects AMDVLK, not RADV. We use RADV (4 GiB limit).

## ROCm Notes (Historical)

ROCm backup files are preserved on joi for rollback:
- `docker-compose.yml.rocm-backup`
- `Dockerfile.rocm`
- `rocm-bin/` (b1198 binaries)

Known ROCm-specific issues (no longer relevant with Vulkan):
- UMA detection bug (#18159)
- ROCm 7.x compiler regression (#19984)
- Slow loading past 64GB (#15018)
- KV cache shared memory (#18011)
- VRAM memory leak during long sessions (#19979)
- `rocm-wmma-tune` branch for FA optimization (ROCm-only)

## Reference Links

- [Strix Halo Wiki - llama.cpp Performance](https://strixhalo.wiki/AI/llamacpp-performance)
- [Strix Halo Wiki - llama.cpp with ROCm](https://strixhalo.wiki/AI/llamacpp-with-ROCm)
- [kyuz0 Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)
- [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm)
- [llm-tracker.info — Strix Halo GPU Performance](https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance)
- [lhl/strix-halo-testing](https://github.com/lhl/strix-halo-testing)

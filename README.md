# CUDA Profiling Tutorial

This project contains educational ML training loop in Python that mirrors every best-practice
criterion from the CUDA C example, using CuPy as the GPU backend.

NVIDIA NSight profiling tools **nsys** and **ncu** are eused to detect and
diagnose four common GPU training anti-patterns, using a simple 3-layer MLP on
synthetic regression data:

- Model: 3-layer MLP  (input → hidden1 → hidden2 → output)
- Task:  Synthetic regression  (random X → random y)
- Loss:  Mean-squared error
- Optimization:   Vanilla SGD with mini-batches

The script `ml_training_cuda.py` implements the following criteria for optimizing the GPU performance.

| Criteria |  Goal  | Notes |
|--------|-------------|--------------|
|  C1 | Minimize host↔device transfers | dataset pinned & uploaded once at startup; only scalar loss is pulled back per epoch |
|  C2 | Coalesced memory access | all weight/activation matrices are                        row-major (CuPy default); matmul reads  contiguous rows → full cache lines |
|  C3 | Shared-memory prefetch | cuBLAS (used by cp.matmul) tiles GEMM internally; we expose this via streams |
|  C4 | Maximise SM occupancy | large batch size fills the GPU; we query and print achieved occupancy via nvml |
|  C5 | Avoid SMEM bank conflicts | weight dims chosen as multiples of 32+1 concept shown via padding commentary |
|  C6 | Overlap compute & transfer | CUDA streams: forward pass on stream0, async D→H loss scalar on stream1 |
|  C7 | Minimise warp divergence | ReLU implemented as cp.maximum (no branch); no data-dependent control flow in hot path |
|  C8 | Register pressure / no spill | operations fused with CuPy ElementwiseKernel to keep intermediates in registers |

Requirements
------------
    pip install cupy-cuda12x nvidia-ml-py tqdm

References:

- NVIDIA NSight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- NVIDIA NSight Compute User Guide: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

---

## A100 SXM4 — Key specs for profiling context

| Property | Value |
|----------|-------|
| SMs | 108 |
| CUDA cores (FP32) | 6912 |
| FP64 CUDA cores | 3456 (half of FP32) |
| Tensor Core units | 432 (4 per SM, 3rd gen) |
| Boost clock | ~1.41 GHz |
| HBM2e bandwidth | **2 TB/s** |
| PCIe 4.0 bandwidth | ~64 GB/s |

### Compute roofline ceilings

The compute roof is derived from first principles (units × FLOP/unit/clock × boost clock) and depends on the data precision.

For instance, A100 CUDA cores do 1 FP64 FMA per clock, but only half the CUDA cores are FP64-capable on A100 — 3456 FP64 cores (half of 6912):
```
FP64 peak = 3456 cores × 2 FLOP/FMA × 1.41 GHz
           = 3456 × 2 × 1.41 × 10⁹
           = 9.75 TFLOP/s  ≈ 9.7 TFLOP/s  
```
Since NVIDIA doesn't publish the exact tile dimensions for each precision publicly,
these Tensor Cores figures are reverse-engineered from the quoted TFLOP/s numbers
and confirmed by microbenchmarks (e.g. Jia et al., 2021). Suppose that Nvidia's 3rd-gen TC on A100
does a 8×4×4 tile for TF32 (not 4×4×4), then 
```
one TC per clock = 8×4×4 tile = 128 multiply-accumulates (FMA) = 256 FLOP
```

The peak values are then computed
```
TF32 peak = 432 TC × 256 FLOP/clock × 1.41 GHz
           = 155.8 TFLOP/s  ≈ 156 TFLOP/s
```
and
```
FP16 peak = 432 TC × 512 FLOP/clock × 1.41 GHz
           = 311.9 TFLOP/s  ≈ 312 TFLOP/s 
```

The following table summarizes the compute roof (the flat lines in the roofline charts):

| Precision | Active units | FLOP/unit/clock | Peak | Ridge point* |
|-----------|-------------|-----------------|------|-------------|
| FP64 scalar | 3456 cores | 2 | **9.7 TFLOP/s** | 4.9 FLOP/byte |
| FP64 Tensor Core | 432 TCs | 256 (8×4×4, half-rate) | **19.5 TFLOP/s** | 9.8 FLOP/byte |
| FP32 scalar | 6912 cores | 2 | **19.5 TFLOP/s** | 9.8 FLOP/byte |
| TF32 Tensor Core | 432 TCs | 256 (8×4×4 tile) | **156 TFLOP/s** | 78 FLOP/byte |
| FP16/BF16 Tensor Core | 432 TCs | 512 (8×4×8 tile) | **312 TFLOP/s** | 156 FLOP/byte |
| FP16 with 2:4 sparsity | 432 TCs | 1024 | **624 TFLOP/s** | 312 FLOP/byte |


### Ridge point: Compute-bound vs Memory-bound kernels.

The memory bandwidth roof is the linear relationship between the peak performance (TFLOP/s)
and arithmetic intensity (FLOP/byte), the slope is the theoretical memory bandwidth (GB/s)
of the GPU, i.e. 2000 GB/s for HBM2e.

The ridge point is the point at which the memory bandwidth boundary meets the peak performance boundary.
Below the ridge point the kernel is memory-bandwidth-bound; above it,
compute-bound. A typical large GEMM achieves ~100–200 FLOP/byte of arithmetic
intensity, placing it right around the TF32 ridge — which is why enabling vs
disabling TF32 Tensor Cores is so impactful on A100.

**FP64 TC vs TF32 TC tile geometry note:** both use an 8×4×4 output tile
(256 FLOP per TC per clock in principle), but the FP64 TC path runs at half
the issue rate because each FP64 multiply requires twice the transistor cycles
of TF32. This is why FP64 TC peaks at 19.5 TFLOP/s while TF32 TC peaks at
156 TFLOP/s despite the same nominal tile size.

### Minimum batch/dim sizing to fill all 108 SMs

cuBLAS decomposes GEMMs into 128×128 output tiles, one tile per SM wave:

```
tiles = ceil(M/128) × ceil(N/128)  ≥  108  (one full SM wave)

Recommended (≥ 2 waves):  tiles ≥ 216
  batch=2048, H1=1024  →  16 ×  8 =  128 tiles  (~1.2 waves, marginal on A100)
  batch=4096, H1=4096  →  32 × 32 = 1024 tiles  (~9.5 waves, good)
```

---

## Anti-Pattern Scripts

The following set of scripts demonstrates how to use **nsys** and **ncu** to detect and
diagnose four common GPU training anti-patterns, using a simple 3-layer MLP on
synthetic regression data.

| Script | Anti-pattern | Primary tool |
|--------|-------------|--------------|
| `ml_training_cuda.py` | **Baseline** (well-optimised reference) | both |
| `ml_training_v1_io_gaps.py` | Blocking I/O per batch → idle GPU gaps | **nsys** |
| `ml_training_v2_sm_underutil.py` | Batch/dims too small → SMs starved | **ncu** |
| `ml_training_v3_no_tensor_cores.py` | float64 + non-aligned dims → no Tensor Cores | **ncu** |
| `ml_training_v4_roofline_gap.py` | Repeated H↔D transfers → far below roofline | **nsys + ncu** |

---

## Quick-start: profile all variants

```bash
# Common nsys flags
NSYS_FLAGS="--trace=cuda,nvtx,cudnn,cublas \
            --cuda-memory-usage=true \
            --gpu-metrics-device=all \
            --force-overwrite=true \
            --delay=10"
            

| Flag | What it adds |
|------|-------------|
| `--trace=cuda,nvtx,cudnn,cublas` | CUDA kernels + NVTX ranges + cuDNN and cuBLAS API calls — maps kernels back to model operations |
| `--trace=...,osrt` | OS Runtime: `write`/`fsync`/`pthread` syscalls — add only for v1 to see filesystem stalls |
| `--cuda-memory-usage=true` | GPU memory allocation/free timeline — spots unexpected mid-loop allocations and peak memory spikes |
| `--gpu-metrics-device=all` | Continuous hardware-sampled metrics: SM active rate, memory bandwidth, NVLink, **Tensor Core Activity** — visible as line charts in the GPU Metrics rows in nsys-ui |
| `--force-overwrite=true` | Overwrites existing report files — convenient for iterative profiling |
| `--delay=10` | Collection start after 10 seconds delay to exclude the warming up stage |

# Baseline
nsys profile --trace=cuda,nvtx,osrt --stats=true -o baseline \
     python ml_training_cuda.py

# Anti-pattern 1 - I/O gaps
nsys profile $NSYS_FLAGS --trace=cuda,nvtx,cudnn,cublas,osrt -o v1_io \
     python ml_training_v1_io_gaps.py

# Anti-pattern 2 - SM underutilisation
nsys profile  $NSYS_FLAGS -o v2_sm python ml_training_v2_sm_underutil.py
ncu --set full --kernel-name regex:gemm --launch-count 100 -o v2_sm_ncu \
     python ml_training_v2_sm_underutil.py

# Anti-pattern 3 — No Tensor Cores
nsys profile $NSYS_FLAGS -o v3_notc python ml_training_v3_no_tensor_cores.py
ncu --set full --kernel-name regex:gemm --launch-count 100 -o v3_tc_ncu \
     python ml_training_v3_no_tensor_cores.py

# Anti-pattern 4 - Roofline gap
nsys profile  $NSYS_FLAGS -o v4_roof \
     python ml_training_v4_roofline_gap.py
ncu --set full --kernel-name regex:gemm --launch-count 100 -o v4_roof_ncu \
     python ml_training_v4_roofline_gap.py
```

**ncu runtime tip:** ncu replays each kernel multiple times to collect all
hardware counter sets, making a full training run 10–100× slower than normal.
CuPy also has significant initialisation overhead (memory pool setup, cuBLAS handle
creation, JIT compilation) that produces many launches before the first real GEMM
appears — so a small `--launch-skip` is not reliable.

The recommended approach is to filter by kernel name, which skips all CuPy
housekeeping kernels and targets only the cuBLAS GEMMs you care about:

```bash
# Capture all GEMM kernel types (forward + backward passes)
ncu --set full --kernel-name regex:gemm --launch-count 100 -o out python script.py
```

`regex:gemm` matches `ampere_sgemm_*`, `ampere_h16816gemm_*` (TF32 TC),
`cutlass_*gemm*`, etc. — everything cuBLAS dispatches for `cp.matmul()` calls.
If you also want to profile the elementwise kernels, add a second pass:

```bash
ncu --set full --kernel-name regex:cupy --launch-count 4 -o out_ew python script.py
```

---
## Baseline

Run the script on a GPU node
```
python ml_training_cuda.py
```

and profiling it with `nsys`
```
nsys profile $NSYS_FLAGS --stats=true -o ml_report python ml_training_cuda.py
```

Open the generated report `ml_report.nsys-rep` with `nsys-ui`
```
nsys-ui ml_report.nsys-rep
```

From the Timeline view, look at the **CUDA HW** row (or **Kernels** row).

To profile with `ncu`, you need permission for performance counter sampling on the GPU node:
```
ncu --set full --launch-skip 4 --launch-count 8 -o baseline_ncu \
    python ml_training_cuda.py
```

## Anti-pattern 1 - Blocking I/O (`v1_io_gaps.py`)

**What is broken:** `json.dump()` + `os.fsync()` called inside every batch loop.
`loss_val.get()` synchronises the stream first; then the disk write stalls the
CPU; the GPU sits completely idle during each write.

**nsys - what to look for:**
1. Open `v1_io.nsys-rep` with `nsys-ui v1_io.nsys-rep` → Timeline view.
2. Zoom into the **CUDA HW** row (or **Kernels** row).
3. You will see a repeating pattern:
   ```
   [orange kernel burst]  ──── large grey void ────  [orange kernel burst]
   ```
4. The NVTX row shows red **"BLOCKING CKPT N"** ranges aligned with each void.
5. In **"CUDA API Statistics"**: `cudaStreamSynchronize` has very high total time.
6. In the **OS Runtime** row (visible because `osrt` is in the trace): `write`
   and `fsync` syscalls align exactly with each grey void.
7. In the **GPU Metrics** rows: the **SM Active** and **Tensor Core Activity**
   charts will drop to 0% during each void — confirming the GPU is completely
   idle while the CPU is doing I/O.

**Baseline comparison:** In `baseline.nsys-rep` the kernel bars are essentially
continuous with gaps < 50 µs between batches, and the SM Active / Tensor Core
Activity charts remain consistently high throughout.

**The fix:**
```python
# BAD — inside the loop:
loss_cpu = float(loss_val.get())   # stream sync
write_checkpoint(loss_cpu)         # disk stall

# GOOD — accumulate on GPU, .get() once per epoch:
epoch_loss_gpu += loss_val         # stays on device, no sync
# ... after the loop:
loss_np = float((epoch_loss_gpu / n_batches).get()[0])
```

---

## Anti-pattern 2 — SM Underutilisation (`v2_sm_underutil.py`)

**What is broken:** `batch_size=32`, `H1=64`, `H2=32`.  The layer-1 forward GEMM
`(32, 512) @ (512, 64)` generates only **1 tile** — 1 of the SMs works, the rest sit idle.

**Rule of thumb (A100, 108 SMs, 128×128 tile):**
```
tiles = ceil(M/128) × ceil(N/128)
      = ceil(batch/128) × ceil(H1/128)

This script:  ceil(32/128)  × ceil(64/128)   =  1 ×  1 =    1 tile  →  1 SM active
Marginal:     ceil(2048/128) × ceil(1024/128) = 16 ×  8 =  128 tiles → ~1.2 SM waves
Good:         ceil(4096/128) × ceil(4096/128) = 32 × 32 = 1024 tiles → ~9.5 SM waves
```

**ncu — what to look for:**
```bash
ncu --metrics \
  launch__grid_size,\
  launch__block_size,\
  launch__waves_per_multiprocessor,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \
  python ml_training_v2_sm_underutil.py
```
- `launch__grid_size` → 1–4 blocks (should be hundreds to thousands)
- `launch__waves_per_multiprocessor` → < 1 (less than one full SM wave)
- `sm__warps_active` → < 5% of peak
- **Roofline chart** in ncu-ui: operating point is below *both* roofs — not
  bandwidth-limited, not compute-limited; simply not enough parallelism to
  reach either ceiling.

**nsys GPU Metrics — what to look for:**
  In the **GPU Metrics** rows in nsys-ui, compare v2 vs baseline side by side:
  - **SM Active** chart: sustained low value in v2 vs high in baseline
  - **Tensor Core Activity** chart: sporadic low spikes in v2 vs sustained
    high activity in baseline — each spike corresponds to one tiny GEMM
    finishing almost instantly before the GPU goes idle again waiting for
    the next kernel launch

**The fix:**
```python
# BAD:  batch=32,   H1=64   →    1 tile  (1 SM active out of 108)
# OK:   batch=2048, H1=1024 →  128 tiles (~1.2 SM waves, marginal on A100)
# GOOD: batch=4096, H1=4096 → 1024 tiles (~9.5 SM waves)
```

---

## Anti-pattern 3 — No Tensor Cores (`v3_no_tensor_cores.py`)

**What is broken:**
- **(a) float64 weights** → cuBLAS selects a DGEMM path using FP64 Tensor Cores
  at 19.5 TFLOP/s — versus TF32 TC at 156 TFLOP/s or FP16 TC at 312 TFLOP/s.
  That is an 8–16× throughput penalty.
- **(b) Hidden dims not multiples of 8** (H1=100, H2=60) → even with fp32,
  cuBLAS cannot fill Tensor Core warp tiles cleanly and may fall back to
  scalar FP32 CUDA cores (19.5 TFLOP/s) instead of TF32 TC (156 TFLOP/s).

**A100 Tensor Core precision hierarchy:**

| dtype input | TC path | Peak |
|-------------|---------|------|
| float64 | FP64 TC (3rd gen) | 19.5 TFLOP/s |
| float32, TF32 disabled | FP32 scalar | 19.5 TFLOP/s |
| float32, TF32 enabled (default) | TF32 TC | 156 TFLOP/s |
| float16 / bfloat16 | FP16/BF16 TC | 312 TFLOP/s |

TF32 is **enabled by default** in cuBLAS 11+ for fp32 inputs — you get it for
free as long as dims are multiples of 4 (multiples of 64 for best efficiency).

**GP100 note:** Pascal does not have Tensor Cores at all.  The comparison here
is DGEMM (~5 TFLOP/s) vs SGEMM (~21 TFLOP/s).  On Volta/Ampere the gap is
much larger because fp16 Tensor Cores deliver 125–312 TFLOP/s.

**ncu — what to look for:**
```bash
# Per-precision Tensor Core activity (A100 3rd-gen specific metrics)
ncu --metrics \
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active \
  python ml_training_v3_no_tensor_cores.py

# Instruction mix — dfma should dominate; ffma/hfma should be 0
ncu --metrics \
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum \
  python ml_training_v3_no_tensor_cores.py
```
- `tensor_op_hmma` (fp16 TC) → 0%
- `tensor_op_dmma` (fp64 TC) → active but at low throughput (19.5 TFLOP/s)
- `op_dfma` instruction count → high;  `op_ffma` / `op_hfma` → 0

**Roofline in ncu-ui:**
- The relevant ceiling is the **FP64 TC roof at 19.5 TFLOP/s**.
- The TF32 roof (156) and FP16 roof (312) will be visible but unreachable —
  the chart makes the 8–16× headroom immediately apparent.

**nsys GPU Metrics — what to look for:**
  - **Tensor Core Activity** chart: present but low — FP64 TC is active but
    running at 19.5 TFLOP/s vs 156 TFLOP/s for TF32. Compare directly with
    the baseline report: the chart height will be visibly lower in v3.
  - This is the continuous-time complement to ncu's per-kernel
    `sm__pipe_tensor_op_dmma_cycles_active` metric.

**The fix:**
```python
# BAD — hits FP64 TC roof (19.5 TFLOP/s), misaligned dims:
self.W = cp.array(..., dtype=cp.float64)
H1, H2 = 100, 60    # not multiples of 8

# GOOD — TF32 TC (156 TFLOP/s), automatic with fp32 + aligned dims:
self.W = cp.array(..., dtype=cp.float32)
H1, H2 = 1024, 512  # multiples of 64 → full TC tile efficiency

# BETTER — FP16 TC (312 TFLOP/s), explicit cast:
self.W = cp.array(..., dtype=cp.float16)
H1, H2 = 1024, 512  # multiples of 8 required for fp16 TC
```

---

## Anti-pattern 4 — Roofline Gap (`v4_roofline_gap.py`)

**What is broken (three compounding issues):**

| # | Issue | Extra bytes | Extra FLOPs |
|---|-------|-------------|-------------|
| a | Full dataset re-uploaded every epoch via PCIe 4.0 | +dataset size/epoch | 0 |
| b | Shuffle index built on CPU, transferred to GPU | +N×4 bytes/epoch | 0 |
| c | Out-of-place weight updates allocate extra tensors | +2× weight size/step | 0 |

All three add memory traffic without adding useful computation → arithmetic
intensity (AI = FLOPs/byte) drops → operating point moves **left** on the
roofline. PCIe 4.0 at ~64 GB/s is also **~31× slower** than HBM2e at 2 TB/s,
so measured bandwidth drops too → point moves **down** as well.
Result: far from both roofs.

**nsys — what to look for:**
1. Open `v4_roof.nsys-rep` → Timeline.
2. Look at the **MemOps** / **DMA** row at the start of each epoch:
   large teal H→D bars will dwarf the compute kernel bars that follow.
3. In **"CUDA API Statistics"**: `cudaMemcpy` total time >> kernel total time.
4. The ratio of DMA time to kernel time directly quantifies the wasted epoch
   time on PCIe transfers.
5. In the **GPU Metrics** rows:
   - **SM Active** and **Tensor Core Activity**: drop to 0% during each H→D
     transfer — the GPU is completely idle while data crosses PCIe.
   - **Memory Bandwidth** (if shown): during the compute phase, HBM2e
     utilisation will be lower than the baseline because data arriving from
     PCIe bypasses the L2 cache, reducing effective reuse.

**ncu — what to look for:**
```bash
ncu --metrics \
  dram__bytes.sum,\
  dram__bytes_read.sum,\
  sm__cycles_elapsed.avg,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
  python ml_training_v4_roofline_gap.py
```
- **Roofline chart**: operating point far left (low AI) AND below the HBM2e
  bandwidth slope — PCIe transfers reduce effective HBM utilisation because
  data bypasses the L2 cache on arrival.
- `l1tex` global load bandwidth: lower than HBM2e peak — cache hierarchy is
  cold every epoch since data originates from the host.

**The fix:**
```python
# BAD — inside epoch loop:
X_dev = cp.array(X_host)            # 64 GB/s PCIe, every epoch
perm_cpu = np.random.permutation(n)
X_shuf = X_dev[cp.array(perm_cpu)]  # index array crosses PCIe too
self.W = self.W - lr32 * self.dW    # out-of-place: 2× allocations

# GOOD — upload once before loop:
X_dev = cp.array(X_host)            # PCIe once only
# inside loop:
perm   = cp.random.permutation(n)   # stays on HBM2e (2 TB/s)
X_shuf = X_dev[perm]
self.W -= lr32 * self.dW            # in-place: no extra allocation
```

---

## Side-by-side metric summary

| Script | GPU idle gaps | SM util | Tensor Cores | Roofline position |
|--------|--------------|---------|--------------|-------------------|
| baseline | minimal | high | TF32 TC active | near TF32 compute roof (156 TFLOP/s) |
| v1_io_gaps | **large (per batch)** | low | TF32 TC active | moderate |
| v2_sm_underutil | moderate | **< 5%** | TF32 TC active | below both roofs (parallelism-starved) |
| v3_no_tensor_cores | minimal | moderate | **FP64 TC only** | near FP64 roof (19.5 TFLOP/s, 8× below TF32) |
| v4_roofline_gap | minimal (compute) | moderate | TF32 TC active | **far below both** (PCIe bottleneck) |

---

## Useful ncu metric cheat sheet (A100 / ncu 12.2+)

```bash
# SM and warp occupancy
ncu --metrics \
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  sm__achieved_occupancy_pct

# Tensor Core utilisation — per precision (A100 3rd-gen TC)
ncu --metrics \
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_active

# Instruction mix (fp32 scalar vs fp64 scalar vs fp16 TC)
ncu --metrics \
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_hfma_pred_on.sum

# HBM2e bandwidth
ncu --metrics \
  dram__bytes_read.sum.per_second,\
  dram__bytes_write.sum.per_second

# L2 cache hit rate
ncu --metrics \
  lts__t_sector_hit_rate.pct

# Grid / launch dims and SM waves
ncu --metrics \
  launch__grid_size,\
  launch__block_size,\
  launch__waves_per_multiprocessor

# Arithmetic intensity (compute yourself from these two)
ncu --metrics \
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
  dram__bytes.sum
# AI = (ffma_count × 2) / dram_bytes
```

### ncu baseline comparison workflow

```bash
# 1. Profile the baseline
ncu --set full --kernel-name regex:gemm --launch-count 100 -o baseline_ncu \
    python ml_training_cuda.py

# 2. Profile an anti-pattern variant
ncu --set full --kernel-name regex:gemm --launch-count 100 -o v3_ncu \
    python ml_training_v3_no_tensor_cores.py

# 3. Open both in ncu-ui
#    Right-click baseline_ncu → "Set as Baseline"
#    All metrics in v3_ncu now show % delta vs baseline — immediately
#    visible how much throughput the anti-pattern costs.
```

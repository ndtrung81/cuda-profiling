# CUDA Profiling Tutorial

This project contains educational ML training loop in Python that mirrors every best-practice
criterion from the CUDA C example, using CuPy as the GPU backend.

The workload is a simple deep learning network training:

- Model: 3-layer MLP  (input → hidden1 → hidden2 → output)
- Task:  Synthetic regression  (random X → random y)
- Loss:  Mean-squared error
- Optimization:   Vanilla SGD with mini-batches

The script `ml_training_cuda.py` implements the following criteria for optimizing the GPU performance.

| Criteria |  Goal  | Notes |
|--------|-------------|--------------|
|  C1 | Minimise host↔device transfers | dataset pinned & uploaded once at startup; only scalar loss is pulled back per epoch |
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
# Baseline
nsys profile --trace=cuda,nvtx,osrt --stats=true -o baseline \
     python ml_training_cuda.py

# Anti-pattern 1 - I/O gaps
nsys profile --trace=cuda,nvtx,osrt --stats=true -o v1_io \
     python ml_training_v1_io_gaps.py

# Anti-pattern 2 - SM underutilisation
nsys profile --trace=cuda,nvtx,osrt --stats=true -o v2_sm \
     python ml_training_v2_sm_underutil.py
ncu --set full --launch-count 4 -o v2_sm_ncu \
     python ml_training_v2_sm_underutil.py

# Anti-pattern 3 - No Tensor Cores
ncu --set full --launch-count 4 -o v3_tc_ncu \
     python ml_training_v3_no_tensor_cores.py

# Anti-pattern 4 - Roofline gap
nsys profile --trace=cuda,nvtx,osrt --stats=true -o v4_roof \
     python ml_training_v4_roofline_gap.py
ncu --set full --launch-count 4 -o v4_roof_ncu \
     python ml_training_v4_roofline_gap.py
```

---

## Anti-pattern 1 - Blocking I/O (`v1_io_gaps.py`)

**What is broken:** `json.dump()` + `os.fsync()` called inside every batch loop.
`loss_val.get()` synchronises the stream first; then the disk write stalls the CPU;
the GPU sits idle the whole time.

**nsys - what to look for:**
1. Open `v1_io.nsys-rep` with `nsys-ui v1_io.nsys-rep` → Timeline view.
2. Zoom into the **CUDA HW** row (or **Kernels** row).
3. You will see a repeating pattern:
   ```
   [orange kernel burst]  ──── large grey void ────  [orange kernel burst]
   ```
4. The NVTX row shows red **"BLOCKING CKPT N"** ranges aligned with each void.
5. In **"CUDA API Statistics"**: `cudaStreamSynchronize` has very high total time.
6. In the **OS Runtime** row: `write` + `fsync` syscalls match each void.

**Baseline comparison:** In `baseline.nsys-rep` the kernel bars are essentially
continuous with gaps < 50 µs between batches.

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

**Rule of thumb (128×128 tile):**
```
tiles = ceil(M/128) × ceil(N/128)
      = ceil(batch/128) × ceil(H1/128)

This script:  ceil(32/128) × ceil(64/128)  =  1 × 1  =  1 tile  → 1 SM
Baseline:     ceil(2048/128) × ceil(1024/128) = 16 × 8 = 128 tiles → ~1 SM waves for A100 (108 SMs) or 2 SM waves for GP100 (56 SMs)
```

**ncu — what to look for:**
```bash
ncu --metrics \
  launch__grid_size,\
  launch__block_size,\
  sm__warps_active.avg.pct_of_peak_sustained_active,\
  sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \
  python ml_training_v2_sm_underutil.py
```
- `launch__grid_size` → 1–4 blocks (should be hundreds)
- `sm__warps_active` → < 5 % of peak
- **Roofline chart** in `ncu-ui`: operating point is below *both* roofs — not
  bandwidth-limited, not compute-limited; simply not enough parallelism.

**The fix:** ensure `batch × H1 ≥ n_SMs × tile²`.  For GP100:
`batch=2048, H1=1024` → 128 tiles; `batch=16384, H1=4096` → 4096 tiles.

---

## Anti-pattern 3 — No Tensor Cores (`v3_no_tensor_cores.py`)

**What is broken:**
- (a) Weights are `float64` → cuBLAS selects a DGEMM kernel; Tensor Cores
  require fp16/bf16/tf32, not fp64.
- (b) Hidden dims are 100 and 60 (not multiples of 8) → even with fp32/fp16,
  cuBLAS cannot fill Tensor Core warp tiles cleanly.

**GP100 note:** Pascal does not have Tensor Cores at all.  The comparison here
is DGEMM (~5 TFLOP/s) vs SGEMM (~21 TFLOP/s).  On Volta/Ampere the gap is
much larger because fp16 Tensor Cores deliver 125–312 TFLOP/s.

**ncu — what to look for:**
```bash
ncu --metrics \
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\
  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
  smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
  python ml_training_v3_no_tensor_cores.py
```
- `sm__pipe_tensor_cycles_active` → 0 % (no Tensor Core activity)
- `op_dfma` count → very high (double-precision FMAs)
- `op_ffma` count → 0 (no single-precision FMAs)

**Roofline in ncu-ui:**
- The FP64 roof (~5 TFLOP/s) is the relevant ceiling, not the FP32 roof.
- Your operating point will be near or below the FP64 roof — the chart makes
  visible how much headroom the FP32 and Tensor Core roofs have.

**The fix:**
```python
# BAD:
self.W = cp.array(..., dtype=cp.float64)   # DGEMM, no Tensor Cores

# GOOD (fp32, SGEMM):
self.W = cp.array(..., dtype=cp.float32)

# GOOD (fp16 Tensor Cores, Volta+):
self.W = cp.array(..., dtype=cp.float16)

# GOOD (dims — always multiples of 8, preferably 64):
H1 = 1024   # not 100
H2 = 512    # not 60
```

---

## Anti-pattern 4 — Roofline Gap (`v4_roofline_gap.py`)

**What is broken (three compounding issues):**

| # | Issue | Extra bytes | Extra FLOPs |
|---|-------|-------------|-------------|
| a | Full dataset re-uploaded every epoch via PCIe | +250 MB/epoch | 0 |
| b | Shuffle index array goes CPU→GPU every epoch | +256 KB/epoch | 0 |
| c | Out-of-place weight updates allocate extra tensors | +2× weight traffic | 0 |

All three add memory traffic without adding useful computation → arithmetic
intensity (AI = FLOPs/byte) drops → operating point moves LEFT on roofline.
PCIe is also ~45× slower than HBM2, so measured bandwidth also drops → point
moves DOWN as well.  Result: far from both roofs.

**nsys — what to look for:**
1. Open `v4_roof.nsys-rep` → Timeline.
2. Look at the **MemOps** / **DMA** row at the start of each epoch:
   large teal bars (H→D) will dwarf the subsequent compute kernel bars.
3. Compare compute time vs transfer time in **"CUDA API Statistics"**:
   `cudaMemcpy` total time >> kernel total time.

**ncu — what to look for:**
```bash
ncu --metrics \
  dram__bytes_read.sum,\
  dram__bytes_write.sum,\
  sm__cycles_elapsed.avg,\
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second \
  python ml_training_v4_roofline_gap.py
```
- **Roofline chart**: operating point far left (low AI) AND below the memory
  bandwidth slope (PCIe bottleneck reduces effective HBM utilisation).
- `l1tex` global load bandwidth will be low — data is not resident in L2
  when it arrives from PCIe, bypassing the cache hierarchy.

**The fix:**
```python
# BAD — inside epoch loop:
X_dev = cp.array(X_host)          # PCIe every epoch
perm_cpu = np.random.permutation(n)
X_shuf = X_dev[cp.array(perm_cpu)]  # perm crosses PCIe too
self.W = self.W - lr * self.dW    # out-of-place: 2× allocations

# GOOD — upload once before loop:
X_dev = cp.array(X_host)          # PCIe once
# inside loop:
perm = cp.random.permutation(n)   # stays on GPU
X_shuf = X_dev[perm]
self.W -= lr * self.dW            # in-place: no extra allocation
```

---

## Side-by-side metric summary

| Script | GPU idle gaps | SM util | Tensor Cores | Roofline AI |
|--------|--------------|---------|--------------|-------------|
| baseline | minimal | high | active (fp32) | near compute roof |
| v1_io_gaps | **large (per batch)** | low | active | moderate |
| v2_sm_underutil | moderate | **< 5 %** | active | below both roofs |
| v3_no_tensor_cores | minimal | moderate | **0 %** | near FP64 roof |
| v4_roofline_gap | minimal (compute) | moderate | active | **far below both** |

---

## Useful ncu metric cheat sheet

```bash
# SM and warp occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
achieved_occupancy

# Tensor Core utilisation
ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active

# Instruction mix (fp32 vs fp64 vs fp16)
ncu --metrics smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hfma_pred_on.sum

# Memory bandwidth
ncu --metrics dram__bytes_read.sum.per_second,dram__bytes_write.sum.per_second

# Grid / launch dims
ncu --metrics launch__grid_size,launch__block_size,launch__waves_per_multiprocessor
```

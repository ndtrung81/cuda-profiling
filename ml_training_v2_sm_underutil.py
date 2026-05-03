"""
ml_training_v2_sm_underutil.py  —  ANTI-PATTERN 2: Batch/layer sizes too small → SMs starved
==============================================================================================
WHAT IS BROKEN
--------------
Both the batch size (32) and hidden dimensions (64 × 32) are far too small to
generate enough GEMM tiles to keep the GPU's SMs busy.

HOW GEMM TILE OCCUPANCY WORKS
------------------------------
cuBLAS decomposes (M, K) @ (K, N) into tiles of roughly 128×128 elements.
The total number of output tiles = ceil(M/128) × ceil(N/128).
Each tile maps to one SM.  If tiles < number_of_SMs, most SMs sit idle.

  Layer 1 forward with this script:
    (32, 512) @ (512, 64)  →  ceil(32/128) × ceil(64/128)  =  1 × 1  =  1 tile
    → only 1 of 56 SMs does useful work per kernel launch.

  Layer 1 forward with the baseline (batch=2048, H1=1024):
    (2048, 512) @ (512, 1024) → ceil(2048/128) × ceil(1024/128) = 16×8 = 128 tiles
    → ~2 waves across all 56 SMs (still not ideal, but far better).

WHAT YOU WILL SEE IN nsys
--------------------------
  Timeline:
    • Very short kernel bars — each GEMM finishes in microseconds because
      only 1–2 SMs are working.
    • High ratio of launch overhead to kernel execution time.
    • Many more kernel launches per unit time (small work → fast finish → next).

  ncu (Nsight Compute) — this is the primary tool for this anti-pattern:
    Section "GPU Speed Of Light Throughput":
      • Compute (SM) Throughput:  very low (< 5% of peak)
      • Memory Throughput:        very low
      → Neither roof is being approached; the work is just too small.

    Section "Launch Statistics":
      • Grid Size:   1–4 thread blocks  (should be hundreds to thousands)
      • Block Size:  256  (fine, but irrelevant when grid is tiny)

    Section "Occupancy":
      • Achieved Occupancy:  low — only a handful of warps are resident
        across the entire GPU.

    Roofline chart:
      • The operating point will be far below BOTH the memory bandwidth
        roof and the compute (Tensor Core / FP32) roof.  This is not a
        bandwidth bottleneck or a compute bottleneck — the kernel simply
        doesn't have enough parallelism to reach either roof.

HOW TO PROFILE
--------------
  # Step 1 – nsys for timeline overview
  nsys profile --trace=cuda,nvtx,osrt --stats=true \\
               -o v2_sm python ml_training_v2_sm_underutil.py
  nsys-ui v2_sm.nsys-rep

  # Step 2 – ncu for per-kernel occupancy and roofline
  ncu --set full --launch-count 4 -o v2_sm_ncu python ml_training_v2_sm_underutil.py
  ncu-ui v2_sm_ncu.ncu-rep

  # Step 3 – compare grid sizes between this script and the baseline
  ncu --metrics launch__grid_size,launch__block_size,sm__warps_active.avg.pct_of_peak_sustained_active \\
      python ml_training_v2_sm_underutil.py

THE FIX
-------
  Rule of thumb for GP100 (56 SMs):
    batch × hidden ≥ 56 × 128 × 2  ≈  14 336 elements per GEMM dimension
  so that cuBLAS generates at least one full wave of tiles across all SMs.

  With batch=2048, H1=1024:
    ceil(2048/128) × ceil(1024/128) = 16 × 8 = 128 tiles → ~2 full SM waves.
  With batch=16384, H1=4096 (larger baseline):
    ceil(16384/128) × ceil(4096/128) = 128 × 32 = 4096 tiles → ~73 SM waves.
"""

import time, contextlib
import cupy as cp
import numpy as np

try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    class _NvtxStub:
        @staticmethod
        def push_range(msg, color=None): pass
        @staticmethod
        def pop_range(): pass
    nvtx = _NvtxStub()
    HAS_NVTX = False

@contextlib.contextmanager
def nvtx_range(label, color="green"):
    nvtx.push_range(label, color=color)
    try:
        yield
    finally:
        nvtx.pop_range()

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False

_relu_backward_fused = cp.ElementwiseKernel(
    'float32 dout, float32 x', 'float32 din',
    'din = (x > 0) ? dout : 0', 'relu_bwd_v2')

class LinearLayer:
    def __init__(self, fan_in, fan_out, stream):
        self.stream = stream
        std = cp.sqrt(cp.array(2.0 / fan_in, dtype=cp.float32))
        with stream:
            self.W = cp.random.randn(fan_in, fan_out).astype(cp.float32) * std
            self.b = cp.zeros(fan_out, dtype=cp.float32)
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        with self.stream:
            return cp.matmul(X, self.W) + self.b

    def backward(self, dout, lr):
        with self.stream:
            batch = self.X_cache.shape[0]
            self.dW = cp.matmul(self.X_cache.T, dout) / batch
            self.db = dout.mean(axis=0)
            dX = cp.matmul(dout, self.W.T)
            self.W -= lr * self.dW
            self.b  -= lr * self.db
        return dX

class MLP:
    def __init__(self, D, H1, H2, stream):
        self.l1 = LinearLayer(D,  H1, stream)
        self.l2 = LinearLayer(H1, H2, stream)
        self.l3 = LinearLayer(H2, 1,  stream)
        self.stream = stream
        self.z1 = self.z2 = None

    def forward(self, X):
        with self.stream:
            self.z1 = self.l1.forward(X)
            a1 = cp.maximum(self.z1, 0.0)
            self.z2 = self.l2.forward(a1)
            a2 = cp.maximum(self.z2, 0.0)
            return self.l3.forward(a2)

    def backward(self, dout, lr):
        with self.stream:
            d2 = self.l3.backward(dout, lr)
            d2 = _relu_backward_fused(d2, self.z2)
            d1 = self.l2.backward(d2, lr)
            d1 = _relu_backward_fused(d1, self.z1)
            self.l1.backward(d1, lr)

def mse_loss(pred, target):
    diff = pred - target
    return (diff * diff).mean(), (2.0 * diff / pred.size)

def _tile_count(M, N, tile=128):
    """How many 128×128 output tiles does cuBLAS need for this GEMM shape?"""
    import math
    return math.ceil(M / tile) * math.ceil(N / tile)

# ─────────────────────────────────────────────────────────────────────────────
def train(
    n_samples : int = 65536,
    n_features: int = 512,
    # ── ANTI-PATTERN: tiny batch + narrow hidden dims ─────────────────────────
    H1        : int = 64,    # baseline uses 1024 — 16× wider
    H2        : int = 32,    # baseline uses  512 — 16× wider
    batch_size: int = 32,    # baseline uses 2048 — 64× larger
    # ─────────────────────────────────────────────────────────────────────────
    epochs    : int = 5,
    lr        : float = 1e-3,
):
    tc_l1_fwd = _tile_count(batch_size, H1)
    tc_l1_bwd = _tile_count(n_features, H1)

    print("=" * 60)
    print(" ANTI-PATTERN 2: Batch/Layer Sizes Too Small → SMs Starved")
    print("=" * 60)
    print(f"  Device    : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Batch     : {batch_size}  H1={H1}  H2={H2}")
    print(f"  L1 fwd GEMM ({batch_size},{n_features})@({n_features},{H1})")
    print(f"    → ~{tc_l1_fwd} output tile(s)  (baseline: ~128 tiles)")
    print(f"    → only {min(tc_l1_fwd, 56)}/56 SMs can work per GEMM launch")
    print(f"  >>> Expect very low SM utilisation and tiny grid sizes in ncu.")
    print()

    X_host_mem = cp.cuda.alloc_pinned_memory(n_samples * n_features * 4)
    y_host_mem = cp.cuda.alloc_pinned_memory(n_samples * 4)
    X_host = np.frombuffer(X_host_mem, dtype=np.float32).reshape(n_samples, n_features)
    y_host = np.frombuffer(y_host_mem, dtype=np.float32).reshape(n_samples, 1)
    rng = np.random.default_rng(42)
    X_host[:] = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y_host[:] = rng.standard_normal((n_samples, 1)).astype(np.float32)

    upload_stream = cp.cuda.Stream(non_blocking=True)
    with upload_stream:
        X_dev = cp.array(X_host)
        y_dev = cp.array(y_host)
    upload_stream.synchronize()

    compute_stream = cp.cuda.Stream(non_blocking=True)
    copy_stream    = cp.cuda.Stream(non_blocking=True)
    loss_ready_event = cp.cuda.Event(disable_timing=True)

    model = MLP(n_features, H1, H2, compute_stream)
    n_batches = n_samples // batch_size

    print(f"  {'Epoch':>5}  {'Loss':>12}  {'ms/epoch':>10}  {'SM util%':>9}")
    print("  " + "-" * 44)

    for epoch in range(epochs):
        with nvtx_range(f"Epoch {epoch}", color="yellow"):
            t0 = time.perf_counter()
            epoch_loss_gpu = cp.zeros(1, dtype=cp.float32)

            with compute_stream:
                perm   = cp.random.permutation(n_samples)
                X_shuf = X_dev[perm]
                y_shuf = y_dev[perm]

            for b in range(n_batches):
                s = b * batch_size
                with compute_stream:
                    Xb   = X_shuf[s:s + batch_size]
                    yb   = y_shuf[s:s + batch_size]
                    pred = model.forward(Xb)
                    loss_val, dloss = mse_loss(pred, yb)
                    epoch_loss_gpu += loss_val
                    model.backward(dloss, lr)

            loss_ready_event.record(compute_stream)
            copy_stream.wait_event(loss_ready_event)
            with copy_stream:
                epoch_loss_gpu /= n_batches
            copy_stream.synchronize()
            loss_np = float(epoch_loss_gpu.get()[0])

            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000

            sm_util = "n/a"
            if HAS_NVML:
                util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
                sm_util = f"{util.gpu:>8d}%"
            print(f"  {epoch:>5}  {loss_np:>12.6f}  {ms:>10.1f}  {sm_util:>9}")

    print()
    print("  ncu diagnosis: launch__grid_size will be 1–4; SM throughput < 5%.")
    print("  Fix: increase batch_size and hidden dims so tile count >> n_SMs.")
    print("=" * 60)

if __name__ == "__main__":
    train()

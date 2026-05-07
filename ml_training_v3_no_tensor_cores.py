"""
ml_training_v3_no_tensor_cores.py  —  ANTI-PATTERN 3: dtype/dims that bypass Tensor Cores
===========================================================================================
WHAT IS BROKEN
--------------
Two independent mistakes combine to prevent optimal Tensor Core utilisation:

  (a) Weights and activations are stored as float64 (double precision).
      On A100, cuBLAS dispatches a DGEMM kernel using the FP64 Tensor Cores
      at 19.5 TFLOP/s — versus TF32 Tensor Cores at 156 TFLOP/s (automatic
      for fp32 inputs) or FP16 Tensor Cores at 312 TFLOP/s.  That is an
      8–16× throughput penalty.

  (b) Hidden dimensions are not multiples of 8 (H1=100, H2=60).
      TF32/FP16 Tensor Cores require M, N, K to be multiples of 4 (tf32)
      or 8 (fp16/bf16) for full tile efficiency.  Non-aligned sizes force
      cuBLAS to handle remainder elements with scalar code or fall back to
      non-TC GEMM paths entirely.

  The combined effect: every GEMM uses the FP64 TC path at 19.5 TFLOP/s
  instead of TF32 TC at 156 TFLOP/s — leaving 87% of the A100 compute
  throughput unused.

WHAT YOU WILL SEE IN ncu
-------------------------
  Section "GPU Speed Of Light Throughput":
    • SM Compute Throughput will be low even for large batches — you are
      using ~10–25% of FP32 peak because DGEMM is slower per flop.

  Section "Tensor Core":
    • tensor_core_utilization (or pipe__tensor_cycles_active metric):
      → 0 %  — Tensor Cores are completely dark.

  Section "Instruction Statistics":
    • DFMA (double-precision fused multiply-add) instructions will dominate
      instead of HFMA (fp16) or FFMA (fp32).

  Roofline chart:
    • The relevant ceiling is the FP64 TC roof at 19.5 TFLOP/s — roughly
      8× lower than the TF32 TC roof (156 TFLOP/s) on A100.  Your operating point will be near the FP64 roof (if arithmetic-
      bound) or below it (if memory-bound), but the FP32 and Tensor Core
      roofs will be unreachable.

HOW TO PROFILE
--------------
  # ncu is the primary tool here — Tensor Core metrics are per-kernel
  ncu --set full --launch-count 4 -o v3_tc_ncu python ml_training_v3_no_tensor_cores.py
  ncu-ui v3_tc_ncu.ncu-rep

  # Key metrics to query directly:
  ncu --metrics sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,\\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active,\\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum \\
      python ml_training_v3_no_tensor_cores.py

  # nsys is also useful to compare kernel duration vs baseline
  nsys profile --trace=cuda,nvtx,osrt --stats=true \\
               -o v3_notc python ml_training_v3_no_tensor_cores.py

THE FIX (for A100)
------------------
  (a) Use float32 — TF32 Tensor Cores are enabled by default in cuBLAS 11+
      for fp32 inputs, so no code change is needed beyond using the right dtype:
        float64 → FP64 TC → 19.5 TFLOP/s   (this script)
        float32 → TF32 TC → 156 TFLOP/s    (8× faster, automatic)
        float16 → FP16 TC → 312 TFLOP/s    (16× faster, explicit cast)

  (b) Align hidden dimensions to multiples of 8 for fp16/tf32, ideally 64:
        H1=100, H2=60  → non-aligned, partial TC tile use  (this script)
        H1=1024, H2=512 → multiples of 64, full TC tile efficiency
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

# ── ReLU backward: must handle float64 ───────────────────────────────────────
# ElementwiseKernel is templated; 'T' picks up the actual dtype at runtime.
_relu_backward_fused = cp.ElementwiseKernel(
    'T dout, T x', 'T din',
    'din = (x > (T)0) ? dout : (T)0',
    'relu_bwd_v3')

class LinearLayer:
    def __init__(self, fan_in, fan_out, stream, dtype=cp.float64):
        self.stream = stream
        self.dtype  = dtype
        std = np.sqrt(2.0 / fan_in)
        with stream:
            # ── ANTI-PATTERN (a): weights stored in float64 ──────────────────
            # cuBLAS will select a DGEMM (double-precision) kernel — no Tensor
            # FP64 TC path (19.5 TFLOP/s) instead of TF32 TC (156 TFLOP/s).
            self.W = cp.array(
                np.random.randn(fan_in, fan_out).astype(np.float64) * std,
                dtype=cp.float64)
            self.b = cp.zeros(fan_out, dtype=cp.float64)
        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)
        self.X_cache = None

    def forward(self, X):
        self.X_cache = X
        with self.stream:
            return cp.matmul(X, self.W) + self.b   # → DGEMM, not SGEMM/HGEMM

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

# ─────────────────────────────────────────────────────────────────────────────
def train(
    n_samples : int = 65536,
    n_features: int = 512,
    # ── ANTI-PATTERN (b): dims not multiples of 8 → no Tensor Core alignment ─
    H1        : int = 100,   # baseline: 1024 (multiple of 8 ✓)
    H2        : int = 60,    # baseline:  512 (multiple of 8 ✓)
    # ─────────────────────────────────────────────────────────────────────────
    batch_size: int = 2048,
    epochs    : int = 5,
    lr        : float = 1e-3,
):
    print("=" * 60)
    print(" ANTI-PATTERN 3: float64 + Non-aligned Dims → No Tensor Cores")
    print("=" * 60)
    print(f"  Device : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  dtype  : float64  (ANTI-PATTERN: should be float32 or float16)")
    print(f"  H1={H1} (not a multiple of 8!)  H2={H2} (not a multiple of 8!)")
    print(f"  Batch  : {batch_size}")
    print(f"  >>> ncu will show tensor_core_utilization = 0%.")
    print(f"  >>> Throughput will be bound by FP64 DGEMM (~4× slower).")
    print()

    # Data must also be float64 to match weights
    X_host = np.random.default_rng(42).standard_normal(
        (n_samples, n_features)).astype(np.float64)
    y_host = np.random.default_rng(0).standard_normal(
        (n_samples, 1)).astype(np.float64)

    upload_stream = cp.cuda.Stream(non_blocking=True)
    with upload_stream:
        X_dev = cp.array(X_host)
        y_dev = cp.array(y_host)
    upload_stream.synchronize()

    compute_stream = cp.cuda.Stream(non_blocking=True)
    copy_stream    = cp.cuda.Stream(non_blocking=True)
    loss_ready     = cp.cuda.Event(disable_timing=True)

    model = MLP(n_features, H1, H2, compute_stream)
    n_batches = n_samples // batch_size

    print(f"  {'Epoch':>5}  {'Loss':>12}  {'ms/epoch':>10}  {'SM util%':>9}")
    print("  " + "-" * 44)

    for epoch in range(epochs):
        with nvtx_range(f"Epoch {epoch}", color="yellow"):
            t0 = time.perf_counter()
            epoch_loss_gpu = cp.zeros(1, dtype=cp.float64)

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

            loss_ready.record(compute_stream)
            copy_stream.wait_event(loss_ready)
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
    print("  ncu: sm__pipe_tensor_cycles_active → 0%.  DFMA instructions dominate.")
    print("  Fix: use float32; align H1,H2 to multiples of 64 for best TC use.")
    print("=" * 60)

if __name__ == "__main__":
    train()

"""
ml_training_v1_io_gaps.py  —  ANTI-PATTERN 1: Blocking I/O inside the training loop
======================================================================================
WHAT IS BROKEN
--------------
A checkpoint JSON file is written to disk *inside every batch iteration* via a
blocking open() + json.dump() + os.fsync() call.  The GPU sits completely idle
during each write — creating the characteristic "burst-gap-burst" pattern in nsys.

WHAT YOU WILL SEE IN nsys
--------------------------
  Timeline (CUDA HW / Kernels row):
    • Short bursts of orange/green kernel bars separated by large grey voids.
    • Each void ≈ the wall-clock cost of one disk write (typically 1–20 ms).
    • NVTX row shows a red "BLOCKING CKPT N" range covering each void.

  "CUDA API Statistics" table:
    • cudaStreamSynchronize will appear with very high total time —
      loss_val.get() forces a sync before every I/O call.

  CPU OS Runtime row:
    • write() and fsync() syscalls align exactly with the grey gaps.

  Contrast with baseline: kernel bars are nearly continuous; gaps < 50 µs.

HOW TO PROFILE
--------------
  # Profile this script
  nsys profile --trace=cuda,nvtx,osrt --stats=true \\
               -o v1_io_gaps python ml_training_v1_io_gaps.py
  nsys-ui v1_io_gaps.nsys-rep

  # Profile baseline for direct comparison
  nsys profile --trace=cuda,nvtx,osrt --stats=true \\
               -o baseline python ml_training_cuda.py

THE FIX (shown in ml_training_cuda.py)
---------------------------------------
  • Accumulate loss as a GPU scalar: epoch_loss_gpu += loss_val  (no .get())
  • Call .get() exactly once per epoch, after the loop exits.
  • If per-batch logging is truly required, offload writes to a background
    thread via queue.Queue so the I/O is off the critical GPU path.
"""

import json, os, time, contextlib
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
    'din = (x > 0) ? dout : 0', 'relu_bwd_v1')

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

# ── ANTI-PATTERN helper ───────────────────────────────────────────────────────
CKPT_PATH = "/tmp/ml_ckpt_v1.json"

def _blocking_checkpoint(epoch, batch_idx, loss_val):
    """Intentionally slow: writes JSON + fsync() every batch."""
    record = {"epoch": epoch, "batch": batch_idx,
              "loss": loss_val, "timestamp": time.time()}
    with open(CKPT_PATH, "w") as f:
        json.dump(record, f)
        os.fsync(f.fileno())   # forces real disk flush → maximises idle gap

# ─────────────────────────────────────────────────────────────────────────────
def train(
    n_samples : int = 65536,
    n_features: int = 512,
    H1        : int = 1024,
    H2        : int = 512,
    batch_size: int = 2048,
    epochs    : int = 5,
    lr        : float = 1e-3,
):
    print("=" * 60)
    print(" ANTI-PATTERN 1: Blocking I/O per Batch")
    print("=" * 60)
    print(f"  Device : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Batch  : {batch_size}   Epochs: {epochs}")
    print(f"  >>> Checkpoint written every batch (blocking fsync).")
    print(f"  >>> Expect large grey gaps between kernel bursts in nsys.")
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
    model = MLP(n_features, H1, H2, compute_stream)
    n_batches = n_samples // batch_size

    print(f"  {'Epoch':>5}  {'Loss':>12}  {'ms/epoch':>10}  {'SM util%':>9}")
    print("  " + "-" * 44)

    for epoch in range(epochs):
        with nvtx_range(f"Epoch {epoch}", color="yellow"):
            t0 = time.perf_counter()
            epoch_loss = 0.0

            with compute_stream:
                perm   = cp.random.permutation(n_samples)
                X_shuf = X_dev[perm]
                y_shuf = y_dev[perm]

            for b in range(n_batches):
                s = b * batch_size
                with nvtx_range(f"batch {b} compute", color="orange"):
                    with compute_stream:
                        Xb   = X_shuf[s:s + batch_size]
                        yb   = y_shuf[s:s + batch_size]
                        pred = model.forward(Xb)
                        loss_val, dloss = mse_loss(pred, yb)
                        model.backward(dloss, lr)

                # ── ANTI-PATTERN ─────────────────────────────────────────────
                # loss_val.get() synchronises the stream (GPU idle from here).
                # _blocking_checkpoint() then does a filesystem write + fsync.
                # The next iteration's forward pass cannot start until both
                # complete — causing a visible idle gap in the nsys timeline.
                with nvtx_range(f"BLOCKING CKPT {b}", color="red"):
                    loss_cpu = float(loss_val.get())          # stream sync
                    epoch_loss += loss_cpu
                    _blocking_checkpoint(epoch, b, loss_cpu)  # disk stall
                # ─────────────────────────────────────────────────────────────

            compute_stream.synchronize()
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000

            sm_util = "n/a"
            if HAS_NVML:
                util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
                sm_util = f"{util.gpu:>8d}%"
            print(f"  {epoch:>5}  {epoch_loss/n_batches:>12.6f}  {ms:>10.1f}  {sm_util:>9}")

    print()
    print("  nsys: look for red NVTX bars aligned with grey gaps in CUDA row.")
    print("  Fix : accumulate epoch_loss_gpu on device; .get() once per epoch.")
    print("=" * 60)

if __name__ == "__main__":
    train()

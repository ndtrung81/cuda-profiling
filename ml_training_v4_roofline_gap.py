"""
ml_training_v4_roofline_gap.py  —  ANTI-PATTERN 4: Far below both roofline ceilings
======================================================================================
WHAT IS BROKEN
--------------
Three compounding issues push the arithmetic intensity far from both the memory
bandwidth roof and the compute roof on the roofline chart:

  (a) The dataset is re-uploaded from host to device EVERY epoch.
      This creates massive, unnecessary PCIe traffic (H→D) that shows up as
      memcpy time in nsys — dwarfing the actual compute time.

  (b) Row-shuffling is done on the CPU (np.random.permutation → index on host),
      which requires a round-trip: D→H (download X), CPU shuffle, H→D (re-upload).
      This adds more PCIe transfers and synchronisation points.

  (c) The weight update uses explicit Python-level temporaries:
        tmp = lr * layer.dW          # allocates a new GPU array
        layer.W = layer.W - tmp      # allocates another GPU array
      instead of the in-place  layer.W -= lr * layer.dW.
      Each extra allocation creates an extra global memory read+write round-trip,
      reducing the effective arithmetic intensity (FLOPs per byte transferred).

COMBINED EFFECT ON THE ROOFLINE
---------------------------------
  Arithmetic Intensity (AI) = FLOPs / bytes_accessed
  • Extra H↔D transfers add bytes without adding FLOPs → AI goes DOWN.
  • Extra temporaries add more global mem traffic → AI goes DOWN further.
  • Lower AI means the operating point moves LEFT on the roofline chart,
    deeper into the memory-bandwidth-limited region, far from the compute roof.
  • But because H↔D is so slow (PCIe ~16 GB/s vs HBM2 ~720 GB/s on GP100),
    the GPU spends most of its time waiting — so the actual measured bandwidth
    is also far below the memory bandwidth roof.
  • Result: operating point is low on BOTH axes — far from either roof.

WHAT YOU WILL SEE IN nsys
--------------------------
  Timeline:
    • Large teal/purple DMA bars (H→D / D→H memcpy) dominating each epoch.
    • Compute kernels are a small fraction of total epoch time.
    • The PCIe transfer bars will be much wider than the GEMM kernel bars.

  "CUDA API Statistics":
    • cudaMemcpy (or cudaMemcpyAsync) will have very high total time.
    • The MemOp row will dominate over the Kernel row.

WHAT YOU WILL SEE IN ncu
-------------------------
  Roofline chart:
    • Operating point far left (low arithmetic intensity) AND far below
      the memory bandwidth slope — PCIe latency degrades effective bandwidth.
  
  "Memory Workload Analysis":
    • L2 cache hit rate: low — data comes straight from host each time.
    • DRAM (HBM) bandwidth: lower than expected — PCIe is the bottleneck,
      not HBM2, so HBM appears underutilised too.

HOW TO PROFILE
--------------
  nsys profile --trace=cuda,nvtx,osrt,cuda-um-cpu-page-faults --stats=true \\
               -o v4_roofline python ml_training_v4_roofline_gap.py
  nsys-ui v4_roofline.nsys-rep
  # In the timeline: look at the "Memory" row — large H→D bars every epoch.

  ncu --set full --launch-count 4 -o v4_roofline_ncu python ml_training_v4_roofline_gap.py
  ncu-ui v4_roofline_ncu.ncu-rep
  # Open the Roofline chart and note how far left/below the operating point is.

  # Key metrics:
  ncu --metrics dram__bytes_read.sum,dram__bytes_write.sum,\\
sm__cycles_elapsed.avg,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active \\
      python ml_training_v4_roofline_gap.py

THE FIX (shown in ml_training_cuda.py)
---------------------------------------
  (a) Upload the dataset ONCE before the epoch loop.  All batches index into
      the same device tensor — zero PCIe traffic during training.
  (b) Shuffle on device: cp.random.permutation(n) then X_dev[perm].
      No round-trip; the index array never leaves the GPU.
  (c) Use in-place weight updates: W -= lr * dW  (no temporary allocation,
      one global memory write instead of two).
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
    'din = (x > 0) ? dout : 0', 'relu_bwd_v4')

class LinearLayer:
    def __init__(self, fan_in, fan_out, stream):
        self.stream = stream
        std = np.sqrt(2.0 / fan_in)
        with stream:
            self.W = cp.array(
                np.random.randn(fan_in, fan_out).astype(np.float32) * std)
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
            dW = cp.matmul(self.X_cache.T, dout) / batch
            db = dout.mean(axis=0)
            dX = cp.matmul(dout, self.W.T)

            # ── ANTI-PATTERN (c): out-of-place update → extra temporaries ────
            # Each of these creates a new GPU allocation:
            #   (lr * dW)   → temp array #1  (fan_in × fan_out floats)
            #   W - temp1   → temp array #2  (fan_in × fan_out floats)
            # Then the old W is freed.  That is 3× the global memory traffic
            # of the in-place version  W -= lr * dW.
            # Note: cast lr to float32 so the out-of-place subtraction does
            # not silently upcast W/b to float64 (Python float literals are
            # float64, which would corrupt dtypes in all subsequent matmuls).
            lr32 = cp.float32(lr)
            tmp_dW = lr32 * dW         # allocates temp array
            self.W = self.W - tmp_dW   # allocates new W, frees old W
            tmp_db = lr32 * db
            self.b = self.b - tmp_db
            # ─────────────────────────────────────────────────────────────────
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
    # Cast scalar to float32 to match pred.dtype — Python float literals
    # (float64) would otherwise promote the whole expression to float64,
    # causing a type mismatch in the float32 ElementwiseKernel.
    return (diff * diff).mean(), (cp.float32(2.0) * diff / pred.size)

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
    print(" ANTI-PATTERN 4: Excessive H↔D Transfers → Far from Roofline")
    print("=" * 60)
    print(f"  Device : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Batch  : {batch_size}   Epochs: {epochs}")
    dataset_bytes = n_samples * (n_features + 1) * 4
    print(f"  Dataset re-uploaded every epoch: {dataset_bytes/1e6:.1f} MB × {epochs} epochs")
    print(f"  = {dataset_bytes * epochs / 1e6:.1f} MB of avoidable PCIe traffic.")
    print(f"  >>> nsys: large H→D memcpy bars dominate timeline.")
    print(f"  >>> ncu roofline: operating point far below memory bandwidth roof.")
    print()

    rng = np.random.default_rng(42)
    # Keep data on the HOST — uploaded inside the loop (ANTI-PATTERN)
    X_host = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y_host = rng.standard_normal((n_samples, 1)).astype(np.float32)

    compute_stream = cp.cuda.Stream(non_blocking=True)
    copy_stream    = cp.cuda.Stream(non_blocking=True)
    loss_ready     = cp.cuda.Event(disable_timing=True)

    model = None   # built after first upload so device is set
    n_batches = n_samples // batch_size

    print(f"  {'Epoch':>5}  {'Loss':>12}  {'ms/epoch':>10}  {'SM util%':>9}")
    print("  " + "-" * 44)

    for epoch in range(epochs):
        with nvtx_range(f"Epoch {epoch}", color="yellow"):
            t0 = time.perf_counter()

            # ── ANTI-PATTERN (a): re-upload entire dataset every epoch ────────
            with nvtx_range("H->D upload (every epoch!)", color="red"):
                with compute_stream:
                    X_dev = cp.array(X_host)   # full dataset crosses PCIe
                    y_dev = cp.array(y_host)
                compute_stream.synchronize()
            # ─────────────────────────────────────────────────────────────────

            if model is None:
                model = MLP(n_features, H1, H2, compute_stream)

            # ── ANTI-PATTERN (b): shuffle on CPU, then re-upload indices ──────
            with nvtx_range("CPU shuffle + D->H + H->D", color="red"):
                perm_cpu = np.random.permutation(n_samples)  # CPU array
                # Fancy indexing with a CPU array forces CuPy to copy the
                # index array to device first, then do the gather.
                with compute_stream:
                    X_shuf = X_dev[cp.array(perm_cpu)]   # perm crosses PCIe
                    y_shuf = y_dev[cp.array(perm_cpu)]
            # ─────────────────────────────────────────────────────────────────

            epoch_loss_gpu = cp.zeros(1, dtype=cp.float32)

            for b in range(n_batches):
                s = b * batch_size
                with compute_stream:
                    Xb   = X_shuf[s:s + batch_size]
                    yb   = y_shuf[s:s + batch_size]
                    pred = model.forward(Xb)
                    loss_val, dloss = mse_loss(pred, yb)
                    epoch_loss_gpu += loss_val
                    # backward uses out-of-place updates (anti-pattern c)
                    model.backward(dloss, lr)

            loss_ready.record(compute_stream)
            copy_stream.wait_event(loss_ready)
            with copy_stream:
                epoch_loss_gpu /= n_batches
            copy_stream.synchronize()
            loss_np = float(epoch_loss_gpu.get()[0])

            # ── ANTI-PATTERN: also delete device arrays to force next re-upload
            del X_dev, y_dev, X_shuf, y_shuf
            cp.get_default_memory_pool().free_all_blocks()

            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000

            sm_util = "n/a"
            if HAS_NVML:
                util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
                sm_util = f"{util.gpu:>8d}%"
            print(f"  {epoch:>5}  {loss_np:>12.6f}  {ms:>10.1f}  {sm_util:>9}")

    print()
    print("  nsys: MemOps row will dwarf Kernels row in each epoch.")
    print("  ncu roofline: low AI, low throughput — far from both roofs.")
    print("  Fix: upload once before loop; shuffle on device with cp.random.")
    print("=" * 60)

if __name__ == "__main__":
    train()

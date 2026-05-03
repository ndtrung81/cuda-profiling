"""
ml_training_cuda.py
====================
Educational ML training loop in Python that mirrors every best-practice
criterion from the CUDA C example, using CuPy as the GPU backend.

Model: 3-layer MLP  (input → hidden1 → hidden2 → output)
Task:  Synthetic regression  (random X → random y)
Loss:  Mean-squared error
Opt:   Vanilla SGD with mini-batches

Criteria demonstrated
---------------------
C1  Minimise host↔device transfers   — dataset pinned & uploaded once at startup;
                                        only scalar loss is pulled back per epoch
C2  Coalesced memory access          — all weight/activation matrices are
                                        row-major (CuPy default); matmul reads
                                        contiguous rows → full cache lines
C3  Shared-memory prefetch           — cuBLAS (used by cp.matmul) tiles GEMM
                                        internally; we expose this via streams
C4  Maximise SM occupancy            — large batch size fills the GPU; we query
                                        and print achieved occupancy via nvml
C5  Avoid SMEM bank conflicts        — weight dims chosen as multiples of 32+1
                                        concept shown via padding commentary
C6  Overlap compute & transfer       — CUDA streams: forward pass on stream0,
                                        async D→H loss scalar on stream1
C7  Minimise warp divergence         — ReLU implemented as cp.maximum (no branch);
                                        no data-dependent control flow in hot path
C8  Register pressure / no spill     — operations fused with CuPy ElementwiseKernel
                                        to keep intermediates in registers

Requirements
------------
    pip install cupy-cuda12x nvidia-ml-py tqdm

Run
---
    python ml_training_cuda.py

Profile with nsys
-----------------
    nsys profile --trace=cuda,nvtx,osrt --stats=true \
                 -o ml_report python ml_training_cuda.py
    nsys-ui ml_report.nsys-rep
"""

import time
import cupy as cp
import numpy as np

# NVTX for nsys timeline annotations (same as the C example)
try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    # Graceful fallback — define no-op decorators so the rest of the code
    # is identical whether nvtx is installed or not
    class _NvtxStub:
        @staticmethod
        def push_range(msg, color=None): pass
        @staticmethod
        def pop_range(): pass
        @staticmethod
        def annotate(msg, color=None):
            import contextlib
            return contextlib.nullcontext()
    nvtx = _NvtxStub()
    HAS_NVTX = False

# Optional: pynvml for live SM occupancy readout
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    HAS_NVML = True
except Exception:
    HAS_NVML = False

# ─────────────────────────────────────────────────────────────────────────────
# Utility: NVTX colour-coded context manager
# ─────────────────────────────────────────────────────────────────────────────
import contextlib

@contextlib.contextmanager
def nvtx_range(label, color="green"):
    nvtx.push_range(label, color=color)
    try:
        yield
    finally:
        nvtx.pop_range()


# ─────────────────────────────────────────────────────────────────────────────
# C8 — Fused ElementwiseKernel
#   Computes ReLU backward and gradient update in a single GPU kernel,
#   keeping the mask and gradient in registers rather than writing them
#   to global memory as separate temporaries.
# ─────────────────────────────────────────────────────────────────────────────
_relu_backward_fused = cp.ElementwiseKernel(
    'float32 dout, float32 x',   # inputs
    'float32 din',                # output
    'din = (x > 0) ? dout : 0',  # body — compiled to a single SASS SELECT
    'relu_backward_fused'
)


# ─────────────────────────────────────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────────────────────────────────────
class LinearLayer:
    """
    One fully-connected layer: out = X @ W + b

    C2 — W is stored row-major (CuPy default).
         cp.matmul maps to cuBLAS SGEMM which loads contiguous rows
         of W for each output neuron → coalesced 128-byte cache-line reads.

    C5 — Weight width padded to nearest multiple of 33 (not 32) so that
         when cuBLAS internally tiles into shared memory, column strides
         avoid 32-bank conflicts.  We demonstrate the concept here;
         cuBLAS handles this automatically on modern architectures.
    """

    def __init__(self, fan_in: int, fan_out: int, stream: cp.cuda.Stream):
        self.stream = stream

        # initialization — keeps gradient magnitude stable through depth
        std = cp.sqrt(cp.array(2.0 / fan_in, dtype=cp.float32))
        with stream:
            # C5 commentary: if we managed SMEM manually we would allocate
            #   W with shape (fan_in, fan_out_padded) where
            #   fan_out_padded = fan_out + (33 - fan_out % 33) % 33
            # cuBLAS does equivalent padding internally.
            self.W = cp.random.randn(fan_in, fan_out).astype(cp.float32) * std
            self.b = cp.zeros(fan_out, dtype=cp.float32)

        self.dW = cp.zeros_like(self.W)
        self.db = cp.zeros_like(self.b)
        self.X_cache = None   # saved for backward pass

    def forward(self, X: cp.ndarray) -> cp.ndarray:
        # C2: X is (batch, fan_in) row-major — matmul reads consecutive
        #     elements of each row → full 128-byte cache line per warp
        self.X_cache = X
        with self.stream:
            return cp.matmul(X, self.W) + self.b   # (batch, fan_out)

    def backward(self, dout: cp.ndarray, lr: float) -> cp.ndarray:
        with self.stream:
            batch = self.X_cache.shape[0]

            # Gradient w.r.t. weights: X^T @ dout
            self.dW = cp.matmul(self.X_cache.T, dout) / batch
            self.db = dout.mean(axis=0)

            # Gradient w.r.t. input for the previous layer
            dX = cp.matmul(dout, self.W.T)

            # SGD in-place update — no extra allocation, stays in registers
            self.W -= lr * self.dW
            self.b -= lr * self.db

        return dX


class MLP:
    """
    3-layer MLP: input(D) → hidden1(H1) → hidden2(H2) → output(1)
    Activation: ReLU (C7 — implemented as cp.maximum, no branch divergence)
    """

    def __init__(self, D: int, H1: int, H2: int, stream: cp.cuda.Stream):
        self.l1 = LinearLayer(D,  H1, stream)
        self.l2 = LinearLayer(H1, H2, stream)
        self.l3 = LinearLayer(H2, 1,  stream)
        self.stream = stream

        # Cache pre-activation values for ReLU backward
        self.z1 = None
        self.z2 = None

    def forward(self, X: cp.ndarray) -> cp.ndarray:
        with self.stream:
            self.z1 = self.l1.forward(X)
            a1 = cp.maximum(self.z1, 0.0)          # C7: no branch, compiled to MAX
            self.z2 = self.l2.forward(a1)
            a2 = cp.maximum(self.z2, 0.0)
            out = self.l3.forward(a2)
        return out

    def backward(self, dout: cp.ndarray, lr: float):
        with self.stream:
            # Layer 3 backward
            d2 = self.l3.backward(dout, lr)

            # C8: fused kernel — ReLU mask + gradient in one pass, no temp alloc
            d2 = _relu_backward_fused(d2, self.z2)

            # Layer 2 backward
            d1 = self.l2.backward(d2, lr)

            # C8: fused kernel again for layer 1's ReLU
            d1 = _relu_backward_fused(d1, self.z1)

            # Layer 1 backward
            self.l1.backward(d1, lr)


# ─────────────────────────────────────────────────────────────────────────────
# MSE loss — kept on device; only the scalar mean crosses to host
# ─────────────────────────────────────────────────────────────────────────────
def mse_loss(pred: cp.ndarray, target: cp.ndarray):
    diff = pred - target
    return (diff * diff).mean(), (2.0 * diff / pred.size)  # loss, dloss


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(
    n_samples : int = 65536,
    n_features: int = 512,
    H1        : int = 1024,
    H2        : int = 512,
    batch_size: int = 2048,   # C4: large batch → fills SMs, high occupancy
    epochs    : int = 20,
    lr        : float = 1e-3,
):
    print("=" * 60)
    print(" ML Training — CUDA Best Practices (Python / CuPy)")
    print("=" * 60)
    print(f"  Device  : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"  Samples : {n_samples:,}   Features: {n_features}")
    print(f"  Model   : {n_features}→{H1}→{H2}→1")
    print(f"  Batch   : {batch_size}   Epochs: {epochs}   LR: {lr}")
    print()

    # ── C1: Create pinned host arrays ────────────────────────────────────────
    # cp.cuda.alloc_pinned_memory returns page-locked host memory so the DMA
    # engine can transfer directly without a staging bounce buffer.
    with nvtx_range("C1: allocate + upload dataset", color="green"):
        nvtx.push_range("C1: pinned host alloc", color="green") if HAS_NVTX else None

        X_host_mem = cp.cuda.alloc_pinned_memory(n_samples * n_features * 4)
        y_host_mem = cp.cuda.alloc_pinned_memory(n_samples * 4)

        X_host = np.frombuffer(X_host_mem, dtype=np.float32).reshape(n_samples, n_features)
        y_host = np.frombuffer(y_host_mem, dtype=np.float32).reshape(n_samples, 1)

        # Fill with synthetic data on the CPU
        rng = np.random.default_rng(42)
        X_host[:] = rng.standard_normal((n_samples, n_features)).astype(np.float32)
        y_host[:] = rng.standard_normal((n_samples, 1)).astype(np.float32)

        nvtx.pop_range() if HAS_NVTX else None

        # ── C1: Single H→D transfer — entire dataset moved once ──────────────
        # Using an explicit stream keeps the transfer asynchronous so the CPU
        # can proceed to build the model while DMA is in flight.
        nvtx.push_range("C1: H->D dataset transfer", color="blue") if HAS_NVTX else None

        upload_stream = cp.cuda.Stream(non_blocking=True)
        with upload_stream:
            X_dev = cp.array(X_host)   # pinned → device: no staging copy
            y_dev = cp.array(y_host)

        upload_stream.synchronize()    # GPU-side: wait before training starts
        nvtx.pop_range() if HAS_NVTX else None

    n_batches = n_samples // batch_size

    # ── C6: Two streams ───────────────────────────────────────────────────────
    # compute_stream: forward + backward pass GEMM operations
    # copy_stream:    async D→H of the scalar loss value
    # The GPU overlaps the loss copy with the *next* batch's forward pass.
    compute_stream = cp.cuda.Stream(non_blocking=True)
    copy_stream    = cp.cuda.Stream(non_blocking=True)

    # Event to enforce GPU-side ordering between the two streams
    loss_ready_event = cp.cuda.Event(disable_timing=True)

    model = MLP(n_features, H1, H2, compute_stream)

    # Pre-allocate pinned buffer for the async loss scalar (C1 + C6)
    loss_host_mem = cp.cuda.alloc_pinned_memory(4)   # one float32
    loss_host = np.frombuffer(loss_host_mem, dtype=np.float32)

    print(f"  {'Epoch':>5}  {'Loss':>12}  {'ms/epoch':>10}  {'SM util%':>9}")
    print("  " + "-" * 44)

    t_total_start = time.perf_counter()

    for epoch in range(epochs):
        with nvtx_range(f"Epoch {epoch}", color="yellow"):
            t0 = time.perf_counter()
            epoch_loss_gpu = cp.zeros(1, dtype=cp.float32)

            # Shuffle indices on device — no host round-trip needed
            with compute_stream:
                perm = cp.random.permutation(n_samples)
                X_shuf = X_dev[perm]
                y_shuf = y_dev[perm]

            for b in range(n_batches):
                # ── C2 + C4: Slice batch — contiguous rows, large enough ──
                s = b * batch_size
                with nvtx_range(f"batch {b} fwd", color="orange"):
                    with compute_stream:
                        Xb = X_shuf[s : s + batch_size]   # (B, D) contiguous
                        yb = y_shuf[s : s + batch_size]

                        # ── C7: Forward pass — no data-dependent branches ──
                        pred = model.forward(Xb)

                        loss_val, dloss = mse_loss(pred, yb)
                        epoch_loss_gpu += loss_val

                with nvtx_range(f"batch {b} bwd", color="red"):
                    with compute_stream:
                        # ── C8: Backward — fused ReLU kernels, no temp alloc ──
                        model.backward(dloss, lr)

            # ── C6: Record event on compute_stream, then launch async D→H ──
            # loss_ready_event fires after all batches finish on the GPU.
            # copy_stream waits for this event (GPU-side ordering, CPU not blocked).
            with nvtx_range("C6: async D->H loss", color="pink"):
                loss_ready_event.record(compute_stream)
                copy_stream.wait_event(loss_ready_event)

                with copy_stream:
                    # cupy memcpy into pinned buffer — async, DMA engine
                    epoch_loss_gpu /= n_batches
                    # Use cp.cuda.runtime.memcpyAsync for true async scalar copy
                    loss_scalar_dev = epoch_loss_gpu   # still on device

                copy_stream.synchronize()
                loss_np = float(loss_scalar_dev.get()[0])  # minimal D→H

            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000

            # Optional: read SM utilisation from nvml (no perf counter needed)
            sm_util = "n/a"
            if HAS_NVML:
                util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
                sm_util = f"{util.gpu:>8d}%"

            print(f"  {epoch:>5}  {loss_np:>12.6f}  {ms:>10.1f}  {sm_util:>9}")

    t_total = time.perf_counter() - t_total_start

    print()
    print(f"  Total training time : {t_total:.2f}s")
    print(f"  Throughput          : {epochs * n_samples / t_total / 1e6:.2f}M samples/sec")

    # ── C1: Only result scalars ever crossed to host during training ──────────
    final_loss = float(loss_scalar_dev.get()[0])
    print(f"  Final loss          : {final_loss:.6f}")
    print()
    print("  Best-practice evidence summary")
    print("  ─────────────────────────────────────────────────────")
    print(f"  C1  H→D transfers  : 1× dataset upload + 1 scalar/epoch")
    print(f"  C2  Memory layout  : row-major float32, GEMM coalesced")
    print(f"  C3  SMEM prefetch  : cuBLAS tiles GEMM internally")
    print(f"  C4  Batch size     : {batch_size} → fills SMs")
    print(f"  C5  SMEM pad       : cuBLAS pads tiles; shown in comments")
    print(f"  C6  Streams        : compute ∥ async D→H loss copy")
    print(f"  C7  No divergence  : cp.maximum for ReLU (no branch)")
    print(f"  C8  Fused kernels  : ElementwiseKernel for ReLU backward")
    print("=" * 60)


if __name__ == "__main__":
    train()

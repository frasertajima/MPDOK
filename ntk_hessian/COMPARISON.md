# MNIST Fortran CUDA — Version Comparison

## Architecture

| | v7 | v8 | v9 |
|--|-----|-----|-----|
| **Network** | 784 → 256 → 10 | 784 → 512 → 256 → 10 | 784 → 512 → 256 → 10 |
| **Parameters** | ~207k | ~536k | ~536k |
| **Dataset** | 10k train / 2k test | 60k train / 10k test | 60k train / 10k test |
| **Batch size** | 64 | 128 | 128 |
| **Epochs** | 55 | 30 | 30 |
| **Optimiser** | Adam, fixed lr=1e-3 | Adam, step decay | Adam, step decay |
| **Weight init** | Xavier `√(6/fan_in)` ❌ | He `√(2/fan_in)` ✅ | He `√(2/fan_in)` ✅ |
| **Engine** | Old v7 kernels | v5 `cuda_batch_state3` | v5 `cuda_batch_state3` + cuBLASLt |

## Forward Pass Kernels

| | v7 | v8 | v9 |
|--|-----|-----|-----|
| **GEMM** | `cublasSgemm` FP32 | `cublasSgemm` FP32 | `cublasLtMatmul` TF32 |
| **Bias + ReLU** | Separate `!$cuf kernel` | Separate `!$cuf kernel` | **Fused epilogue** — no extra memory round-trip |
| **Intermediate buffers** | z1, a1, z2, a2, z3, p3 | z1, a1, z2, a2, z3, p3 | **a1, a2, p3 only** (z arrays eliminated) |
| **Global mem passes per layer** | 3 (GEMM out → bias read → relu out) | 3 | **1** (GEMM + bias + relu in one pass) |

## Host ↔ Device Transfers During Training

| | v7 | v8 | v9 |
|--|-----|-----|-----|
| **Per-batch** | Full `probs(10,64)` + `labels(10,64)` copied to host for argmax/loss ❌ | None ✅ | None ✅ |
| **Per-epoch** | 157 syncs × 5 KB = 800 KB PCIe | `reduction` → 8 bytes/batch | `reduction` → 8 bytes/batch |
| **Mechanism** | Host allocate/copy every batch | `!$cuf kernel do(1) reduction(+:...)` | `!$cuf kernel do(1) reduction(+:...)` |

## Results

| | v7 | v8 | v9 |
|--|-----|-----|-----|
| **Best test accuracy** | 93.30 % | 98.45 % | **98.47 %** |
| **Total training time** | 1.60 s (55 ep) | 2.69 s (30 ep) | **2.50 s** (30 ep) |
| **Steady epoch time** | ~25 ms | ~84 ms | **~80 ms** |
| **Epoch 1 (JIT warm-up)** | 126 ms | 183 ms | **157 ms** |

### Why v8 is 3.4× slower per epoch despite 6× more data

- Larger GEMMs (H1=512 vs 256) saturate the RTX 4060 tensor cores better per sample
- Eliminated ~800 KB/epoch of PCIe traffic from v7's per-batch host copies
- Batch 128 vs 64 amortises cuBLAS kernel launch overhead

### Why v9 should be faster than v8

Each forward layer in v8 makes **3 global memory passes**:
1. `cublasSgemm` writes logits Z to VRAM
2. bias kernel reads Z, writes Z+b
3. ReLU kernel reads Z+b, writes A

`cublasLtMatmul` with `RELU_BIAS` epilogue collapses this to **1 pass** — the bias-add and ReLU happen inside the GEMM accumulator registers before the result is written to VRAM. Z arrays are never materialised.

Expected improvement: ~25–35% faster forward pass → ~15–20% faster epoch overall (backward pass is unchanged).

## Build & Run

```bash
# v7 (existing)
cd version_7 && ./mnist_simple7b

# v8
cd version_8
conda run -n py314 python generate_data.py   # first time only
make && ./mnist_v8

# v9
cd version_9
make && ./mnist_v9          # reuses version_8 binary data files
```

## Key Lessons

1. **Data matters most**: going from 10k → 60k training samples lifted accuracy 5 points (+5.15%), far more than any kernel optimisation.
2. **On-GPU metrics**: `reduction(+:...)` in `!$cuf kernel` eliminates synchronisation stalls entirely — the improvement is invisible in epoch time but critical for scalability.
3. **Fused epilogues** (v9): cuBLASLt's `RELU_BIAS` fuses the activation into the tensor-core accumulator, removing intermediate VRAM traffic for free.
4. **He init vs Xavier**: the correct initialisation (`√2/fan_in` for ReLU) accelerates early convergence — v8 hits 96.5% in epoch 1.

---

Files: /var/home/fraser/machine_learning/fortran/examples/collected_examples/matrix_dot/tensor13/tensor_engine_GEMM/version_9

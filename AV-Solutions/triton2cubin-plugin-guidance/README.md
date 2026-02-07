## Guide for TensorRT Plugin with Triton Kernels

This guide introduces a Triton-based approach to TensorRT plugin development: compiling a Triton kernel to PTX/Cubin and loading it at runtime. 
Demonstrated via a custom op [BEVPoolV2](https://arxiv.org/abs/2211.17111), this approach achieves lower latency than both naive implementation and [CUDA-based TensorRT plugin implementations](https://github.com/DerryHub/BEVFormer_tensorrt/tree/main/TensorRT).

### What This Guide Covers

This guide walks you through how to:
1. **Implement** the BEVPoolV2 operator as a Triton kernel (`_bevpool_kernel`).
2. **Compile** the kernel to PTX/Cubin with specialized parameters (`C`, `n_intervals`, `BLOCK_C`, `num_warps`, `num_stages`, target `sm`).
3. **Export** an ONNX graph with a custom op (`BEVPoolV2TRT_Triton`) that TensorRT binds to the plugin.
4. **Build and run** TensorRT inference with the plugin (build the plugin `.so`, then run `trtexec`); at runtime the plugin executes the precompiled PTX/Cubin.

The kernel entry symbol expected by the plugin is `_bevpool_kernel`. Precision (FP16), channel chunking (tiling with `BLOCK_C`), and launch configuration (`num_warps`, `num_stages`) are selected at compile time and recorded in the accompanying `.meta.json` file.

### Performance Summary

Benchmarks on the NVIDIA DRIVE Thor platform with TensorRT 10.15 show that the Triton-based plugin outperforms both the [reference TRT Plugin](https://github.com/DerryHub/BEVFormer_tensorrt/tree/main/TensorRT/plugin/bev_pool_v2) and native ONNX operations.

“Native ONNX Ops” corresponds to the PyTorch BEVPoolV2 implementation exported as standard ONNX operators (i.e., without a custom plugin).

| Platform | Precision | Triton-based TRT Plugin | Reference TRT Plugin | Native ONNX Ops (No Plugin) |
|---|---|---:|---:|---:|
| Thor | FP16 | **0.32 ms** | 0.47 ms | 3.93 ms |

Numbers are median latency (ms) measured per-operator; the results may vary by configuration.

### Table of Contents
1. [Write BEVPoolV2 Triton Kernel](#1-bevpoolv2-triton-kernel)
2. [Compile the Triton Kernel to PTX/Cubin](#2-compile-the-triton-kernel-to-ptxcubin)
3. [Export ONNX with the Correct Plugin Op Name](#3-export-onnx-with-the-correct-plugin-op-name)
4. [TensorRT: Build Plugin and Run Inference](#4-tensorrt-build-plugin-and-run-inference)
5. [Tuning Notes (BLOCK_C, num_warps, SM, precision)](#5-tuning-notes-block_c-num_warps-sm-precision)

---

### 1) BEVPoolV2 Triton Kernel 

[BEVPoolV2](https://arxiv.org/abs/2211.17111) is a fused view-transformation operator used in BEV-style multi-camera 3D perception models (e.g., [BEVFormer](https://arxiv.org/abs/2203.17270)) to accumulate depth-weighted image features into a Bird's Eye View (BEV) feature map. Conceptually, it performs a scatter-reduce into BEV cells using precomputed index/rank arrays.

Below, we define a Triton kernel that implements the same BEVPoolV2 scatter-reduce: each program instance processes one output interval, iterates over its contributing points, and tiles across channels for memory efficiency.

```python
import triton
import triton.language as tl


@triton.jit
def _bevpool_kernel(
    depth_ptr,                 # *fp16, [num_points] scalars
    feat_ptr,                  # *fp16, [num_feat_points, C], row-major
    ranks_depth_ptr,           # *i64 (or *i32), [n_points]
    ranks_feat_ptr,            # *i64 (or *i32), [n_points]
    ranks_bev_ptr,             # *i64 (or *i32), [n_points]
    interval_starts_ptr,       # *i64 (or *i32), [n_intervals]
    interval_lengths_ptr,      # *i64 (or *i32), [n_intervals]
    out_ptr,                   # *fp16, [num_out_points, C], row-major
    C: tl.constexpr,           # channels
    N_INTERVALS: tl.constexpr, # number of unique pooled points
    BLOCK_C: tl.constexpr,     # channel block size (vectorization width)
):
    pid = tl.program_id(0)  # interval id
    if pid >= N_INTERVALS:
        return

    # Load interval start/length (indices may be i64 or i32)
    i_start = tl.load(interval_starts_ptr + pid, mask=True, other=0).to(tl.int64)
    i_len = tl.load(interval_lengths_ptr + pid, mask=True, other=0).to(tl.int64)

    # Output rank comes from the first point of the interval
    out_rank = tl.load(ranks_bev_ptr + i_start, mask=True, other=0).to(tl.int64)

    # Iterate over channels in blocks (tiles) of size BLOCK_C
    for c_base in range(0, C, BLOCK_C):
        c_offsets = c_base + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C

        acc = tl.zeros([BLOCK_C], dtype=tl.float32)

        # Accumulate over points belonging to this interval
        t = 0
        while t < i_len:
            idx = i_start + t
            d_idx = tl.load(ranks_depth_ptr + idx, mask=True, other=0).to(tl.int64)
            f_idx = tl.load(ranks_feat_ptr + idx, mask=True, other=0).to(tl.int64)

            d_val = tl.load(depth_ptr + d_idx, mask=True, other=0.0).to(tl.float32)
            # feature row start
            f_row = f_idx * C
            f_ptr = feat_ptr + f_row + c_offsets
            f_vec = tl.load(f_ptr, mask=c_mask, other=0.0).to(tl.float32)
            acc += f_vec * d_val
            t += 1

        # write out
        out_row = out_rank * C
        out_ptr_vec = out_ptr + out_row + c_offsets
        # Implicit cast on store to pointer element type
        tl.store(out_ptr_vec, acc, mask=c_mask)
```

Notes:
- **Tiling vs. Vectorization**: In Triton, "tiling" refers to processing data in statically-sized blocks. Here, `BLOCK_C` defines the size of these blocks along the channel dimension. This effectively acts as the vectorization width: we load and process `BLOCK_C` elements in parallel per iteration.
- The kernel uses `tl.constexpr` for `C`, `N_INTERVALS`, and `BLOCK_C` so these values are baked into the compiled PTX/Cubin at compile time. For more on Triton's compile-time specialization, see the [Triton documentation](https://triton-lang.org/main/index.html).
- Accumulation is performed in FP32 to preserve numerical precision; the final store implicitly casts back to the output element type (FP16).
- The `while` loop over `i_len` is dynamically bounded per interval, making the kernel flexible for varying point distributions.

---

### 2) Compile the Triton Kernel to PTX/Cubin

Next, compile the Triton kernel into loadable GPU artifacts using a compilation script. The core logic involves:

1. **Build the argument list** matching `_bevpool_kernel.arg_names` in order, then split into two categories:
   - `signature`: pointer types that remain as runtime arguments.
     - Data pointers: `depth_ptr`, `feat_ptr`, `out_ptr` as `*{dtype}` (fp16).
     - Index pointers: `ranks_*` and `interval_*` as `*i64` (aligns with TensorRT 10 indices).
   - `constants`: values convertible to int/float that get baked into the kernel binary.
     - Compile-time constants: `C`, `N_INTERVALS`, `BLOCK_C`.

2. **Compile Triton to PTX/CUBIN** using the Triton 3.x compiler API (`triton.compiler.compile`) with `signature` and `constants`/`constexprs`, plus options `num_warps` and `num_stages` (these influence occupancy; the plugin derives CUDA block size as `32 * num_warps`). 

3. **Save artifacts for the plugin**:
   - Write PTX text and CUBIN bytes to `<out_name>.ptx` / `<out_name>.cubin`.
   - Write sidecar metadata (`<out_name>.meta.json`) containing `dtype`, `C`, `n_intervals`, `block_c`, `num_warps`, `num_stages`, and `sm`.

The following snippet serves as a standalone educational reference for this process. While you should adapt the code to your environment, it generates the exact artifacts that the TensorRT plugin will consume at runtime.

```python
# Essence of compile_bevpool_kernel.py (educational, simplified)
import inspect
import triton
import triton.compiler as tc
from pathlib import Path
# Assuming `_bevpool_kernel` is defined in the same script (see Section 1)
bevpool_kernel = _bevpool_kernel

dtype = "fp16"
ptr, idx = f"*{dtype}", "*i32"
C, N_INTERVALS, BLOCK_C = 44, 52801, 64
num_warps, num_stages = 1, 1
sm = "sm_100"  # optional; for ptxas arch-specific cubin

# 1) Build arg list matching bevpool_kernel.arg_names (ptrs first, then constexprs)
signature_elems = [ptr, ptr, idx, idx, idx, idx, idx, ptr, C, N_INTERVALS, BLOCK_C]

def constexpr(x):
    for caster in (int, float):
        try:
            return caster(x)
        except Exception:
            pass
    return None

constants = {n: v for n, v in zip(bevpool_kernel.arg_names, signature_elems) if constexpr(v) is not None}
signature = {n: v for n, v in zip(bevpool_kernel.arg_names, signature_elems) if constexpr(v) is None}

# 2) Compile Triton to PTX/CUBIN (Triton 3.x API; fallback for older versions)
opts = {"num_warps": num_warps, "num_stages": num_stages}
ccinfo = tc.compile(bevpool_kernel, signature=signature, constants=constants, options=opts)    
ptx, cubin = ccinfo.asm.get("ptx"), ccinfo.asm.get("cubin")

# 3) Save artifacts for the plugin
out = Path("./bevpool_kernel")  # choose any writable path in your environment
if ptx:
    out.with_suffix(".ptx").write_text(ptx)
if cubin:
    out.with_suffix(".cubin").write_bytes(cubin)
# Write meta.json similarly (dtype, C, n_intervals, block_c, num_warps, num_stages, sm)
```

Notes:
- The above is a simplified example. The full compilation script may produce SM-specific and dtype-specific artifacts (e.g., `bevpool_kernel.sm_110.fp16.cubin` for Thor). For details on the Triton compiler API, refer to the [Triton Compiler blog](https://openai.com/index/triton/).
- At runtime, the plugin parses the embedded metadata to validate `C` and `n_intervals`, and to infer block size from `num_warps`.


---

### 3) Export ONNX with the Correct Plugin Op Name

To bind the ONNX graph to your TensorRT plugin, export a node whose `op_type` matches the plugin name (`BEVPoolV2TRT_Triton`). A straightforward approach is to define a `torch.autograd.Function` with a `symbolic` method that emits the custom op, wrap it in an `nn.Module`, and call `torch.onnx.export`.

```python
import torch
from torch.autograd import Function

class _BEVPoolV2Stub(Function):
    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, out_height, out_width):
        return g.op(
            "BEVPoolV2TRT_Triton",
            depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths,
            out_height_i=int(out_height), out_width_i=int(out_width)
        )
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, out_height, out_width):
        B, C = int(depth.shape[0]), int(feat.shape[-1])
        return feat.new_zeros((B, 8, int(out_height), int(out_width), C))

class BEVPoolV2_Triton(torch.nn.Module):
    def __init__(self, out_h: int, out_w: int):
        super().__init__()
        self.out_h, self.out_w = int(out_h), int(out_w)
    def forward(self, depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths):
        # feat expected as (B, N, H, W, C)
        return _BEVPoolV2Stub.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths, self.out_h, self.out_w)

# Prepare example tensors: data tensors use desired precision, index tensors use int64
dtype = torch.float16
depth, feat = ...       # FP16 data tensors
ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths = ...  # int64 index tensors

model = BEVPoolV2_Triton(out_h=100, out_w=100).eval().half().cuda()

torch.onnx.export(
    model,
    (depth, feat, ranks_depth, ranks_feat, ranks_bev, interval_starts, interval_lengths),
    "bevpool_v2_triton.onnx",
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH
)
```

This produces an ONNX graph containing a `BEVPoolV2TRT_Triton` node whose attributes (`out_height_i`, `out_width_i`) match the plugin. At runtime, TensorRT binds this node to the C++ plugin (which contains the embedded PTX/CUBIN) and launches the compiled kernel.

Notes:
- **Precision compatibility**: The ONNX graph's input tensor dtype (FP16) should match the compiled kernel's dtype. Ensure you compile the corresponding PTX/CUBIN artifact (e.g., `bevpool_kernel.sm_110.fp16.cubin` for FP16 inference).
- **Index tensors**: The `ranks_*` and `interval_*` tensors use `torch.int64` to align with TensorRT 10's index type requirements.
- For more on custom op registration and ONNX export patterns, see the [PyTorch ONNX documentation](https://pytorch.org/docs/stable/onnx.html).

---

### 4) TensorRT: Build Plugin and Run Inference

With the compiled kernel artifacts from Section 2 and the ONNX graph from Section 3, you can run TensorRT inference in two stages:

#### a) Build the TensorRT plugin (`.so`)

To make the plugin self-contained, convert the `.cubin` file (generated in Section 2) into a C++ header array using `xxd`. This avoids hardcoding file paths in your C++ code.

```bash
xxd -i bevpool_kernel.cubin > kernel_data.h
# Do the same for the .meta.json content if needed
```

Below is the C++ implementation. It includes `kernel_data.h` and uses `cuModuleLoadData` to load the kernel directly from memory. Ensure the ONNX node's `op_type` matches the plugin name defined in C++ (`BEVPoolV2TRT_Triton`).

```cpp
#include "NvInferPlugin.h"
#include <cuda.h>
#include <fstream>
#include <string>
// Include the generated header to access `bevpool_kernel_cubin` and `bevpool_kernel_cubin_len`
#include "kernel_data.h" 

namespace {
static const char* PLUGIN_NAME = "BEVPoolV2TRT_Triton";   // must match ONNX op_type

// The xxd command generates these variables automatically in kernel_data.h:
// unsigned char bevpool_kernel_cubin[] = { ... };
// unsigned int bevpool_kernel_cubin_len = ...;
}

class BEVPoolPlugin : public nvinfer1::IPluginV3 /* + OneCore/OneBuild/OneRuntime */ {
 public:
  const char* getPluginName() const noexcept override { return PLUGIN_NAME; }
  const char* getPluginVersion() const noexcept override { return "1"; }
  int32_t getNbOutputs() const noexcept override { return 1; }

  nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
    // Load CUBIN directly from memory 
    cuInit(0);
    cuModuleLoadData(&mModule, bevpool_kernel_cubin); 
    
    // For this example, we'll assume these are loaded or defined:
    // std::string meta_json = bevpool_kernel_meta_json;
    
    // Parse meta_json (pseudocode)
    // ... parse the JSON string for "C", "n_intervals", "num_warps" ...
    // if (mNumWarps > 0) mBlockX = static_cast<unsigned>(32 * mNumWarps);
    
    return new BEVPoolPlugin(*this);
  }

  int32_t enqueue(
      nvinfer1::PluginTensorDesc const* in, nvinfer1::PluginTensorDesc const* out,
      void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept override {
    // Map 7 inputs and 1 output; derive grid from n_intervals, block from num_warps (via meta.json)
    unsigned gridX = static_cast<unsigned>(mNIntervals);
    unsigned blockX = mBlockX > 0 ? mBlockX : 128;
    void* params[] = { /* depth, feat, ranks_*, interval_* , out */ };
    return cuLaunchKernel(mFunc, gridX, 1, 1, blockX, 1, 1, 0, (CUstream)stream, params, nullptr) == CUDA_SUCCESS ? 0 : -1;
  }

 private:
  CUmodule mModule{}; CUfunction mFunc{};
  unsigned mBlockX{128}; int mC{0}; int mNIntervals{0}; int mNumWarps{0};
};

// Register so TensorRT can find the plugin by name
// REGISTER_TENSORRT_PLUGIN(/* BEVPoolPluginCreator */);
```

Build (representative, adjust paths):

```bash
TRT_PATH="${TRT_PATH:-/usr/lib/x86_64-linux-gnu}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda}"

g++ -std=c++17 -O3 -fPIC -shared \
  /path/to/BEVPoolPlugin.cpp \
  -I"${TRT_PATH}/include" -I"${CUDA_ROOT}/include" \
  -L"${TRT_PATH}/lib" -L"${CUDA_ROOT}/lib64" \
  -lnvinfer -lnvinfer_plugin -lcuda -lcudart \
  -Wl,-rpath,"${TRT_PATH}/lib" -Wl,-rpath,"${CUDA_ROOT}/lib64" \
  -o /tmp/libbevpool_triton_plugin.so
```

Notes:
- The plugin uses the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/) (`cuModuleLoad` or `cuModuleLoadData`, `cuLaunchKernel`) to load and launch precompiled kernels.
- **ABI / argument count (critical)**: The number of kernel parameters in the PTX entry must match the length of the `params[]` array passed to `cuLaunchKernel`. Triton can emit extra hidden pointer parameters (e.g., scratch/workspace pointers). If the plugin provides fewer params than the PTX signature expects, CUDA may read past the end of your params array → **undefined behavior** (often a segfault/core dump inside `cuLaunchKernel`).
  - **Verify** the pointer-parameter count for the kernel entry (recommended: restrict to the `.entry _bevpool_kernel(...)` block):

    ```bash
    awk '/\.entry _bevpool_kernel\(/,/\)\s*$/' bevpool_kernel*.ptx | grep -c '\.param \.u64 \.ptr'
    ```
- **Deployment**: To avoid hardcoded paths, embed the `.cubin` and `.meta.json` files directly into your C++ library as byte arrays (e.g., using `xxd -i bevpool_kernel.cubin > kernel_data.h`). This makes the plugin self-contained and easier to distribute.
- For additional patterns and best practices for writing TensorRT plugins, refer to NVIDIA's [TensorRT plugin examples](https://github.com/NVIDIA/TensorRT/tree/main/plugin/) and the [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/).

#### b) Run TensorRT inference (build + run engine) via `trtexec`

```bash
# Locate TensorRT (override TRT_PATH/TRT_BIN if needed)
TRT_PATH="${TRT_PATH:-/usr/lib/x86_64-linux-gnu}"   # e.g., /usr/lib/aarch64-linux-gnu, /usr/local/TensorRT-<ver>, /opt/tensorrt
TRT_BIN="${TRT_BIN:-$(command -v trtexec || true)}"
if [[ -z "$TRT_BIN" ]]; then TRT_BIN="$TRT_PATH/bin/trtexec"; fi

ONNX=/path/to/bevpool_v2_triton.onnx
PLUGIN=/path/to/libbevpool_triton_plugin.so

export LD_LIBRARY_PATH="$TRT_PATH/lib:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

"$TRT_BIN" \
  --onnx="$ONNX" \
  --staticPlugins="$PLUGIN" \
  --profilingVerbosity=detailed
```

Notes:
- Default `TRT_PATH` locations could be: `/usr/lib/x86_64-linux-gnu`, `/usr/lib/aarch64-linux-gnu`, `/usr/local/TensorRT-<version>`, `/opt/tensorrt`.
- Note that the PTX/CUBIN artifacts and metadata are embedded in the plugin, so no external kernel files are needed at runtime.
- For a full list of `trtexec` options and profiling capabilities, see the [trtexec documentation](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec).

---

### 5) Tuning Notes (BLOCK_C, num_warps, SM, precision)

To achieve optimal performance, tune several interdependent compile-time parameters. These choices affect register pressure, occupancy, memory bandwidth utilization, and ultimately kernel latency. Experiment with different combinations on your target hardware to find the best configuration for your workload.

- **BLOCK_C**: The block size (tile) along the channel dimension. In this 1D reduction context, it functions like a **vectorization width**: it determines how many channel elements are loaded and processed simultaneously in a single iteration. Larger tiles (e.g., 64 or 128) improve memory coalescing but increase register usage. Typical values: 64 or 128, depending on `C` and the target GPU's register file capacity.

- **num_warps / num_stages**: Triton launch parameters that influence occupancy. `num_warps` determines the CUDA block size (`32 × num_warps` threads per block); `num_stages` controls software pipelining depth for memory operations. Lower values reduce register pressure but may underutilize the GPU.

- **SM (compute capability)**: When producing arch-specific Cubins with `ptxas`, target an architecture like `sm_89` or `sm_110`. Alternatively, rely on PTX JIT compilation at runtime—this trades first-run latency for flexibility across GPU generations. See [the NVIDIA blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/) for details on JIT behavior.

- **precision (dtype)**: Compile artifacts for the dtype you plan to run (e.g., FP16), and ensure the ONNX graph + TensorRT engine are built in a compatible precision mode (e.g., `trtexec --fp16` when appropriate).

For further reading on GPU kernel optimization and Triton performance tuning, see the [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) and NVIDIA's [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).

For authoritative background on PTX and the rationale for embedding PTX/Cubin to preserve forward compatibility across GPU generations, see NVIDIA's "Understanding PTX, the Assembly Language of CUDA GPU Computing" [developer blog](https://developer.nvidia.com/blog/understanding-ptx-the-assembly-language-of-cuda-gpu-computing/).

---
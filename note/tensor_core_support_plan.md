# Tensor Core対応計画

**ステータス**: 保留（CUDA対応環境が必要）

## 概要

NVIDIAのTensor Coreを活用して、行列積+加算（D = A × B + C）をハードウェアレベルで高速化する機能の実装計画。

## 背景

Tensor Coreは16x16や8x8のタイル単位で行列演算を1サイクルで実行できる専用ハードウェア。FP16/BF16/TF32などの低精度演算で特に効率的。

## 現状分析

| 項目 | 現状 | Tensor Core利用に必要 |
|------|------|---------------------|
| バックエンド | OpenCL, Metal, C | **CUDAバックエンド** |
| 精度型 | f32, f64のみ | FP16/BF16/TF32 |
| matmul実装 | `unsqueeze→mul→reduce`分解 | 専用GEMMカーネル |
| 融合パターン | ループ融合 | `D = A @ B + C`パターン認識 |

### 現アーキテクチャの強み

- **拡張性の高いRenderer trait**: 新バックエンド追加が容易
- **Suggesterベースの最適化**: パターン認識を追加しやすい
- **DeviceCapabilities**: ハードウェア機能検出の仕組みが既存

## 実装フェーズ

### Phase 1: CUDAバックエンド基盤

```
crates/backend-cuda/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── device.rs      # CudaDevice実装
    ├── buffer.rs      # CudaBuffer実装
    ├── kernel.rs      # CudaKernel実装
    ├── compiler.rs    # nvrtc経由のコンパイル
    └── renderer.rs    # CUDA Cレンダラー
```

**依存関係**: `cuda-sys` or `cudarc` クレート

```rust
pub struct CudaDevice {
    context: CUcontext,
    stream: CUstream,
}

impl Device for CudaDevice {
    type Buffer = CudaBuffer;
    type Kernel = CudaKernel;
    // ...
}
```

### Phase 2: 低精度型サポート

```rust
// DType拡張
pub enum DType {
    F16,   // IEEE 754 half precision
    BF16,  // Brain floating point
    TF32,  // TensorFloat-32 (NVIDIA)
    F32, F64,
    // ...
}

// TensorDType trait実装
impl TensorDType for half::f16 { ... }
impl TensorDType for half::bf16 { ... }
```

**依存関係**: `half` クレート

### Phase 3: Tensor Core検出パターン

```rust
/// matmul + bias add パターンを検出
pub struct TensorCoreFusionSuggester;

impl Suggester for TensorCoreFusionSuggester {
    fn suggest(&self, ast: &Ast) -> Vec<Suggestion> {
        // パターン: reduce_sum(mul(unsqueeze(A), unsqueeze(B))) + bias
        // → TensorCoreMatmulノードに変換
    }
}
```

### Phase 4: WMMA専用カーネル生成

```cuda
// 生成されるCUDAカーネル例
#include <mma.h>
using namespace nvcuda;

__global__ void gemm_tensor_core(
    half* __restrict__ A,
    half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K
) {
    // 16x16x16 タイルサイズ
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // タイル単位でロード・計算
    wmma::load_matrix_sync(a_frag, A + offset_a, lda);
    wmma::load_matrix_sync(b_frag, B + offset_b, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    wmma::store_matrix_sync(C + offset_c, c_frag, ldc, wmma::mem_row_major);
}
```

## 最適化パイプライン変更

```
現在:
  AST → ループ融合 → 並列化 → ベクトル化 → OpenCL/Metal

Tensor Core対応後:
  AST → パターン認識 ─┬→ Tensor Core対応 → WMMA専用カーネル (CUDA)
                      └→ その他 → 従来の最適化 → 汎用カーネル
```

## DeviceCapabilities拡張

```rust
pub enum DeviceFeature {
    HalfPrecision,
    TensorCore,        // 追加
    TensorCoreGen2,    // Ampere以降
    TensorCoreGen3,    // Hopper以降
}

pub struct DeviceCapabilities {
    pub compute_capability: (u32, u32),  // e.g., (8, 0) for Ampere
    pub tensor_core_size: Option<(usize, usize, usize)>,  // M, N, K
}
```

## 必要なハードウェア

- NVIDIA GPU (Volta以降: SM 7.0+)
  - Volta (V100): SM 7.0
  - Turing (RTX 20xx): SM 7.5
  - Ampere (RTX 30xx, A100): SM 8.0/8.6
  - Hopper (H100): SM 9.0

## 参考資料

- [CUDA WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma)
- [cuBLAS Tensor Core使用](https://docs.nvidia.com/cuda/cublas/)
- [cutlass - CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)

## 備考

- OpenCL/Metalには同等のAPIがないため、CUDA専用機能となる
- Intel AMX (Advanced Matrix Extensions) も同様のアプローチで対応可能
- Apple Silicon の AMX は非公開APIのため対応困難

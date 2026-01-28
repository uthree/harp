# 最適化

## 概要

Eclatは二段階の最適化を行う:

1. **グラフレベル**: 演算融合、冗長除去
2. **ASTレベル**: ループ変換、並列化、ベクトル化

## 探索戦略

### RuleBaseOptimizer

ルールベースの単一パス最適化。決定的で高速。

### BeamSearchOptimizer

ビーム幅を持つ幅優先探索。

```rust
BeamSearchOptimizer {
    beam_width: usize,      // ビーム幅
    max_iterations: usize,  // 最大イテレーション
}
```

### PrunedDfsOptimizer

枝刈り付き深さ優先探索。

### PrunedBfsOptimizer

枝刈り付き幅優先探索。

## サジェスター

変換候補を生成するコンポーネント。

### LoopFusionSuggester

隣接するループを融合。

```c
// Before
for (i = 0; i < N; i++) { a[i] = f(i); }
for (i = 0; i < N; i++) { b[i] = g(a[i]); }

// After
for (i = 0; i < N; i++) {
    a[i] = f(i);
    b[i] = g(a[i]);
}
```

### LoopTilingSuggester

キャッシュ効率のためのタイリング。

```c
// Before
for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
        ...

// After (tile size = T)
for (ii = 0; ii < N; ii += T)
    for (jj = 0; jj < M; jj += T)
        for (i = ii; i < min(ii+T, N); i++)
            for (j = jj; j < min(jj+T, M); j++)
                ...
```

### LoopInterchangeSuggester

ループ順序の交換。

```c
// Before (row-major access pattern)
for (j = 0; j < M; j++)
    for (i = 0; i < N; i++)
        a[i][j] = ...

// After (better cache locality)
for (i = 0; i < N; i++)
    for (j = 0; j < M; j++)
        a[i][j] = ...
```

### ParallelizationSuggester

並列化プラグマの挿入。

```c
// OpenMP
#pragma omp parallel for
for (i = 0; i < N; i++) { ... }

// GPU (group-level)
// Maps to workgroup/threadblock
```

並列化タイプ:
- `Local`: OpenMPローカル並列
- `Group`: GPUワークグループ
- `GPU`: GPUグローバル

### VectorizationSuggester

SIMD化。

```c
// Before
for (i = 0; i < N; i++)
    a[i] = b[i] + c[i];

// After (vector width = 4)
for (i = 0; i < N; i += 4)
    a[i:i+4] = b[i:i+4] + c[i:i+4];
```

### CseSuggester

共通部分式除去。

```c
// Before
x = a + b;
y = a + b;

// After
tmp = a + b;
x = tmp;
y = tmp;
```

### FunctionInliningSuggester

関数インライン展開。

### VariableExpansionSuggester

並列化のための変数展開。

```c
// Before (reduction)
sum = 0;
for (i = 0; i < N; i++)
    sum += a[i];

// After (parallel reduction)
sum[thread_id] = 0;
#pragma omp parallel for
for (i = 0; i < N; i++)
    sum[thread_id] += a[i];
// Final reduction...
```

### RuleBaseSuggester

代数的・論理的最適化ルール。

## 最適化ルール

### 代数的簡略化

| ルール | 変換 |
|--------|------|
| 加法単位元 | `x + 0 → x` |
| 乗法単位元 | `x * 1 → x` |
| 乗法零元 | `x * 0 → 0` |
| 二重否定 | `--x → x` |
| 自己減算 | `x - x → 0` |
| 自己除算 | `x / x → 1` |

### 定数畳み込み

コンパイル時に計算可能な式を評価。

```c
// Before
x = 2 + 3;

// After
x = 5;
```

### ビット演算最適化

| ルール | 変換 |
|--------|------|
| AND単位元 | `x & 0xFF..FF → x` |
| AND零元 | `x & 0 → 0` |
| OR零元 | `x \| 0 → x` |
| XOR零元 | `x ^ 0 → x` |
| 自己XOR | `x ^ x → 0` |

## コスト推定

### SimpleCostEstimator

```rust
cost = operation_count + memory_weight * memory_accesses
```

- 演算コスト: 各演算に重み付け
- メモリコスト: ロード/ストア回数

## 最適化履歴

変換のスナップショットを記録:

- 各ステップのAST状態
- 適用した変換の説明
- 代替候補とその説明

可視化ツール(`eclat-viz`)で探索過程を確認可能。

### WmmaSuggester

WMMA (Tensor Core) を使用した行列積最適化。3重ループの行列積パターンを検出し、`WmmaMatmul`ノードに変換する。

**検出パターン**:
```c
// Lowererが生成する3重ループ構造
for (ridx0 = 0; ridx0 < M; ridx0++) {        // M次元
    for (ridx2 = 0; ridx2 < N; ridx2++) {    // N次元
        acc = 0.0;
        for (ridx1 = 0; ridx1 < K; ridx1++) { // K次元 (reduce)
            acc += A[ridx0 * K + ridx1] * B[ridx1 * N + ridx2];
        }
        C[ridx0 * N + ridx2] = acc;
    }
}
```

**適用条件**:
- データ型: F16
- M, K, N が16の倍数
- Row-majorレイアウト、線形インデックス式

**変換後**:
```rust
AstNode::WmmaMatmul {
    a_ptr, a_offset, a_stride,  // 行列A
    b_ptr, b_offset, b_stride,  // 行列B
    c_ptr, c_offset, c_stride,  // 行列C (出力)
    m, k, n,                    // 次元サイズ
    dtype_ab: F16,              // 入力型
    dtype_c: F32,               // 累積型
}
```

**生成CUDAコード** (`CudaRenderer`):
```cuda
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

for (int kk = 0; kk < K; kk += 16) {
    wmma::load_matrix_sync(a_frag, &A[...], lda);
    wmma::load_matrix_sync(b_frag, &B[...], ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
}
wmma::store_matrix_sync(&C[...], c_frag, ldc, wmma::mem_row_major);
```

**対応GPU**: Volta以降 (Compute Capability 7.0+)

**生成Metalコード** (`MetalRenderer`):
```metal
simdgroup_half8x8 a_frag;
simdgroup_half8x8 b_frag;
simdgroup_float8x8 c_frag;

c_frag = simdgroup_float8x8(0);

for (uint kk = 0; kk < K; kk += 8) {
    simdgroup_load(a_frag, &A[...], lda);
    simdgroup_load(b_frag, &B[...], ldb);
    simdgroup_multiply_accumulate(c_frag, a_frag, b_frag, c_frag);
}
simdgroup_store(c_frag, &C[...], ldc);
```

**対応GPU**: Apple Silicon (M1以降)

**タイルサイズ**:
- CUDA (WMMA): 16x16x16
- Metal (simdgroup_matrix): 8x8

デバイスのタイルサイズは `DeviceProfile::matrix_tile_size` で取得可能。
`DeviceFeature::MatrixOperations` で対応状況を確認できる

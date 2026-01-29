# 最適化

## 概要

Eclatは二段階の最適化を行う:

1. **グラフレベル最適化**: パターン検出（MatMul等）と融合パス（ViewFusion、ElementwiseReduceFusion）を統合したビームサーチ最適化
2. **ASTレベル**: ループ変換、並列化、ベクトル化

## パイプライン

```
Tensor::realize()
    ↓
execute_graph() [backend/executor.rs]
    ↓
CompilationPipeline::lower_with_lowerer() [backend/compile.rs]
    ↓ (opt_level > 0 の場合)
GraphBeamSearchOptimizer [opt/graph/optimizer]
    ├── MatMulDetectorSuggester      (MatMulパターン検出)
    ├── ViewFusionSuggester          (連続View融合)
    └── ElementwiseReduceFusionSuggester (Elementwise+Reduce融合)
    ↓
Lowerer::lower() → AST
    ↓
CompilationPipeline::optimize() [BeamSearchOptimizer]
    ↓
Backend (レンダリング・コンパイル・実行)
```

## 最適化の統合

各融合パス（ViewFusion, ElementwiseReduceFusion）は直接`GraphSuggester`トレイトを実装する。これにより:

- **統一的な探索**: すべての変換がコスト評価に基づいて選択される
- **順序最適化**: 変換の適用順序も探索で決定
- **シンプルな実装**: アダプターなしで直接Suggesterとして動作

### CompilationPipelineの動作

- `opt_level > 0`: `GraphBeamSearchOptimizer`による統合最適化
- `opt_level = 0`: `AllFusions`のみ適用（高速・決定的）

### Lowererの動作

- `lower()`: 純粋なローワリングのみ（最適化なし）
- グラフ最適化は`CompilationPipeline`が担当

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

### SharedMemorySuggester

GPUの共有メモリを活用した最適化。GpuThread並列化されたループ内で、内側ループで再利用されるデータを共有メモリにプリロードする。

**検出パターン**:
```c
// タイリング後のループで複数スレッドがデータを再利用
for (tile = 0; tile < N/TILE; tile++) {      // GroupId
    for (local = 0; local < TILE; local++) { // GpuThread
        for (k = 0; k < K; k++) {             // 内側ループ
            acc += A[tile*TILE + local] * B[k];
            // A[tile*TILE + local] は k に対して不変 → 共有メモリ候補
        }
    }
}
```

**変換後**:
```c
__shared__ float shared_a[TILE];  // 共有メモリ確保

// 協調ロード
shared_a[local] = A[tile*TILE + local];
__syncthreads();  // バリア

for (k = 0; k < K; k++) {
    acc += shared_a[local] * B[k];  // 共有メモリからロード
}
```

**生成コード**:
- CUDA: `__shared__` 配列
- Metal: `threadgroup` 配列
- OpenCL: `__local` 配列

**適用条件**:
- GpuThread並列化されたループ内
- 内側ループで同一データが複数回アクセスされる
- 再利用回数が閾値以上（デフォルト4回）
- 共有メモリサイズ制限内（デフォルト48KB）

**コスト評価**:
`SimpleCostEstimator`は`SharedLoad`/`SharedStore`にグローバルメモリアクセスの約1/4のコストを割り当てる。

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

## グラフレベル最適化

グラフレベルでの最適化は、ASTへの変換前に高レベルなパターンを検出して最適化する。

### GraphBeamSearchOptimizer

ビームサーチを使用した探索。ASTレベルより候補が少ないため、より小さいビーム幅を使用。

```rust
GraphBeamSearchOptimizer {
    beam_width: 5,       // ビーム幅
    max_steps: 100,      // 最大ステップ数
    no_improvement_limit: Some(3),  // 改善がない場合の早期終了
}
```

### MatMulDetectorSuggester

行列積パターンを検出し、`GraphOp::MatMul`ノードに変換する。

**検出パターン**:
```
A.unsqueeze(-1).expand([M, K, N]) * B.unsqueeze(0).expand([M, K, N]) → sum(axis=K)
```

これはGraphレベルでは以下のように表現される:
- MapReduce { map: Mul(Wildcard(0), Wildcard(1)), reduce: Some(Sum, K_axis) }

**適用条件**:
- データ型: F16
- M, K, N が16の倍数

**変換後**:
```rust
GraphOp::MatMul {
    transpose: (bool, bool),           // (trans_a, trans_b)
    accumulator_dtype: Option<DType>,  // アキュムレータ型
}
```

### SimpleGraphCostEstimator

グラフのコスト推定。対数スケールで返却。

- FLOPs推定: 演算タイプに基づく
- メモリアクセスコスト: 読み書きのバランス
- MatMul効率係数: WMMA対応の場合は高効率

## WMMA/MatMul最適化

行列積は以下の流れで最適化される:

1. **Graph段階**: MatMulDetectorSuggesterがパターンを検出し、`GraphOp::MatMul`に変換
2. **Lowerer**: MatMulノードを処理
   - WMMA対応（F16 + 16の倍数）: `WmmaMatmul` ASTノードを生成
   - 非対応: 効率的な3重ループを生成
3. **Backend**: WmmaMatmulまたは3重ループをターゲットコードに変換

**WmmaMatmul生成条件**:
- データ型: F16
- M, K, N が16の倍数

**生成されるASTノード**:
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

**タイルサイズ**:
- CUDA (WMMA): 16x16x16
- Metal (simdgroup_matrix): 8x8

**コスト評価**:
`SimpleGraphCostEstimator`はMatMulノードにWMMA効率係数（約0.05）を適用し、
通常のMapReduceより大幅に低いコストを割り当てる。
`SimpleCostEstimator`（AST）はWmmaMatmulノードを同様に高効率として評価。

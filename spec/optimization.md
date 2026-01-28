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

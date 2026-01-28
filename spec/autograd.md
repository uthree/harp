# 自動微分

## 概要

Eclatは逆モード自動微分（バックプロパゲーション）をサポート。

## 基本的な使い方

```rust
use eclat::prelude::*;
use eclat::grad::backward;

// 勾配追跡を有効化
let x = Tensor::<D2, f32>::input([32, 64]);
x.requires_grad_(true);

// 順伝播
let y = (&x * &x).sum_all();

// 逆伝播
let grads = backward(&y, &[&x]);
let dx = grads.get(&x).unwrap();
```

## 勾配ルール (VJP)

Vector-Jacobian Product（VJP）として実装。

### 要素演算

| 演算 | 勾配 |
|------|------|
| `y = x + c` | `dx = dy` |
| `y = x * c` | `dx = dy * c` |
| `y = a + b` | `da = dy`, `db = dy` |
| `y = a * b` | `da = dy * b`, `db = dy * a` |
| `y = 1/x` | `dx = -dy / x^2` |
| `y = sqrt(x)` | `dx = dy / (2 * sqrt(x))` |
| `y = exp2(x)` | `dx = dy * y * ln(2)` |
| `y = log2(x)` | `dx = dy / (x * ln(2))` |
| `y = sin(x)` | `dx = dy * cos(x)` |
| `y = max(a, b)` | `da = dy * (a >= b)`, `db = dy * (a < b)` |
| `y = min(a, b)` | `da = dy * (a <= b)`, `db = dy * (a > b)` |

### リダクション

| 演算 | 勾配 |
|------|------|
| `y = sum(x, axis)` | `dx = broadcast(dy, axis)` |
| `y = max(x, axis)` | `dx = dy * (x == broadcast(y, axis))` |
| `y = min(x, axis)` | `dx = dy * (x == broadcast(y, axis))` |
| `y = prod(x, axis)` | `dx = dy * y / x` (ゼロ要素なしの場合) |

### ビュー演算

| 演算 | 勾配 |
|------|------|
| `y = reshape(x, shape)` | `dx = reshape(dy, x.shape)` |
| `y = permute(x, order)` | `dx = permute(dy, inverse(order))` |
| `y = expand(x, shape)` | `dx = sum(dy, expanded_axes)` |
| `y = slice(x, ranges)` | `dx = pad_zeros(dy, ...)` |

### 累積演算

| 演算 | 勾配 |
|------|------|
| `y = cumsum(x)` | `dx = reverse_cumsum(dy)` |
| `y = cumprod(x)` | `dx = reverse_cumsum(dy * y) / x` |
| `y = cummax(x)` | 複雑（マスク付き伝播） |

## 実装詳細

### GradContext

勾配計算のコンテキスト。

```rust
pub struct GradContext {
    grad_map: HashMap<GraphNodeId, GraphNode>,
    visited: HashSet<GraphNodeId>,
}
```

### 勾配グラフ構築

1. 出力ノードから逆順にトポロジカルソート
2. 各ノードに対してVJPルールを適用
3. 複数の経路からの勾配を加算

```
Forward:  x → f → g → y
Backward: dx ← df/dx ← dg/df ← dy
```

### 勾配の蓄積

同じ入力が複数箇所で使われる場合、勾配を加算:

```rust
// y = x + x  →  dx = dy + dy = 2 * dy
// y = x * x  →  dx = dy * x + dy * x = 2 * dy * x
```

## サポートされない演算

以下の演算は現在勾配をサポートしていない:

- `Scatter` (scatter-add)
- `AtomicAdd`, `AtomicMax`
- 一部のビット演算

これらの演算に対しては、ゼロ勾配が返される。

## 勾配チェックポイント

大規模モデルでのメモリ節約のため、中間結果を再計算する機能（将来実装予定）。

## 高階微分

現在はサポートされていないが、勾配グラフも通常のグラフなので原理的には可能。

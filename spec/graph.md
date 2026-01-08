# Graph モジュール仕様書

計算グラフを表現するための`src/graph/`モジュールの仕様。

## 概要

Graphモジュールは高レベルなテンソル計算を表現する計算グラフ（DAG）を提供します。
各ノード（`GraphNode`）は1つの演算または入力バッファを表し、
`Lowerer`によって実行可能なAST（`AstNode`）に変換されます。

## ディレクトリ構成

```
src/graph/
├── mod.rs          # モジュール定義、re-exports
├── node.rs         # GraphNode, GraphInner, GraphOp, ReduceOp
├── ops.rs          # 演算子オーバーロード（Add, Mul, sqrt等）
├── builder.rs      # input(), constant()等のビルダー関数
├── traversal.rs    # グラフ走査ユーティリティ
├── shape/          # 形状式（Expr, View）
│   ├── mod.rs
│   ├── expr.rs
│   └── view.rs
└── tests.rs        # 統合テスト
```

## 主要な型

### GraphNode

計算グラフの基本ノード。`Rc<GraphInner>`をラップしDAG構造を実現。

```rust
pub struct GraphNode(pub Rc<GraphInner>);

pub struct GraphInner {
    pub src: Vec<GraphNode>,      // 入力ノード（複数可）
    pub view: View,               // 出力のビュー
    pub op: GraphOp,              // 演算の種類
    pub dtype: DType,             // データ型
    pub name: Option<String>,     // デバッグ用名前
    pub buffer_id: Option<usize>, // 外部バッファ参照（入力ノード用）
}
```

### GraphOp

グラフノードが実行する演算の種類。

```rust
pub enum GraphOp {
    /// ビュー変換のみ（データ変更なし）
    View(View),

    /// Map-Reduce演算
    MapReduce {
        map: AstNode,                      // 要素ごとの演算
        reduce: Option<(ReduceOp, usize)>, // 縮約（演算, 軸）
    },
}
```

### ReduceOp

縮約演算の種類。

```rust
pub enum ReduceOp {
    Sum,   // 総和
    Max,   // 最大値
    Min,   // 最小値
    Prod,  // 総乗
}
```

## 入力の参照方法

`MapReduce`の`map`フィールドでは、入力ノードを`AstNode::Wildcard`で参照します。

- `Wildcard("0")`: 最初の入力（src[0]）
- `Wildcard("1")`: 2番目の入力（src[1]）
- ...

Lowering時に`Wildcard`は対応する`Load`演算に置き換えられます。

## ビルダー関数

```rust
// 入力テンソル
let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);

// 名前付き入力
let y = named_input("weights", vec![Expr::Const(32), Expr::Const(64)], DType::F32);

// 定数テンソル
let c = constant(vec![Expr::Const(32)], DType::F32);

// 動的形状の入力
let d = dynamic_input(3, DType::F32); // 3次元、サイズは実行時決定
```

## 演算子オーバーロード

`GraphNode`に対して標準的な演算子が実装されています。

```rust
let a = input(vec![Expr::Const(32)], DType::F32);
let b = input(vec![Expr::Const(32)], DType::F32);

let c = &a + &b;   // 加算
let d = &a * &b;   // 乗算
let e = &a / &b;   // 除算
let f = &a - &b;   // 減算
let g = -&a;       // 否定
```

## メソッド

### ビュー変換

```rust
x.permute(&[1, 0])              // 軸入れ替え
x.unsqueeze(0)                  // 次元追加（サイズ1）
x.squeeze(1)                    // 次元削除（サイズ1のみ）
x.expand(1, Expr::Const(64))    // 次元拡張（サイズ1→指定サイズ、ブロードキャスト）
x.reshape(new_shape)            // 形状変更
x.flip(0)                       // 軸反転
x.repeat(0, Expr::Const(2))     // 繰り返し
```

### 縮約

```rust
x.sum(axis)   // 総和
x.max(axis)   // 最大値
x.min(axis)   // 最小値
x.prod(axis)  // 総乗
```

### 数学関数

```rust
x.sqrt()    // 平方根
x.recip()   // 逆数
x.log2()    // log₂
x.exp2()    // 2^x
x.sin()     // sin
x.cos()     // cos
x.floor()   // 切り捨て
x.ln()      // 自然対数
x.exp()     // e^x
x.abs()     // 絶対値
```

### 比較・論理

```rust
x.lt(&y)         // <
x.gt(&y)         // >
x.le(&y)         // <=
x.ge(&y)         // >=
x.eq_node(&y)    // ==
x.ne_node(&y)    // !=
x.logical_and(&y)
x.logical_or(&y)
x.logical_not()
```

### 条件演算

```rust
x.where_cond(&cond, &y)   // cond ? x : y
x.maximum(&y)             // max(x, y)
x.minimum(&y)             // min(x, y)
x.clamp(&min, &max)       // clamp(x, min, max)
```

## グラフ走査

```rust
// トポロジカルソート（入力→出力順）
let sorted = topological_sort(&[output_node]);

// 入力ノードの収集
let inputs = collect_inputs(&[output_node]);

// ノード数カウント
let count = count_nodes(&[output_node]);

// サイクル検出
let has_cycle = has_cycle(&[output_node]);

// 共通部分式の検出
let common = find_common_subexpressions(&[output_node]);
```

## グラフ変換

`GraphTransform`トレイトでグラフの変換を実装できます。

```rust
impl GraphTransform for MyTransform {
    fn transform(&self, node: &GraphNode) -> Option<GraphNode> {
        // 変換を適用（Noneで元のノードを維持）
    }
}

let transformed = my_transform.apply(&[root]);
```

## 使用例

```rust
use harp::graph::{input, Expr, DType, topological_sort};

// 計算グラフ構築
let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");

// (a + b) * a を計算し、軸1で総和
let c = (&(&a + &b) * &a).sum(1).with_name("output");

// グラフをトポロジカル順に取得
let nodes = topological_sort(&[c]);
```

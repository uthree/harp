# 計算グラフ
テンソル（多次元配列）単位での演算をDAGで表現する。

## ファイル構成
- `src/graph/mod.rs` - Graph、GraphNode、GraphOp、AxisStrategyの定義
- `src/graph/ops.rs` - 演算子オーバーロードとヘルパー関数
- `src/graph/shape/mod.rs` - shapeモジュール定義
- `src/graph/shape/expr.rs` - Shape式（Expr）の実装 (280行)
- `src/graph/shape/tests.rs` - Shape式のテストコード
- `src/graph/shape/view.rs` - View実装

## 概要

計算グラフはテンソル演算をDAG（有向非巡回グラフ）として表現します。各ノードは演算または入力データを表し、エッジはデータの流れを表します。

## データ構造

### Graph
計算グラフ全体を表す構造体。

- **inputs**: 入力ノードの名前とWeak参照のマップ（循環参照を防ぐため）
- **outputs**: 出力ノードの名前とノードのマップ

```rust
pub struct Graph {
    inputs: HashMap<String, Weak<GraphNodeData>>,
    outputs: HashMap<String, GraphNode>,
}
```

### GraphNode
計算グラフのノード。内部的には`Rc<GraphNodeData>`でラップされている。

```rust
pub struct GraphNode(Rc<GraphNodeData>);

pub struct GraphNodeData {
    pub dtype: DType,      // データ型
    pub op: GraphOp,       // 演算の種類（並列化戦略を含む）
    pub src: Vec<GraphNode>, // 入力ノード
    pub view: View,        // テンソルのView
}
```

### DType
グラフレベルでのデータ型。ASTのDTypeとは異なり、VecやPtrは扱わない。

- **Unknown**: 型が未定または推論中
- **F32**: 32bit浮動小数点数

### AxisStrategy
各軸の並列化戦略を表すenum。lowering時の実装方法を制御します。

```rust
pub enum AxisStrategy {
    Sequential { simd_width: usize },   // 逐次実行
    Thread { simd_width: usize },       // スレッドで並列化
    ThreadGroup { simd_width: usize },  // スレッドグループ/ブロック
}
```

- **Sequential { simd_width }**: 逐次実行
  - `simd_width: 1` - SIMDベクトル化なし（デフォルト）
  - `simd_width: n (n>=2)` - SIMD幅nでベクトル化
- **Thread { simd_width }**: スレッドレベルで並列化
  - `simd_width: 1` - 各スレッド内でSIMDなし
  - `simd_width: n (n>=2)` - 各スレッド内でSIMD幅nでベクトル化
- **ThreadGroup { simd_width }**: スレッドグループ/ブロックレベルで並列化（GPU向け）
  - `simd_width: 1` - 各スレッドグループ内でSIMDなし
  - `simd_width: n (n>=2)` - 各スレッドグループ内でSIMD幅nでベクトル化

**設計変更点（v2）:**
- **Autoの除去**: 明示的な戦略指定を必須化。デフォルトは`Sequential { simd_width: 1 }`
- **SIMD幅の明示化**: `Option<usize>`から`usize`に変更。SIMD化しない場合は`1`で表現

この設計により、並列化とSIMDベクトル化を独立して制御可能です。

### ヘルパーメソッド
簡潔に戦略を作成するためのヘルパーメソッドを提供：

```rust
AxisStrategy::sequential()          // Sequential { simd_width: 1 }
AxisStrategy::sequential_simd(4)    // Sequential { simd_width: 4 }
AxisStrategy::thread()              // Thread { simd_width: 1 }
AxisStrategy::thread_simd(8)        // Thread { simd_width: 8 }
AxisStrategy::thread_group()        // ThreadGroup { simd_width: 1 }
AxisStrategy::thread_group_simd(4)  // ThreadGroup { simd_width: 4 }
```

## GraphOpについて
GraphOpは最適化の段階で最終的に融合されるので、最適化よりも演算子の種類を減らすことを重視する。そのため、Add, Negを組み合わせて減算を表現するようなことがある。

並列化戦略（axis_strategies）は各GraphOpバリアントの一部として保持されます。これにより、演算の種類と並列化戦略が密接に関連付けられます。

```rust
pub enum GraphOp {
    Input,
    Const(Literal),
    View(View),
    Contiguous {
        axis_strategies: Option<Vec<AxisStrategy>>,
    },
    Elementwise {
        op: ElementwiseOp,
        axis_strategies: Option<Vec<AxisStrategy>>,
    },
    Reduce {
        op: ReduceOp,
        axis: usize,
        axis_strategies: Option<Vec<AxisStrategy>>,
    },
    Cumulative {
        axis_strategies: Option<Vec<AxisStrategy>>,
    },
}

pub enum ReduceOp {
    Add, // 合計
    Mul, // 積
    Max, // 最大値
}
```

### 演算の種類

- **Input**: 入力ノード（並列化戦略なし）
- **Const(Literal)**: 定数ノード（スカラー、並列化戦略なし）
- **View(View)**: Viewを変更する（並列化戦略なし）
- **Contiguous**: Viewに従って要素を並べ直す（並列化戦略あり）
- **Elementwise**: 要素ごとに演算を行う（並列化戦略あり）
  - Add, Mul, Max, Rem, Idiv, Neg, Recip
- **Reduce**: 縮約（並列化戦略あり）
  - op: ReduceOp（Add, Mul, Max）
  - axis: 縮約する軸のインデックス
- **Cumulative**: 累積（未実装、並列化戦略あり）

## View
Viewは、各軸の添え字からメモリオフセットに変換する写像を表現します。
この仕組みにより、ほぼゼロコストのView操作によって転置などが実現可能です。

### 現在の実装
- **Linear**: 線形変換で表現可能なView
  - shape: 論理的なテンソルのサイズ
  - strides: 各次元の添え字の係数
  - offset: オフセット

### View操作
- `contiguous()`: 連続したメモリレイアウトのViewを作成
- `permute()`: 軸の順序を変更
- `unsqueeze()`: 新しい次元を追加（サイズ1）
- `squeeze()`: サイズ1の次元を削除
- `flip()`: 指定した軸を反転
- `expand()`: サイズ1の次元を拡張（strideを0にする）
- `is_contiguous()`: 連続したメモリレイアウトかどうかを判定

## ノードの作成

### 入力ノードの作成
InputNodeBuilderパターンを使用して入力ノードを作成します。

```rust
let mut graph = Graph::new();
let input = graph.input("x")
    .with_dtype(DType::F32)
    .with_shape(vec![10, 20])
    .build();
```

- `with_dtype()`: データ型を指定（省略可、デフォルトはUnknown）
- `with_shape()`: 形状を指定（省略可、デフォルトは空=スカラー）
- `build()`: GraphNodeを作成してGraphに登録

### 出力ノードの登録
```rust
graph.output("y", result_node);
```

## 演算

### 演算子のオーバーロード
GraphNodeに対して、標準的な演算子が実装されています（`src/graph/ops.rs`）。

- `+` (Add): 要素ごとの加算
- `*` (Mul): 要素ごとの乗算
- `-` (Neg): 符号反転
- `%` (Rem): 剰余
- `/` (Div): 整数除算

```rust
let result = a + b * c;
let neg_result = -result;
```

### ヘルパー関数

**要素ごとの演算:**
- `ops::recip(node)`: 逆数
- `ops::max(a, b)`: 要素ごとの最大値

**縮約演算:**
- `ops::reduce(node, op, axis)`: 指定軸を指定演算で縮約（汎用）
- `ops::reduce_sum(node, axis)`: 指定軸の合計
- `ops::reduce_mul(node, axis)`: 指定軸の積
- `ops::reduce_max(node, axis)`: 指定軸の最大値

**GraphNodeのメソッド:**
- `node.reduce(op, axis)`: 指定軸を指定演算で縮約
- `node.reduce_sum(axis)`: 指定軸の合計
- `node.reduce_mul(axis)`: 指定軸の積
- `node.reduce_max(axis)`: 指定軸の最大値

### DType推論
演算時に自動的にDTypeが推論されます。

- 両方が同じDType → そのDType
- 片方がUnknown → もう片方のDType
- 異なるDType → Unknown

### View推論（重要な設計方針）

**明示的なshape変換のみを許可します。**

演算を行う2つのノードは、**完全に同じshape**である必要があります。異なるshapeの場合、実行時にpanicします。

```rust
// OK: 同じshape
let a = graph.input("a").with_shape(vec![10, 20]).build();
let b = graph.input("b").with_shape(vec![10, 20]).build();
let result = a + b; // OK

// NG: 異なるshape
let scalar = graph.input("scalar").build(); // shape = []
let tensor = graph.input("tensor").with_shape(vec![10, 20]).build();
let result = scalar + tensor; // panic!
```

この設計により：
- **明示性**: shape変換は全て明示的に行う必要がある
- **安全性**: 意図しないshape変換によるバグを防止
- **将来の拡張性**: `expand()`, `broadcast_to()`などの明示的な関数を追加しやすい

## 使用例

### 基本的な演算

```rust
// グラフの作成
let mut graph = Graph::new();

// 入力ノードの作成
let a = graph.input("a")
    .with_dtype(DType::F32)
    .with_shape(vec![10, 20])
    .build();

let b = graph.input("b")
    .with_dtype(DType::F32)
    .with_shape(vec![10, 20])
    .build();

// 演算
let sum = a.clone() + b.clone();
let product = a * b;
let result = sum * ops::recip(product);

// 出力ノードの登録
graph.output("result", result);
```

### Reduce演算の例

```rust
use harp::graph::{Graph, DType, ReduceOp, ops};

let mut graph = Graph::new();

// 3次元テンソル（バッチ x 高さ x 幅）
let input = graph.input("image")
    .with_dtype(DType::F32)
    .with_shape(vec![32, 224, 224])  // 32枚の224x224画像
    .build();

// 軸2（幅）方向の合計: (32, 224, 224) -> (32, 224)
let sum_width = input.reduce_sum(2);

// 軸1（高さ）方向の最大値: (32, 224, 224) -> (32, 224)
let max_height = input.reduce_max(1);

// 全ピクセルの積（複数の軸を順次縮約）
let prod_all = input.reduce_mul(2).reduce_mul(1).reduce_mul(0);  // -> スカラー

// 汎用reduceを使う例
let custom_reduce = input.reduce(ReduceOp::Add, 0);  // バッチ方向の合計

graph.output("sum_width", sum_width);
graph.output("max_height", max_height);
graph.output("prod_all", prod_all);
```

## Lowering戦略

### axis_strategies
各GraphOpのバリアント（Contiguous, Elementwise, Reduce, Cumulative）は`axis_strategies: Option<Vec<AxisStrategy>>`フィールドを持ち、各軸の並列化戦略を指定できます。

- **None**: 全ての軸がデフォルト値`Sequential { simd_width: 1 }`（逐次実行、SIMD化なし）
- **Some(vec)**: ベクトルの長さは`view.ndim()`と一致する必要がある。各要素で軸ごとの戦略を明示的に指定

### 演算ノードの作成

演算子オーバーロードを使用する場合、デフォルトでは`axis_strategies: None`（全軸Sequential { simd_width: 1 }）になります：

```rust
// デフォルト（全軸Sequential、SIMD化なし）
let result = a + b; // GraphOp::Elementwise { op: Add, axis_strategies: None }
```

明示的に並列化戦略を指定する場合は、GraphNodeを直接作成します：

```rust
use std::rc::Rc;

// 戦略を明示的に指定
let node = GraphNode(Rc::new(GraphNodeData {
    dtype: DType::F32,
    op: GraphOp::Elementwise {
        op: ElementwiseOp::Add,
        axis_strategies: Some(vec![
            AxisStrategy::ThreadGroup { simd_width: 4 }, // 軸0: スレッドグループ、SIMD幅4
            AxisStrategy::Thread { simd_width: 1 },      // 軸1: スレッド並列化のみ（SIMD化なし）
            AxisStrategy::Sequential { simd_width: 8 },  // 軸2: 逐次実行、SIMD幅8
        ]),
    },
    src: vec![a, b],
    view: View::contiguous(vec![10, 20, 30]),
}));

// ヘルパーメソッドを使用した例
let node = GraphNode(Rc::new(GraphNodeData {
    dtype: DType::F32,
    op: GraphOp::Elementwise {
        op: ElementwiseOp::Add,
        axis_strategies: Some(vec![
            AxisStrategy::thread_group_simd(4), // 軸0
            AxisStrategy::thread(),              // 軸1
            AxisStrategy::sequential_simd(8),    // 軸2
        ]),
    },
    src: vec![a, b],
    view: View::contiguous(vec![10, 20, 30]),
}));
```

### 設計方針
- **明示的な戦略指定**: Autoを廃止し、全ての戦略を明示的に指定。デフォルトは`Sequential { simd_width: 1 }`
- **演算と戦略の一体化**: axis_strategiesをGraphOpの一部にすることで、演算の種類と並列化戦略が密接に関連付けられる
- **並列化とSIMDの独立制御**: 各戦略バリアントが`simd_width`フィールドを持つため、並列化方式（Sequential/Thread/ThreadGroup）とSIMDベクトル化を独立して制御可能
- **SIMD幅の明示化**: `simd_width: 1`はSIMD化なし、`simd_width >= 2`はSIMD化ありを意味する
- **柔軟な組み合わせ**: 例えば「スレッドで並列化しつつ、各スレッド内でSIMD幅4でベクトル化」= `Thread { simd_width: 4 }`
- **View操作との連携**: View操作を組み合わせることで、新たな軸を追加してアンローリング的なことが可能
  - その責務はグラフ最適化処理(`opt/graph`)に分離
- **将来の拡張性**: WMMA(Tensor Core)対応などの高度な機能は将来的に追加予定
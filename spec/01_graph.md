# Graph モジュール仕様

## 概要

Graphモジュールは計算グラフの表現を提供します。テンソル演算を有向非巡回グラフ（DAG）として表現し、各ノードが演算、各エッジがデータの流れを表します。

## 主要な型

### GraphNode

計算グラフのノードを表す型。`Rc<GraphNodeData>`のラッパー。

```rust
pub struct GraphNode(Rc<GraphNodeData>);

pub struct GraphNodeData {
    pub op: GraphOp,
    pub dtype: DType,
    pub view: View,
}
```

**特徴:**
- 参照カウント方式（Rc）により効率的な共有
- ポインタアドレスベースの等価性判定（同一ノードの判定）
- 強参照カウントによる分岐検出

**主要メソッド:**
- `cast(target_dtype: DType)`: 型変換
- `input_nodes()`: 入力ノードのリストを取得
- `strong_count()`: 強参照カウント（分岐検出に使用）

### GraphOp

グラフノードが表す演算の種類。

```rust
pub enum GraphOp {
    // 基本演算
    Input(usize),                    // 入力（インデックス付き）
    Const(ConstLiteral),             // 定数
    Elementwise(ElementwiseOp),      // 要素ごとの演算
    Reduce(ReduceOp, usize, GraphNode),  // 軸縮約
    Cumulative(CumulativeOp, usize, GraphNode),  // 累積演算

    // メモリレイアウト操作
    View(GraphNode),                 // ビュー変更（コピーなし）
    Contiguous(GraphNode),           // 連続メモリへの変換
    Cast(GraphNode, DType),          // 型変換

    // 特殊演算
    Fold(usize, usize, usize, usize, GraphNode),  // col2im操作

    // 融合済み演算（最適化後）
    FusedElementwise(AstNode, Vec<GraphNode>),
    FusedReduce(ReduceOp, Vec<usize>, GraphNode),
    FusedElementwiseReduce(AstNode, Vec<GraphNode>, ReduceOp, Vec<usize>),
    FusedElementwiseCumulative(AstNode, Vec<GraphNode>, CumulativeOp),
}
```

### Graph

計算グラフ全体を表す型。

```rust
pub struct Graph {
    pub inputs: Vec<Weak<GraphNodeData>>,
    pub outputs: Vec<GraphNode>,
    pub shape_variables: Vec<ShapeVariableSignature>,
}
```

**役割:**
- 入力ノードの管理（Weak参照で循環参照を防止）
- 出力ノードの管理
- シェイプ変数の定義

**主要メソッド:**
- `input(dtype: DType, shape: Vec<ShapeExpr>)`: 入力ノードを作成
- `output(node: GraphNode)`: 出力ノードを登録
- `shape_var(var_name: &str, default: isize)`: シェイプ変数を定義

## 演算の種類

### ElementwiseOp（要素単位演算）

各要素に独立して適用される演算。

```rust
pub enum ElementwiseOp {
    // 二項演算
    Add(GraphNode, GraphNode),
    Mul(GraphNode, GraphNode),
    Max(GraphNode, GraphNode),
    Mod(GraphNode, GraphNode),
    LessThan(GraphNode, GraphNode),
    Eq(GraphNode, GraphNode),

    // 単項演算
    Neg(GraphNode),
    Recip(GraphNode),
    Sin(GraphNode),
    Sqrt(GraphNode),
    Log2(GraphNode),
    Exp2(GraphNode),

    // 条件選択
    Select(GraphNode, GraphNode, GraphNode),
}
```

### ReduceOp（縮約演算）

特定の軸に沿って値を集約する演算。

```rust
pub enum ReduceOp {
    Sum,  // 総和
    Max,  // 最大値
}
```

### CumulativeOp（累積演算）

特定の軸に沿って累積的に演算を適用。

```rust
pub enum CumulativeOp {
    Sum,  // 累積和
}
```

## シェイプとビュー

### ShapeExpr

シェイプの各次元を表す式。

```rust
pub enum Expr {
    Const(isize),           // 定数
    Var(String),            // 変数
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    // ... その他の演算
}
```

**用途:**
- 動的なシェイプの表現
- シェイプ変数によるパラメータ化

### View

メモリレイアウトの情報。

```rust
pub enum View {
    Linear {
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        offset: Expr,
    }
}
```

**ビュー変換操作:**
- `unsqueeze(dim: usize)`: 次元を追加（サイズ1）
- `squeeze(dim: usize)`: 次元を削除（サイズ1のみ）
- `permute(dims: Vec<usize>)`: 次元の順序を変更
- `expand(shape: Vec<Expr>)`: サイズ1の次元をブロードキャスト
- `reshape(shape: Vec<Expr>)`: シェイプを変更

## ビュー操作とContiguous操作の違い

### View操作
- メモリコピーなし
- ストライドとオフセットのみ変更
- 元のメモリを参照

### Contiguous操作
- メモリコピーあり
- ストライドを連続メモリ用に調整
- 新しいメモリ領域を確保

## グラフの構築例

```rust
let mut graph = Graph::new();

// 入力を定義
let a = graph.input(DType::F32, vec![2.into(), 3.into()]);
let b = graph.input(DType::F32, vec![2.into(), 3.into()]);

// 演算を適用
let c = a + b;
let d = c * 2.0;

// 出力を登録
graph.output(d);
```

## シグネチャ

グラフの入出力の型情報。

```rust
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>,
    pub inputs: Vec<ArraySignature>,
    pub outputs: Vec<ArraySignature>,
}

pub struct ArraySignature {
    pub dtype: DType,
    pub shape: Vec<ShapeExpr>,
}
```

**用途:**
- コンパイル時の型チェック
- カーネル呼び出し時の引数検証

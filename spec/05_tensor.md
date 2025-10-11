# Tensor モジュール仕様

## 概要

Tensorモジュールはユーザー向けの高レベルAPIを提供します。型安全な多次元配列操作と自動微分機能を備えています。

## 主要な型

### TensorType トレイト

Rust型からDTypeへのマッピング。

```rust
pub trait TensorType: 'static {
    const DTYPE: DType;
}
```

**実装:**
- `f32` → `DType::F32`
- `isize` → `DType::Isize`
- `usize` → `DType::Usize`

### Dimension トレイト

テンソルの次元数を表現。

```rust
pub trait Dimension: Clone {
    const NDIM: Option<usize>;
    fn check_shape(shape: &[usize]) -> bool;
}
```

**提供される次元型:**
- `Dyn`: 動的次元（実行時に決定）
- `D0`: 0次元（スカラー）
- `D1`: 1次元（ベクトル）
- `D2`: 2次元（行列）
- `D3`: 3次元
- `D4`: 4次元

### Tensor<T, D>

型パラメータ化されたテンソル。

```rust
pub struct Tensor<T: TensorType, D: Dimension = Dyn> {
    inner: TensorBase,
    _phantom: PhantomData<(T, D)>,
}
```

**型パラメータ:**
- `T`: 要素型（f32, isize, usize）
- `D`: 次元型（Dyn, D0, D1, ...）

**型エイリアス:**
```rust
pub type Tensor0<T = f32> = Tensor<T, D0>;  // スカラー
pub type Tensor1<T = f32> = Tensor<T, D1>;  // ベクトル
pub type Tensor2<T = f32> = Tensor<T, D2>;  // 行列
pub type Tensor3<T = f32> = Tensor<T, D3>;
pub type Tensor4<T = f32> = Tensor<T, D4>;
pub type TensorDyn<T = f32> = Tensor<T, Dyn>;  // 動的次元
```

### TensorBase

実際のデータとメタデータを保持する基底クラス。

```rust
pub struct TensorBase {
    id: TensorId,
    data: Option<TensorData>,
    graph_node: Option<GraphNode>,
    graph: Option<Graph>,
    backend_name: String,
    dtype: DType,
    shape: Vec<usize>,
    requires_grad: bool,
    grad: Option<Box<TensorBase>>,
    autograd_meta: Option<TensorMeta>,
}
```

## テンソルの作成

### from_vec

Vecからテンソルを作成。

```rust
let data = vec![1.0, 2.0, 3.0, 4.0];
let tensor: Tensor2<f32> = Tensor::from_vec(data, &[2, 2], "c");
```

**検証:**
- データ長とシェイプの整合性チェック
- 次元型との整合性チェック
- データ型の一致チェック

### from_graph_node

計算グラフノードからテンソルを作成（内部用）。

```rust
let tensor = Tensor::from_graph_node(graph_node, graph, backend_name);
```

## テンソルのプロパティ

### メタデータ取得

```rust
tensor.shape()        // &[usize]: シェイプ
tensor.dtype()        // &DType: データ型
tensor.ndim()         // usize: 次元数
tensor.numel()        // usize: 要素数
tensor.id()           // usize: 一意なID
```

## 勾配管理

### 勾配の有効化/無効化

```rust
let x = tensor.enable_grad();   // 勾配計算を有効化
let y = tensor.disable_grad();  // 勾配計算を無効化

if tensor.is_requires_grad() {
    // 勾配計算が有効
}
```

### 勾配の取得と設定

```rust
// 勾配を取得
if let Some(grad) = tensor.grad() {
    println!("Gradient: {:?}", grad);
}

// 勾配を設定
tensor.set_grad(grad_tensor);

// 勾配をゼロクリア
tensor.zero_grad();
```

### 逆伝播

```rust
let mut loss = forward_pass();
loss.backward();  // 勾配を計算

// 各パラメータの勾配を取得
if let Some(grad) = param.grad() {
    // 勾配を使ってパラメータを更新
}
```

### detach

計算グラフから切り離す。

```rust
let x = tensor.enable_grad();
let y = (x.clone() * 2.0).detach();  // yは勾配追跡しない
```

## 演算

テンソル演算は`src/tensor/ops.rs`で定義されています。

### 要素単位演算

二項演算子のオーバーロード:
```rust
let c = &a + &b;  // 加算
let d = &a * &b;  // 乗算
let e = -&a;      // 符号反転
```

### ビュー変換

```rust
// 次元の追加/削除
let unsqueezed = tensor.unsqueeze(1);
let squeezed = tensor.squeeze(1);

// 次元の順序変更
let permuted = tensor.permute(vec![1, 0]);

// ブロードキャスト
let expanded = tensor.expand(vec![2, 5, 3]);

// リシェイプ
let reshaped = tensor.reshape(vec![4, 2]);
```

### 縮約演算

```rust
let sum = tensor.sum(axis);   // 合計
let max = tensor.max(axis);   // 最大値
```

### 累積演算

```rust
let cumsum = tensor.cumsum(axis);  // 累積和
```

## データストレージ

### TensorData

実際のデータを保持する内部型。

```rust
enum TensorData {
    F32(Vec<f32>),
    Isize(Vec<isize>),
    Usize(Vec<usize>),
}
```

**特徴:**
- 評価済みテンソルのみデータを保持
- 遅延評価テンソルは`None`

## 自動微分

### TensorId

テンソルの一意な識別子。

```rust
pub type TensorId = usize;
```

**用途:**
- 計算グラフ内でのテンソル識別
- 勾配の対応付け

### TensorMeta

自動微分のメタデータ。

```rust
pub struct TensorMeta {
    pub graph_node: GraphNode,
    pub inputs: Vec<(TensorId, GraphNode)>,
    pub grad_fn: Option<GradFn>,
}
```

### GradFn

勾配計算関数。

```rust
pub trait GradFn {
    fn backward(
        &self,
        grad_output: GraphNode,
        inputs: &[GraphNode],
    ) -> Vec<Option<GraphNode>>;
}
```

**実装例:**
- AddBackward: 加算の逆伝播
- MulBackward: 乗算の逆伝播
- NegBackward: 符号反転の逆伝播

## 使用例

### 基本的な演算

```rust
use harp::tensor::{Tensor, Tensor2};

let a: Tensor2<f32> = Tensor::from_vec(
    vec![1.0, 2.0, 3.0, 4.0],
    &[2, 2],
    "c"
);

let b: Tensor2<f32> = Tensor::from_vec(
    vec![5.0, 6.0, 7.0, 8.0],
    &[2, 2],
    "c"
);

let c = &a + &b;
```

### 自動微分

```rust
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c")
    .enable_grad();

let y = &x * 2.0;
let mut z = y.sum(0);

z.backward();

if let Some(grad) = x.grad() {
    // grad = [2.0, 2.0, 2.0]
}
```

## 型安全性

### コンパイル時のチェック

```rust
// OK: 1次元テンソル
let vec: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0], &[2], "c");

// コンパイルエラー: 1次元テンソルに2次元シェイプ
let vec: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0], &[1, 2], "c");
```

### 動的次元での柔軟性

```rust
// 動的次元は任意のシェイプを受け入れる
let dyn_tensor: TensorDyn<f32> = Tensor::from_vec(vec![1.0], &[1], "c");
let dyn_tensor: TensorDyn<f32> = Tensor::from_vec(vec![1.0], &[1, 1], "c");
```

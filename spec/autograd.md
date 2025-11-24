# 自動微分 (Autograd)

PyTorchライクな逆向きモード自動微分を提供します。

## 設計思想

### グラフからグラフへの変換

自動微分は**計算グラフから勾配計算グラフへの変換**として実装されています。既存のコンパイラパイプライン（Lowerer、Optimizer、Backend）には一切手を加えず、独立したレイヤーとして機能します。

```
[Autograd Layer]
  Tensor API (requires_grad管理、backward実行)
  ↓
[Graph Layer]
  GraphNode (計算グラフ表現)
  ↓
[Optimizer + Lowerer + Backend]
  (既存のパイプライン)
```

### 演算子の最小性

GraphやASTと同様に、**演算子の種類を最小限に抑える**設計方針を採用しています。複雑な演算は基本演算の組み合わせで表現します。

例：
- 減算: `a - b = a + (-b)`
- 除算: `a / b = a * recip(b)`

最終的に最適化とコンパイルが行われるため、ノード数の増加はパフォーマンス上の問題になりません。

## アーキテクチャ

### Tensor

`GraphNode`をラップし、勾配計算機能を追加した型です。

```rust
pub struct Tensor {
    pub data: GraphNode,              // 計算グラフノード
    requires_grad: bool,               // 勾配を計算するか
    grad: Rc<RefCell<Option<GraphNode>>>, // 累積された勾配
    grad_fn: Option<Rc<GradFnWrapper>>,   // backward時の勾配計算関数
}
```

### GradFn

各演算の勾配計算規則を定義するtraitです。

```rust
pub trait GradFn {
    fn apply(&self, grad_output: &Tensor, inputs: &[Tensor]) -> Vec<Option<Tensor>>;
}
```

実装済みの勾配関数：
- `AddBackward`: ∂L/∂a = ∂L/∂out, ∂L/∂b = ∂L/∂out
- `MulBackward`: ∂L/∂a = ∂L/∂out * b, ∂L/∂b = ∂L/∂out * a
- `NegBackward`: ∂L/∂a = -∂L/∂out
- `RecipBackward`: ∂L/∂a = -∂L/∂out / (a²)
- `ReduceSumBackward`: ∂L/∂a = expand(∂L/∂out)
- `AddConstBackward`: 定数との加算
- `MulConstBackward`: 定数との乗算
- `Conv1dBackward`: 1次元畳み込みの勾配（stride対応）
- `Conv2dBackward`: 2次元畳み込みの勾配（stride対応）
- `Conv3dBackward`: 3次元畳み込みの勾配（stride対応）

### backward処理

1. **トポロジカルソート**: 出力から入力に向かって計算順序を決定
2. **勾配伝播**: 逆順に各ノードの勾配を計算
3. **勾配累積**: 各入力テンソルに勾配を保存

重要な実装詳細：
- `GraphNodeData`のポインタをキーとして勾配を管理
- `Tensor`のクローンでも同じ`grad`フィールド（`Rc<RefCell<>>`）を共有

## 使用例

```rust
use harp::prelude::*;

fn main() {
    let mut graph = Graph::new();

    // 入力テンソル（requires_grad=true）
    let x = Tensor::from_graph_node(
        graph.input("x")
            .with_dtype(DType::F32)
            .with_shape([10, 20])
            .build(),
        true
    );

    let w = Tensor::from_graph_node(
        graph.input("w")
            .with_dtype(DType::F32)
            .with_shape([10, 20])
            .build(),
        true
    );

    // 前向き計算
    let y = 2.0 * &x + &w;
    let loss = y.sum(0);  // スカラーに縮約

    // 逆伝播
    loss.backward();

    // 勾配取得
    let grad_x = x.grad().unwrap();
    let grad_w = w.grad().unwrap();

    // 勾配グラフも通常のGraphNodeなので、既存のパイプラインでコンパイル可能
}
```

## 演算子オーバーロード

PyTorchライクな直感的な記法をサポート：

```rust
// Tensor op Tensor
let c = &a + &b;
let d = &a * &b;

// Tensor op f32
let e = &a + 2.0;
let f = &a * 3.14;

// f32 op Tensor
let g = 2.0 * &a;
let h = 1.0 / &a;

// 複雑な式
let result = 2.0 * &x + &w * 0.5;
```

## 拡張性

新しい演算の勾配関数を追加する場合：

1. `grad_fn.rs`に新しい`GradFn`実装を追加
2. `tensor.rs`に対応するメソッドを追加
3. テストを追加

基本演算から構成できる演算は、自動的に微分可能です。

## 畳み込み演算の実装

### Conv1d/2d/3dのBackward

畳み込みの逆伝播は、unfold（im2col）とfold（col2im）を用いて実装されています。

**アルゴリズム：**

1. **grad_input（入力の勾配）**: Transposed Convolution
   - カーネルを反転・転置: (C_out, C_in, k) → (C_in, C_out, k)
   - grad_outputとカーネルを要素ごとに乗算（ブロードキャスト）
   - C_out次元で総和を取る
   - Fold演算で元の入力形状に戻す（strideパラメータを使用）

2. **grad_kernel（カーネルの勾配）**: Correlation
   - 入力をunfold（stride対応）: (C_in, L) → (C_in, k, L_out)
   - grad_outputをreshape: (C_out, L_out) → (C_out, 1, 1, L_out)
   - 要素ごとに乗算してL_out次元で総和

**現在の対応状況：**
- ✅ stride != 1 をサポート
- ❌ dilation != 1 は未対応（panic）
- ❌ groups != 1 は未対応（panic）

### Fold演算（col2im）

Unfoldの逆操作。畳み込みのbackward計算で使用されます。

**実装：**
- `src/lowerer/fold.rs`: カーネル生成
- 2段階のループ：
  1. 出力バッファをゼロで初期化
  2. 入力の各要素を適切な出力位置に加算（accumulate）

## 制限事項

現在の実装での制限：
- backwardはスカラーテンソル（ndim=0）に対してのみ実行可能
- 比較演算（Max等）の勾配は簡易実装
- 高階微分は未実装
- 畳み込み演算はdilation=1, groups=1のみサポート

将来的な拡張：
- ベクトル値関数のJacobian計算
- 前向きモード自動微分
- 高階微分
- 畳み込み演算でのdilation, groupsサポート

## 将来的な改善計画

### Conv/Fold/Unfoldの実装統合

**現状：**
Conv1d/2d/3d、Fold1d/2d/3d、Unfold1d/2d/3dがそれぞれ独立した実装として存在します。

**問題点：**
- コードの重複が多い（約80%は同じロジック）
- 新しい次元（4D等）を追加する際の保守コストが高い
- 各次元で微妙に異なるバグが混入しやすい

**統合案：**

1. **マクロベースの実装（推奨）**
   ```rust
   macro_rules! impl_conv_backward {
       ($name:ident, $dim:expr, $unfold:ident, $fold:ident) => {
           // 共通ロジックを生成
       };
   }

   impl_conv_backward!(Conv1dBackward, 1, unfold1d, fold1d);
   impl_conv_backward!(Conv2dBackward, 2, unfold2d, fold2d);
   impl_conv_backward!(Conv3dBackward, 3, unfold3d, fold3d);
   ```

   **メリット：**
   - コード重複を大幅に削減
   - 型安全性を維持
   - Rustらしいアプローチ

   **デメリット：**
   - マクロデバッグが難しい
   - エディタのコード補完が効きにくい

2. **実行時次元数パラメータ**
   ```rust
   struct ConvBackward {
       ndim: usize,
       kernel_size: Vec<usize>,
       stride: Vec<usize>,
       // ...
   }
   ```

   **メリット：**
   - 実装がシンプル
   - 柔軟性が高い

   **デメリット：**
   - 型安全性が低下
   - 実行時エラーが増える可能性

3. **ジェネリクスとconst generics**
   ```rust
   struct ConvBackward<const N: usize> {
       kernel_size: [usize; N],
       stride: [usize; N],
       // ...
   }
   ```

   **メリット：**
   - 型安全性が最も高い
   - コンパイル時最適化が効く

   **デメリット：**
   - 実装が複雑
   - Rustのconst generics機能がまだ発展途上

**推奨アプローチ：**
マクロベースの統合を第一候補とします。現在の実装でも十分機能的ですが、将来的にコードベースが大きくなった際の保守性を考慮すると、統合によるメリットが大きくなります。

**実施時期：**
- 優先度：低（現状で動作に問題なし）
- タイミング：新しい次元の追加や大規模リファクタリング時に検討

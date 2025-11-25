# ニューラルネットワークモジュール (harp-nn)

PyTorchの`torch.nn`に相当する機能を提供するサブクレートです。

> **注意**: `harp-nn`は独立したクレートとして`crates/harp-nn/`に配置されています。
> 使用する場合は`harp_nn`としてインポートしてください。

## 設計思想

### 軽量なModule trait

harpの設計哲学（演算子の最小性、レイヤー分離）に従い、**必要最小限の機能に絞った軽量な実装**を採用しています。

```
[NN Layer]
  Module trait (parameters管理、forward実行)
  Parameter (requires_grad=true のTensor wrapper)
  ↓
[Autograd Layer]
  Tensor (自動微分機能)
  ↓
[Graph Layer]
  GraphNode (計算グラフ表現)
```

### New type pattern

`Parameter`は`Tensor`のnewtype wrapperとして実装されています。これにより：

- **型安全性**: パラメータとテンソルを型レベルで区別
- **ゼロコスト**: ランタイムオーバーヘッドなし
- **requires_grad強制**: コンストラクタで必ず`requires_grad=true`に設定

### 明示的な管理

PyTorchのような自動登録ではなく、Rustの所有権システムを活かした明示的なパラメータ管理を採用：

- パラメータは構造体のフィールドとして保持
- `parameters()`メソッドで明示的にリストを返す
- 階層的なモジュールは手動で管理

## アーキテクチャ

### Parameter

学習可能なパラメータを表す型。

```rust
pub struct Parameter(Tensor);

impl Parameter {
    pub fn new(tensor: Tensor) -> Self;
    pub fn zeros(shape: Vec<usize>) -> Self;
    pub fn ones(shape: Vec<usize>) -> Self;
    pub fn tensor(&self) -> &Tensor;
    pub fn tensor_mut(&mut self) -> &mut Tensor;
}

// Deref実装でTensorのメソッドを透過的に使用可能
impl Deref for Parameter {
    type Target = Tensor;
}
```

### Module trait

ニューラルネットワークのパラメータ管理を抽象化するtrait。

**重要な設計判断**:
1. `forward`メソッドは含まれていません（各モジュールで自由に定義）
2. パラメータは**必ず名前付き**で管理されます

```rust
pub trait Module {
    /// 名前付きパラメータの辞書を返す（必須実装）
    fn named_parameters(&self) -> HashMap<String, &Parameter>;

    /// 可変参照版（必須実装）
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter>;

    /// パラメータリストを返す（デフォルト実装）
    fn parameters(&self) -> Vec<&Parameter> {
        self.named_parameters().values().copied().collect()
    }

    /// 可変参照リスト（デフォルト実装）
    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        self.named_parameters_mut().into_values().collect()
    }

    /// 全パラメータの勾配をゼロクリア
    fn zero_grad(&self) { /* デフォルト実装あり */ }

    /// パラメータの総数を返す
    fn num_parameters(&self) -> usize { /* デフォルト実装あり */ }
}
```

#### 名前付きパラメータの利点

パラメータに名前を付けることで、以下が可能になります：

1. **モデルの保存/読み込み**: `state_dict()`で名前ベースのシリアライズ
2. **デバッグ**: どのパラメータが問題かを特定しやすい
3. **選択的な更新**: 特定の層のみ凍結（fine-tuning）
4. **転移学習**: 事前学習モデルのパラメータをロード
5. **既存フレームワークとの互換性**: PyTorch、TensorFlow等と同じ設計

#### forwardメソッドについて

`forward`は**traitメソッドではなく、各モジュールの通常のメソッド**として定義します。
これにより、以下の柔軟性が得られます：

- **引数の数が異なる**: `forward(&self, x: &Tensor)` vs `forward(&self, q: &Tensor, k: &Tensor, v: &Tensor)`
- **引数の型が異なる**: `forward(&self, x: &Tensor, mask: Option<&Tensor>)`
- **返り値が複数**: `forward(&self, x: &Tensor) -> (Tensor, Tensor)`（LSTMなど）

## impl_module!マクロ

Module traitの実装を自動化するマクロです。ボイラープレートを削減し、コードを簡潔に保ちます。

### 基本的な使い方

```rust
use harp_nn::{impl_module, Module, Parameter};

struct Linear {
    weight: Parameter,
    bias: Parameter,
}

// Parameterフィールドを指定するだけでModule traitが実装される
impl_module! {
    for Linear {
        parameters: [weight, bias]
    }
}
```

### 階層的なモジュール

サブモジュールを含む場合も簡単に実装できます：

```rust
struct MLP {
    layer1: Linear,
    layer2: Linear,
}

// サブモジュールを指定
impl_module! {
    for MLP {
        modules: [layer1, layer2]
    }
}

// パラメータ名は自動的に階層的になる
// "layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"
```

### パラメータとモジュールの混在

```rust
struct CustomNet {
    scale: Parameter,
    offset: Parameter,
    mlp: MLP,
}

impl_module! {
    for CustomNet {
        parameters: [scale, offset],
        modules: [mlp]
    }
}

// 生成されるパラメータ名：
// "scale", "offset", "mlp.layer1.weight", "mlp.layer1.bias", ...
```

### derive macroによる自動実装（実装済み）

`#[derive(DeriveModule)]`を使うことで、structの宣言だけでModule traitが自動実装されます：

```rust
use harp::prelude::*;
use harp_nn::{Module, Parameter};

#[derive(DeriveModule)]
struct Linear {
    weight: Parameter,
    bias: Parameter,
}
```

**自動検出機能:**
- `Parameter`型のフィールド → 自動的にパラメータとして登録
- `Module` traitを実装している型のフィールド → 自動的にサブモジュールとして登録
- 階層的な名前も自動生成（例: `mlp.layer1.weight`）

**比較：derive vs 宣言的マクロ**

| 特徴 | `#[derive(DeriveModule)]` | `impl_module!` |
|------|---------------------------|----------------|
| 使いやすさ | ✅ struct宣言に1行追加するだけ | フィールドを明示的に列挙 |
| 自動検出 | ✅ Parameter型を自動検出 | ❌ フィールドを手動で指定 |
| 依存関係 | `harp-derive` crate必要 | 不要 |
| コンパイル時間 | やや長い（proc-macro） | 短い |

**両方使える:**
どちらのマクロも利用可能です。好みに応じて選択できます。

## 使用例

### 基本的な使い方

```rust
use harp::prelude::*;
use harp_nn::{Module, Parameter};
use std::collections::HashMap;

// カスタムモジュールの定義
struct Linear {
    weight: Parameter,
    bias: Parameter,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            weight: Parameter::zeros(vec![out_features, in_features]),
            bias: Parameter::zeros(vec![out_features]),
        }
    }

    // forwardは通常のメソッドとして定義
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // matmulは将来実装予定
        // input.matmul(&self.weight.t()) + &self.bias
        unimplemented!()
    }
}

impl Module for Linear {
    fn named_parameters(&self) -> HashMap<String, &Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &self.weight);
        params.insert("bias".to_string(), &self.bias);
        params
    }

    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter> {
        let mut params = HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        params.insert("bias".to_string(), &mut self.bias);
        params
    }
}

// 使用例
let linear = Linear::new(10, 5);
let input = Tensor::ones(vec![10]);
let output = linear.forward(&input);

// パラメータ数を確認
println!("Total parameters: {}", linear.num_parameters());

// 名前付きパラメータの取得
for (name, param) in linear.named_parameters() {
    println!("{}: shape = {:?}", name, param.data.view.shape());
}

// 勾配をゼロクリア
linear.zero_grad();
```

### 複数引数のforward

```rust
// Attentionモジュールの例
struct Attention {
    weight_q: Parameter,
    weight_k: Parameter,
    weight_v: Parameter,
}

impl Attention {
    // 複数の引数を受け取るforward
    pub fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> Tensor {
        // Q, K, V の計算
        // let q = query.matmul(&self.weight_q);
        // let k = key.matmul(&self.weight_k);
        // let v = value.matmul(&self.weight_v);
        // attention計算...
        unimplemented!()
    }
}

impl Module for Attention {
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight_q, &self.weight_k, &self.weight_v]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight_q, &mut self.weight_k, &mut self.weight_v]
    }
}

// 使用
let attention = Attention { /* ... */ };
let output = attention.forward(&q, &k, &v);
```

### 階層的なモジュール

```rust
struct MLP {
    layer1: Linear,
    layer2: Linear,
}

impl MLP {
    // forwardを通常のメソッドとして定義
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let h = self.layer1.forward(input);
        let h = h.relu(); // 活性化関数
        self.layer2.forward(&h)
    }
}

impl Module for MLP {
    fn named_parameters(&self) -> HashMap<String, &Parameter> {
        let mut params = HashMap::new();

        // layer1のパラメータにプレフィックスを付けて追加
        for (name, param) in self.layer1.named_parameters() {
            params.insert(format!("layer1.{}", name), param);
        }

        // layer2のパラメータにプレフィックスを付けて追加
        for (name, param) in self.layer2.named_parameters() {
            params.insert(format!("layer2.{}", name), param);
        }

        params
    }

    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Parameter> {
        let mut params = HashMap::new();

        for (name, param) in self.layer1.named_parameters_mut() {
            params.insert(format!("layer1.{}", name), param);
        }

        for (name, param) in self.layer2.named_parameters_mut() {
            params.insert(format!("layer2.{}", name), param);
        }

        params
    }
}

// 使用例
let mlp = MLP { layer1, layer2 };

// 階層的な名前が取得できる
// "layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"
for (name, param) in mlp.named_parameters() {
    println!("{}", name);
}
```

## 初期化関数 (nn::init)

パラメータの初期化をサポートする関数群。

### 実装済み

```rust
pub fn constant(shape: Vec<usize>, value: f32) -> Tensor;
```

定数で初期化します。

### 将来実装予定

- `xavier_uniform(shape)`: Xavier/Glorot uniform初期化
- `xavier_normal(shape)`: Xavier/Glorot normal初期化
- `kaiming_uniform(shape)`: Heの初期化（uniform版）
- `kaiming_normal(shape)`: Heの初期化（normal版）
- `uniform(shape, low, high)`: 一様分布で初期化
- `normal(shape, mean, std)`: 正規分布で初期化

乱数初期化を実装するには、バックエンドに乱数生成機能が必要です。

## Optimizer (nn::optim)

パラメータの最適化アルゴリズムを提供するモジュール。

### Optimizer trait

```rust
pub trait Optimizer {
    /// パラメータを更新する
    fn step(&mut self, parameters: &mut [&mut Parameter]);

    /// 全パラメータの勾配をゼロクリア（デフォルト実装あり）
    fn zero_grad(&self, parameters: &[&Parameter]);
}
```

### SGD (確率的勾配降下法)

最もシンプルなオプティマイザー。

**更新式**: `param = param - learning_rate * grad`

```rust
pub struct SGD {
    lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self;
    pub fn lr(&self) -> f32;
    pub fn set_lr(&mut self, lr: f32);
}
```

**使用例:**

```rust
use harp::prelude::*;
use harp_nn::optim::{Optimizer, SGD};

let mut module = MyModule::new();
let mut optimizer = SGD::new(0.01);

// 学習ループ
for _ in 0..100 {
    // Forward
    let output = module.forward(&input);
    let loss = compute_loss(&output, &target);

    // Backward
    loss.backward();

    // Update parameters
    let mut params = module.parameters_mut();
    optimizer.step(&mut params);
    optimizer.zero_grad(&params);
}
```

**実装の詳細:**

- パラメータの更新は`***param = new_tensor`の形で行う
- `param`は`&mut Parameter`型なので、3段階のderef（`***`）でTensorにアクセス
- 勾配の計算グラフは追跡しない（`requires_grad=false`）
- 更新後の新しいParameterは`requires_grad=true`で作成

## 実装済みの機能

- ✅ `Parameter` 構造体
- ✅ `Module` trait
- ✅ `impl_module!` マクロ（宣言的マクロ）
- ✅ `#[derive(DeriveModule)]` マクロ（proc-macro）
- ✅ `Deref`/`DerefMut` による透過的なアクセス
- ✅ `parameters()` / `parameters_mut()`
- ✅ `named_parameters()` / `named_parameters_mut()`
- ✅ `zero_grad()`
- ✅ `num_parameters()`
- ✅ `nn::init::constant()`
- ✅ `Optimizer` trait
- ✅ `SGD` optimizer

## 未実装の機能

### 高優先度

- ❌ `matmul` 演算（Linear層に必須）
- ❌ `Linear` 層
- ❌ `Conv1d/2d/3d` 層のラッパー
- ❌ 追加のOptimizer (Adam, RMSprop, AdamW等)

### 中優先度

- ❌ 乱数初期化関数（Xavier, He等）
- ❌ `BatchNorm1d/2d/3d`
- ❌ `LayerNorm`
- ❌ `Dropout`

### 低優先度

- ❌ `Sequential` コンテナ
- ❌ `state_dict()` / `load_state_dict()`（モデル保存/読み込み）
- ❌ `train()` / `eval()` モード切り替え
- ❌ フック機能

## 将来的な拡張

### matmul演算の実装

Linear層を実装するには、行列積演算が必要です。

**実装候補:**

1. **unfold + reduce_sumで実装** (推奨)
   - 既存のconv演算と同じアプローチ
   - `src/graph/hlops.rs`に追加
   - 実装が比較的シンプル

2. **専用のMatmul演算を追加**
   - `GraphOp::Matmul`を新規追加
   - より効率的だが実装が大きい
   - Lowerer、Optimizer、Backend全てに変更が必要

### 追加のOptimizer実装

基本的なSGDは既に実装済み。将来的に以下のOptimizerを追加可能：

**Adam (Adaptive Moment Estimation):**

```rust
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    // 1次モーメント（移動平均）
    m: HashMap<usize, Tensor>,
    // 2次モーメント（移動分散）
    v: HashMap<usize, Tensor>,
    t: usize,  // タイムステップ
}
```

**SGD with Momentum:**

```rust
pub struct SGDMomentum {
    lr: f32,
    momentum: f32,
    velocity: HashMap<usize, Tensor>,
}
```

**注意:** 状態を管理するためのキーとして、Parameterのアドレス（`usize`）を使用。`*const GraphNode`よりも安全。

### 畳み込み層のラッパー

既存の`conv1d/2d/3d`演算を使って、`Module`として実装：

```rust
pub struct Conv2d {
    weight: Parameter,
    bias: Option<Parameter>,
    stride: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
}
```

## PyTorchとの比較

| 機能 | PyTorch | harp | 備考 |
|------|---------|------|------|
| Module trait | ✅ | ✅ | harpは軽量実装 |
| Parameter | ✅ | ✅ | newtype pattern |
| parameters() | ✅ | ✅ | |
| named_parameters() | ✅ | ✅ | 実装済み |
| Module自動実装（宣言的マクロ） | ❌ | ✅ | `impl_module!` |
| Module自動実装（derive） | ❌ | ✅ | `#[derive(DeriveModule)]` |
| Optimizer trait | ✅ | ✅ | 実装済み |
| SGD | ✅ | ✅ | 基本版のみ（momentumなし） |
| Adam | ✅ | ❌ | 将来実装 |
| state_dict() | ✅ | ❌ | 将来実装 |
| 自動登録 | ✅ | ❌ | 明示的管理を採用 |
| train/eval mode | ✅ | ❌ | 将来実装 |
| フック | ✅ | ❌ | 優先度低 |

## 設計の利点

1. **シンプル**: 必要最小限の機能のみ
2. **型安全**: Rustの型システムを最大限活用
3. **ゼロコスト**: ランタイムオーバーヘッドなし
4. **拡張性**: 段階的に機能を追加可能
5. **テスト容易**: 各コンポーネントが独立

## 設計のトレードオフ

**利点:**
- Rustらしい設計
- 保守性が高い
- パフォーマンスが予測可能
- `impl_module!`マクロによりボイラープレート削減

**欠点:**
- PyTorchとの互換性は低い
- 階層的なモジュールは明示的に管理（マクロで軽減）
- 自動パラメータ登録がない（型安全性とのトレードオフ）

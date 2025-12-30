# harp-nn ニューラルネットワークモジュール

ニューラルネットワーク構築のための層・損失関数・オプティマイザを提供します。

## ディレクトリ構成

```
crates/
├── nn/                       # メインクレート
│   └── src/
│       ├── lib.rs            # エントリポイント
│       ├── module.rs         # Module trait, Parameter
│       ├── layers/           # レイヤー
│       │   ├── mod.rs        # 公開API
│       │   ├── linear.rs     # 全結合層
│       │   ├── conv.rs       # 畳み込み層 (Conv/ConvTranspose)
│       │   └── activation.rs # 活性化関数層
│       ├── loss.rs           # 損失関数
│       └── optim/            # オプティマイザ
│           ├── mod.rs        # Optimizer trait
│           ├── sgd.rs        # SGD
│           ├── momentum.rs   # Momentum SGD
│           ├── rmsprop.rs    # RMSProp
│           └── adam.rs       # Adam
└── nn-derive/                # derive マクロ
    └── src/
        └── lib.rs            # #[derive(Module)]
```

## 主要コンポーネント

### Module トレイト

学習可能なパラメータを持つ計算ユニットの基底トレイト。

```rust
pub trait Module<T: FloatDType> {
    fn parameters(&mut self) -> HashMap<String, &mut Parameter<T>>;
    fn load_parameters(&mut self, params: HashMap<String, Parameter<T>>);
    fn zero_grad(&mut self) { /* default impl */ }
    fn num_parameters(&mut self) -> usize { /* default impl */ }
}
```

### Parameter<T>

学習可能なテンソルのラッパー。`requires_grad = true` で作成される。

```rust
pub struct Parameter<T: FloatDType>(Tensor<T, DimDyn>);
```

### レイヤー

#### 全結合層

- `Linear<T>` - 全結合層 (`y = x @ W + b`)

#### 畳み込み層

`unfold + broadcast multiply + sum` アプローチで実装。groups による分割畳み込み、depthwise 畳み込みに対応。
groups=1 と groups>1 は同一のロジックで処理され、groups=1 は groups>1 の特殊ケースとして扱われる。

| 層 | 入力 | カーネル | 出力 |
|----|------|----------|------|
| `Conv1d<T>` | `[N, C_in, L]` | k | `[N, C_out, L_out]` |
| `Conv2d<T>` | `[N, C_in, H, W]` | (kH, kW) | `[N, C_out, H_out, W_out]` |
| `Conv3d<T>` | `[N, C_in, D, H, W]` | (kD, kH, kW) | `[N, C_out, D_out, H_out, W_out]` |

**パラメータ：** stride, padding, dilation, groups, bias

```rust
// ビルダーパターン
let conv = Conv2d::<f32>::new(3, 64, (3, 3))
    .stride((2, 2))
    .padding((1, 1))
    .groups(1)
    .bias(true)
    .build();

let output = conv.forward(&input);
```

#### 転置畳み込み層

`matmul + fold` アプローチで実装。アップサンプリングや生成モデルで使用。

| 層 | 入力 | カーネル | 出力 |
|----|------|----------|------|
| `ConvTranspose1d<T>` | `[N, C_in, L]` | k | `[N, C_out, L_out]` |
| `ConvTranspose2d<T>` | `[N, C_in, H, W]` | (kH, kW) | `[N, C_out, H_out, W_out]` |
| `ConvTranspose3d<T>` | `[N, C_in, D, H, W]` | (kD, kH, kW) | `[N, C_out, D_out, H_out, W_out]` |

**パラメータ：** stride, padding, output_padding, dilation, groups, bias

出力サイズ（1次元の場合）：
`L_out = (L_in - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1`

```rust
let conv_t = ConvTranspose2d::<f32>::new(64, 3, (3, 3))
    .stride((2, 2))
    .padding((1, 1))
    .output_padding((1, 1))  // stride > 1 で出力サイズの曖昧さを解消
    .build();

let upsampled = conv_t.forward(&input);
```

#### 活性化関数層

全て `FloatDType` でジェネリック化されており、f32/f64 に対応。

| 層 | 関数 | 説明 |
|----|------|------|
| `ReLU<T>` | `max(0, x)` | 負の値を0にする |
| `LeakyReLU<T>` | `max(alpha*x, x)` | 負の値に小さな傾きを持たせる |
| `Sigmoid<T>` | `1 / (1 + exp(-x))` | 0〜1に正規化 |
| `Tanh<T>` | `(exp(2x)-1)/(exp(2x)+1)` | -1〜1に正規化 |
| `GELU<T>` | `x * sigmoid(1.702 * x)` | 高速近似版GELU |
| `SiLU<T>` | `x * sigmoid(x)` | Swish関数 |
| `Softplus<T>` | `ln(1 + exp(x))` | 滑らかなReLU |
| `Mish<T>` | `x * tanh(softplus(x))` | Mish活性化 |
| `ELU<T>` | `x if x>0, else alpha*(exp(x)-1)` | 指数線形ユニット |

```rust
// 使用例
let relu = ReLU::<f32>::new();
let output = relu.forward(&input);

let leaky = LeakyReLU::<f64>::new(0.01);
let output = leaky.forward(&input);
```

### #[derive(Module)] マクロ

`Module` トレイトを自動実装する derive マクロ。

**フィールド分類（自動検出）：**
- `Parameter<T>` 型 → パラメータとして直接追加
- `_` で始まるフィールド → スキップ
- `PhantomData` → スキップ
- その他 → Module として再帰的に探索

```rust
#[derive(Module)]
#[module(crate = "crate")]  // クレート内部で使用する場合
pub struct Linear<T: FloatDType = f32> {
    weight: Parameter<T>,
    bias: Parameter<T>,
    _dtype: PhantomData<T>,  // 自動スキップ
}
```

**外部クレートで使用する場合：**
```rust
use harp_nn::{Module, Parameter, Linear};

#[derive(Module)]
struct MyModel<T: FloatDType> {
    linear1: Linear<T>,     // Module として認識
    linear2: Linear<T>,     // Module として認識
    scale: Parameter<T>,    // Parameter として認識
    _marker: PhantomData<T>,  // スキップ
}
```

### Optimizer トレイト

パラメータ更新のための最適化アルゴリズム。

```rust
pub trait Optimizer<T: FloatDType> {
    fn step<M: Module<T>>(&mut self, module: &mut M);
}
```

**実装済みオプティマイザ：**
- `SGD<T>`: 確率的勾配降下法
- `Momentum<T>`: モメンタム付き SGD
- `RMSProp<T>`: 勾配二乗の移動平均による適応的学習率 (Hinton, 2012)
- `Adam<T>`: 一次・二次モーメントによる適応的学習率 (Kingma & Ba, 2014)

## 使用例

### MLP

```rust
use harp::tensor::{Tensor, Dim2, FloatDType};
use harp_nn::{Linear, ReLU, Module, SGD, Optimizer};

// モデル作成
let mut linear = Linear::<f32>::new(784, 128);
let relu = ReLU::<f32>::new();
let mut optimizer = SGD::new(0.01);

// 学習ループ
linear.zero_grad();
let hidden = linear.forward(&input);
let output = relu.forward(&hidden.into_dim2());
let loss = mse_loss(&output, &target);
loss.backward();
optimizer.step(&mut linear);
```

### CNN

```rust
use harp::tensor::{Tensor, Dim4};
use harp_nn::{Conv2d, ReLU};

// 畳み込みネットワーク
let conv1 = Conv2d::<f32>::new(3, 64, (3, 3))
    .padding((1, 1))
    .build();
let conv2 = Conv2d::<f32>::new(64, 128, (3, 3))
    .stride((2, 2))
    .padding((1, 1))
    .build();
let relu = ReLU::<f32>::new();

let input = Tensor::<f32, Dim4>::rand([1, 3, 32, 32]);
let h = relu.forward(&conv1.forward(&input).into_dyn()).into_dim4();
let h = relu.forward(&conv2.forward(&h).into_dyn()).into_dim4();
```

## ジェネリクス

全てのコンポーネントは `FloatDType`（f32, f64）でジェネリック化されている。

```rust
// f32 (デフォルト)
let linear_f32 = Linear::<f32>::new(10, 5);
let relu_f32 = ReLU::<f32>::new();
let sgd_f32 = SGD::<f32>::new(0.01);

// f64
let linear_f64 = Linear::<f64>::new(10, 5);
let relu_f64 = ReLU::<f64>::new();
let sgd_f64 = SGD::<f64>::new(0.01);

// 各オプティマイザの使い分け
let sgd = SGD::<f32>::new(0.01);
let momentum = Momentum::<f32>::new(0.01, 0.9);
let rmsprop = RMSProp::<f32>::new(0.001);
let adam = Adam::<f32>::new(0.001);
```

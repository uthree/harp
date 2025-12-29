# harp-nn ニューラルネットワークモジュール

ニューラルネットワーク構築のための層・損失関数・オプティマイザを提供します。

## ディレクトリ構成

```
crates/
├── nn/                       # メインクレート
│   └── src/
│       ├── lib.rs            # エントリポイント
│       ├── module.rs         # Module trait, Parameter
│       ├── layers.rs         # Linear など
│       ├── loss.rs           # 損失関数
│       └── optim/            # オプティマイザ
│           ├── mod.rs        # Optimizer trait
│           ├── sgd.rs        # SGD
│           └── momentum.rs   # Momentum SGD
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

## 使用例

```rust
use harp::tensor::{Tensor, Dim2, FloatDType};
use harp_nn::{Linear, Module, SGD, Optimizer};

// モデル作成
let mut linear = Linear::<f32>::new(784, 128);
let mut optimizer = SGD::new(0.01);

// 学習ループ
linear.zero_grad();
let output = linear.forward(&input);
let loss = mse_loss(&output, &target);
loss.backward();
optimizer.step(&mut linear);
```

## ジェネリクス

全てのコンポーネントは `FloatDType`（f32, f64）でジェネリック化されている。

```rust
// f32 (デフォルト)
let linear_f32 = Linear::<f32>::new(10, 5);
let sgd_f32 = SGD::<f32>::new(0.01);

// f64
let linear_f64 = Linear::<f64>::new(10, 5);
let sgd_f64 = SGD::<f64>::new(0.01);
```

# eclat-nn 仕様

eclat-nn は PyTorch ライクなニューラルネットワーク層とオプティマイザを提供する crate です。

## モジュール構成

```
eclat-nn/src/
├── lib.rs              (エントリーポイント、re-exports)
├── layers/             (ニューラルネットワーク層)
│   ├── mod.rs
│   ├── conv.rs         (畳み込み層)
│   ├── linear.rs       (全結合層)
│   ├── module.rs       (Module トレイト)
│   └── parameter.rs    (パラメータ管理)
├── functional/         (純粋な演算関数)
│   ├── mod.rs
│   ├── conv.rs         (畳み込み演算)
│   └── linear.rs       (線形演算)
└── optim/              (オプティマイザ)
    ├── mod.rs
    ├── optimizer.rs    (Optimizer トレイト)
    ├── sgd.rs          (SGD)
    └── adam.rs         (Adam)
```

## 設計方針

### layers と functional の分離

- `layers`: パラメータを持つ構造体（Conv2d, Linear など）
- `functional`: 純粋な演算関数（conv2d, linear など）

レイヤー構造体は内部で functional 関数を呼び出すことで、演算ロジックの再利用性を確保しています。

### functional モジュール

演算関数はパラメータを持たず、テンソルのみを受け取って計算を行います。

```rust
// 線形変換: y = xW^T + b
pub fn linear(input: &Tensor<D2, f32>, weight: &Tensor<D2, f32>, bias: Option<&Tensor<D1, f32>>) -> Tensor<D2, f32>

// 2D 畳み込み
pub fn conv2d(input: &Tensor<D4, f32>, weight: &Tensor<D4, f32>, bias: Option<&Tensor<D1, f32>>, stride: (usize, usize), padding: (usize, usize), dilation: (usize, usize)) -> Tensor<D4, f32>
```

### layers モジュール

レイヤー構造体はパラメータを管理し、forward メソッドで functional 関数を呼び出します。

```rust
impl Linear {
    pub fn forward_d2(&self, input: &Tensor<D2, f32>) -> Tensor<D2, f32> {
        let bias = self.bias.as_ref().map(|b| b.tensor());
        functional::linear(input, &self.weight.tensor(), bias.as_deref())
    }
}
```

## 利用可能な層

### 畳み込み層
- `Conv1d`, `Conv2d`, `Conv3d`: 順畳み込み
- `ConvTranspose1d`, `ConvTranspose2d`, `ConvTranspose3d`: 転置畳み込み

### 全結合層
- `Linear`: 全結合層 (y = xW^T + b)

## オプティマイザ

- `SGD`: モメンタム付き確率的勾配降下法
- `Adam`: 適応的モーメント推定

# eclat-nn 仕様

eclat-nn は PyTorch ライクなニューラルネットワーク層とオプティマイザを提供する crate です。

## モジュール構成

```
eclat-nn/src/
├── lib.rs              (エントリーポイント、re-exports)
├── layers/             (ニューラルネットワーク層)
│   ├── mod.rs
│   ├── activation.rs   (活性化層: PReLU)
│   ├── attention.rs    (アテンション層: MultiheadAttention)
│   ├── conv.rs         (畳み込み層)
│   ├── linear.rs       (全結合層)
│   ├── module.rs       (Module トレイト)
│   └── parameter.rs    (パラメータ管理)
├── functional/         (純粋な演算関数)
│   ├── mod.rs
│   ├── activation.rs   (活性化関数)
│   ├── attention.rs    (アテンション演算)
│   ├── conv.rs         (畳み込み演算)
│   ├── linear.rs       (線形演算)
│   ├── loss.rs         (損失関数)
│   └── pool.rs         (プーリング演算)
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
    pub fn forward(&self, input: &Tensor<D2, f32>) -> Tensor<D2, f32> {
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
  - 2D テンソル `[batch, in_features]` を受け取り `[batch, out_features]` を出力
  - 高次元テンソルに適用する場合は reshape で 2D に変換してから適用

### 活性化層（パラメータあり）
- `PReLU`: 学習可能な負の傾きを持つ ReLU

### アテンション層
- `MultiheadAttention`: マルチヘッドアテンション（Transformer の基本構成要素）
  - `new(embed_dim, num_heads, bias)` でインスタンス化
  - `forward(query, key, value, attn_mask)` で推論
  - Q, K, V 投影と出力投影を内蔵
  - 内部で 3D -> 2D -> Linear -> 3D の reshape を使用

## 活性化関数 (functional)

パラメータを持たない活性化関数は `functional` モジュールで提供されます。

### 基本的な活性化関数
- `relu`: ReLU (max(0, x))
- `leaky_relu`: Leaky ReLU (max(0, x) + α * min(0, x))
- `sigmoid`: シグモイド (1 / (1 + exp(-x)))
- `tanh`: 双曲線正接
- `elu`: ELU (x if x > 0, else α * (exp(x) - 1))

### 高度な活性化関数
- `gelu`: GELU (Gaussian Error Linear Unit)
- `silu`: SiLU / Swish (x * sigmoid(x))

### Softmax 系
- `softmax`: Softmax（指定した軸に沿って正規化）
- `log_softmax`: Log-Softmax（数値安定性向上版）

### パラメータ付き
- `prelu`: PReLU（学習可能な weight を受け取る、任意次元に対応）

## アテンション関数 (functional)

- `scaled_dot_product_attention`: Scaled Dot-Product Attention
  - 入力: query, key, value (D4テンソル `[B, H, L, D]`)、オプションでマスク
  - 出力: `softmax(QK^T / sqrt(d)) @ V`

## 損失関数 (functional)

- `cross_entropy_loss`: クロスエントロピー損失
  - 入力: logits `[N, C]`、target class indices `[N]`、num_classes
  - 出力: スカラー損失 `D0` テンソル
  - 内部で log_softmax + one-hot エンコーディングを使用
- `predict_classes`: ロジットからクラス予測を取得
  - 入力: logits `[N, C]`
  - 出力: 予測クラスインデックスのベクトル
- `accuracy`: 予測精度の計算
  - 入力: predictions, targets
  - 出力: 精度 (0.0 - 1.0)

## プーリング層

### 最大プーリング
- `MaxPool1d`, `MaxPool2d`, `MaxPool3d`: 最大プーリング

### 平均プーリング
- `AvgPool1d`, `AvgPool2d`, `AvgPool3d`: 平均プーリング

### 適応的プーリング
- `AdaptiveAvgPool2d`, `AdaptiveMaxPool2d`: 出力サイズを指定するプーリング

## オプティマイザ

- `SGD`: モメンタム付き確率的勾配降下法
- `Adam`: 適応的モーメント推定

## 使用例

### MNIST CNN の学習

完全な実装例は `examples/mnist_cnn/` にあります。

```bash
cargo run --release -p mnist_cnn
```

```rust
// モデル定義
let conv1 = Conv2d::new(1, 32, (3, 3)).with_padding((1, 1)).with_bias();
let conv2 = Conv2d::new(32, 64, (3, 3)).with_padding((1, 1)).with_bias();
let fc1 = Linear::new(3136, 128, true);
let fc2 = Linear::new(128, 10, true);
let pool = MaxPool2d::new((2, 2));

// 順伝播
let x = conv1.forward(&input);
let x = relu(&x);
let x = pool.forward(&x);
// ...
let logits = fc2.forward(&x);

// 損失計算
let loss = cross_entropy_loss(&logits, &targets, 10);

// 逆伝播
loss.backward()?;

// パラメータ更新
optimizer.step()?;
```

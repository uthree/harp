# Tensor モジュール仕様

統合Tensor型を提供するモジュール。遅延評価と自動微分をサポート。

## 設計思想

tinygrad/microgradの設計哲学に基づき、最小のプリミティブ演算の組み合わせで複雑な機能を実現。

## 主要コンポーネント

### Tensor<D>

```rust
pub struct Tensor<D: Dimension = DimDyn> {
    node: GraphNode,           // 計算グラフノード
    shape: Vec<usize>,         // テンソル形状
    dtype: DType,              // データ型
    autograd: Option<Rc<TensorData>>,  // 勾配追跡データ
    _dim: PhantomData<D>,      // 次元マーカー
}
```

### Dimension トレイト

静的次元と動的次元を統一的に扱う。

- `Dim<N>`: 静的次元（コンパイル時に次元数が決定）
  - `Dim0` ~ `Dim6`: 0〜6次元テンソル用の型エイリアス
- `DimDyn`: 動的次元（実行時に次元数が決定）

### GradFn トレイト

勾配関数のインターフェース。

```rust
pub trait GradFn {
    fn backward(&self, grad_output: &Tensor<DimDyn>) -> Vec<Tensor<DimDyn>>;
    fn inputs(&self) -> Vec<Tensor<DimDyn>>;
    fn name(&self) -> &'static str;
}
```

## 演算の分類

### primops（プリミティブ演算）

最小限の基本演算。これらの組み合わせで全ての演算を表現する。

#### 初期化
| 演算 | 説明 |
|------|------|
| `Const` | 定数テンソル |
| `Rand` | 一様乱数 [0, 1) |

#### 要素ごとの演算（二項）
| 演算 | 説明 |
|------|------|
| `Add` | 加算 |
| `Mul` | 乗算 |
| `Max` | 最大値 |
| `Idiv` | 整数除算 |

#### 要素ごとの演算（単項）
| 演算 | 説明 |
|------|------|
| `Neg` | 否定 (-x) |
| `Recip` | 逆数 (1/x) |
| `Sqrt` | 平方根 |
| `Log2` | 2を底とする対数 |
| `Exp2` | 2のべき乗 |
| `Sin` | 正弦 |

#### 縮約演算
| 演算 | 説明 |
|------|------|
| `Reduce(Add)` | 総和 |
| `Reduce(Mul)` | 総積 |
| `Reduce(Max)` | 最大値 |

#### 形状変更
| 演算 | 説明 |
|------|------|
| `Squeeze` | サイズ1の次元を削除 |
| `Unsqueeze` | サイズ1の次元を追加 |
| `Repeat` | 次元方向に繰り返し |
| `Reshape` | 形状変更 |
| `Contiguous` | メモリレイアウト正規化 |

---

### hlops（高級演算）

primopsの組み合わせで表現される演算。

| 演算 | primopsによる表現 |
|------|-------------------|
| `Sub(a, b)` | `Add(a, Neg(b))` |
| `Div(a, b)` | `Mul(a, Recip(b))` |
| `Exp(x)` | `Exp2(Mul(x, log2(e)))` |
| `Ln(x)` | `Mul(Log2(x), ln(2))` |
| `Cos(x)` | `Sin(Add(x, π/2))` |
| `ReLU(x)` | `Max(x, 0)` |
| `Sigmoid(x)` | `Recip(Add(1, Exp(Neg(x))))` |
| `Tanh(x)` | `Div(Sub(Exp(2x), 1), Add(Exp(2x), 1))` |
| `Mean(x, axes)` | `Div(Reduce(Add, x, axes), count)` |
| `Softmax(x)` | `Div(Exp(x - max), Reduce(Add, Exp(x - max)))` |
| `MatMul(a, b)` | `Reduce(Add, Mul(Unsqueeze(a), Unsqueeze(b)))` |
| `Conv2d` | Unfold + MatMul (im2col方式) |

#### MatMulの展開例

```
A: [M, K], B: [K, N] → C: [M, N]

1. Unsqueeze(A, -1)     → [M, K, 1]
2. Unsqueeze(B, 0)      → [1, K, N]
3. Mul                  → [M, K, N]  (broadcast)
4. Reduce(Add, axis=1)  → [M, N]
```

## API

### テンソル作成

```rust
// 静的次元
let zeros = Tensor::<Dim2>::zeros([3, 4]);
let ones = Tensor::<Dim2>::ones([3, 4]);
let full = Tensor::<Dim2>::full([3, 4], 2.5);
let input = Tensor::<Dim2>::input("x", [3, 4]);

// 動的次元
let zeros = Tensor::<DimDyn>::zeros_dyn(&[3, 4, 5]);
```

### 演算

```rust
// 算術演算
let c = &a + &b;
let c = &a - &b;
let c = &a * &b;
let c = &a / &b;
let c = -&a;

// スカラー演算
let c = &a + 1.0;
let c = 2.0 * &a;

// 関数
let y = x.exp();
let y = x.ln();
let y = x.sqrt();
let y = x.sin();
let y = x.relu();

// 縮約
let sum = x.sum(&[0, 1], true);  // keepdim=true
let mean = x.mean(&[1], false);

// 形状操作
let y = x.reshape([6]);
let y = x.permute(&[1, 0]);
let y = x.transpose();
let y = x.expand(&[4, 3]);
let y = x.unsqueeze(0);
let y = x.squeeze();
let y = x.flatten();
```

### 勾配追跡

```rust
// 勾配追跡を有効化
let x = Tensor::<Dim2>::ones([2, 2]).set_requires_grad(true);

// 演算（勾配が自動追跡される）
let y = &x * &x;

// 逆伝播
y.backward();

// 勾配取得
let grad = x.grad().unwrap();

// 勾配リセット
x.zero_grad();

// 計算グラフから切り離し
let detached = x.detach();
```

### 計算実行

```rust
// デフォルトデバイスで実行
x.forward()?;

// 結果取得
let data: Vec<f32> = x.data().unwrap();
```

## 勾配関数実装

### primops の勾配

| 演算 | 勾配計算 |
|------|----------|
| `z = Add(a, b)` | ∂L/∂a = ∂L/∂z, ∂L/∂b = ∂L/∂z |
| `z = Mul(a, b)` | ∂L/∂a = ∂L/∂z · b, ∂L/∂b = ∂L/∂z · a |
| `z = Max(a, b)` | ∂L/∂a = ∂L/∂z · (a ≥ b), ∂L/∂b = ∂L/∂z · (b > a) |
| `z = Neg(a)` | ∂L/∂a = -∂L/∂z |
| `z = Recip(a)` | ∂L/∂a = -∂L/∂z / a² |
| `z = Sqrt(a)` | ∂L/∂a = ∂L/∂z / (2·√a) |
| `z = Log2(a)` | ∂L/∂a = ∂L/∂z / (a · ln(2)) |
| `z = Exp2(a)` | ∂L/∂a = ∂L/∂z · 2^a · ln(2) |
| `z = Sin(a)` | ∂L/∂a = ∂L/∂z · cos(a) |
| `z = Reduce(Add)` | ∂L/∂a = expand(∂L/∂z) |
| `z = Reduce(Mul)` | ∂L/∂a = ∂L/∂z · z / a |
| `z = Reduce(Max)` | ∂L/∂a = ∂L/∂z · (a == max) |

### hlops の勾配

hlopsはprimopsに展開されるため、連鎖律により自動的に勾配が計算される。
便宜上、よく使う演算の勾配を示す：

| 演算 | 勾配計算 |
|------|----------|
| `z = Sub(a, b)` | ∂L/∂a = ∂L/∂z, ∂L/∂b = -∂L/∂z |
| `z = Div(a, b)` | ∂L/∂a = ∂L/∂z / b, ∂L/∂b = -∂L/∂z · a / b² |
| `z = Exp(a)` | ∂L/∂a = ∂L/∂z · exp(a) |
| `z = Ln(a)` | ∂L/∂a = ∂L/∂z / a |
| `z = ReLU(a)` | ∂L/∂a = ∂L/∂z · (a > 0) |
| `z = Mean(a)` | ∂L/∂a = expand(∂L/∂z) / count |

## グローバルデバイス管理

`thread_local!`でスレッドごとにデフォルトデバイスを管理。

```rust
// デバイス設定
set_default_device(device, DeviceKind::Metal);

// デバイス種類取得
let kind = get_default_device_kind();

// デバイス取得
let device: Arc<MetalDevice> = get_default_device().unwrap();

// スコープ付きデバイス変更
with_device(other_device, DeviceKind::OpenCL, || {
    // このスコープ内ではother_deviceがデフォルト
});

// デバイスクリア
clear_default_device();
```

## エラー型

### ForwardError

```rust
pub enum ForwardError {
    NoDefaultDevice,           // デフォルトデバイス未設定
    DeviceUnavailable(String), // デバイス利用不可
    CompilationError(String),  // コンパイル失敗
    ExecutionError(String),    // 実行失敗
    MissingInputData(String),  // 入力データ不足
    NoComputation,             // 計算グラフなし
}
```

### BackwardError

```rust
pub enum BackwardError {
    NoGrad,                    // 勾配追跡無効
    ShapeMismatch { expected, got }, // 形状不一致
}
```

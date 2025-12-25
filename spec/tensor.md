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

## プリミティブ演算

### 二項演算
- `Add`: 加算
- `Mul`: 乗算
- `Max`: 要素ごとの最大値

### 単項演算
- `Neg`: 否定
- `Recip`: 逆数 (1/x)
- `Exp2`: 2^x
- `Log2`: log2(x)
- `Sin`: 正弦
- `Sqrt`: 平方根

### 派生演算（プリミティブの組み合わせ）
- `Sub`: 減算 (a + (-b))
- `Div`: 除算 (a * recip(b))
- `Exp`: e^x (exp2(x * log2(e)))
- `Ln`: 自然対数 (log2(x) * ln(2))
- `Relu`: max(0, x)

### 移動演算
- `Reshape`: 形状変更
- `Permute`: 軸の並べ替え
- `Expand`: ブロードキャスト
- `Shrink`: 部分テンソル抽出
- `Pad`: パディング

### 縮約演算
- `ReduceSum`: 合計
- `ReduceMax`: 最大値

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

各演算に対応するGradFn実装：

| 演算 | GradFn | 勾配計算 |
|------|--------|----------|
| z = a + b | AddBackward | dL/da = dL/dz, dL/db = dL/dz |
| z = a - b | SubBackward | dL/da = dL/dz, dL/db = -dL/dz |
| z = a * b | MulBackward | dL/da = dL/dz * b, dL/db = dL/dz * a |
| z = a / b | DivBackward | dL/da = dL/dz / b, dL/db = -dL/dz * a / b² |
| z = -a | NegBackward | dL/da = -dL/dz |
| z = exp(a) | ExpBackward | dL/da = dL/dz * exp(a) |
| z = ln(a) | LogBackward | dL/da = dL/dz / a |
| z = sqrt(a) | SqrtBackward | dL/da = dL/dz / (2 * sqrt(a)) |
| z = sin(a) | SinBackward | dL/da = dL/dz * cos(a) |
| z = sum(a) | SumBackward | dL/da = expand(dL/dz) |
| z = mean(a) | MeanBackward | dL/da = expand(dL/dz) / count |

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

# Tensor モジュール仕様

**Tensorはharpの主要なAPIです。** 統合Tensor型を提供し、遅延評価、自動微分、Eager Fusionをサポートします。

## 設計思想

tinygrad/microgradの設計哲学に基づき、最小のプリミティブ演算の組み合わせで複雑な機能を実現。

## アーキテクチャ

### 内部構造

TensorはTensorInnerを内包し、TensorOpで演算を管理します。入力テンソルはTensorOp内に埋め込まれます。

```rust
pub struct Tensor<D: Dimension = DimDyn> {
    inner: Arc<TensorInner>,      // 内部データ（Arc for sharing）
    _dim: PhantomData<D>,         // 次元型マーカー
}

pub struct TensorInner {
    op: TensorOp,                     // 演算種類（入力テンソルを含む）
    view: View,                       // メモリレイアウト
    shape: Vec<usize>,                // テンソル形状
    dtype: DType,                     // データ型
    name: Option<String>,             // バッファ名（オプション）
    autograd: Option<AutogradMeta>,   // 勾配追跡データ
    buffer: RwLock<Option<Vec<f32>>>, // 実行結果バッファ
}

pub type TensorRef = Arc<Tensor<DimDyn>>;
```

### TensorOp

演算の種類を表すenum。入力テンソルはOp内に埋め込まれる。

```rust
pub enum TensorOp {
    // ソース演算（入力なし）
    Buffer { name: String },
    Const(Literal),
    ConstFill(Literal),
    Rand,
    Arange,
    Executed,

    // 単項演算（1入力）
    View { input: TensorRef },
    Contiguous { input: TensorRef },
    Cast { input: TensorRef, target_dtype: DType },
    Clone { input: TensorRef },

    // 統一計算演算（Compute）
    Compute {
        inputs: Vec<TensorRef>,      // 入力テンソル群
        expr: AstNode,               // 計算式
        reduce_op: Option<ReduceOp>, // 縮約演算（オプション）
        axes: Vec<usize>,            // 縮約軸
        keepdim: bool,               // 次元を維持するか
    },

    // 構造演算
    Pad { input: TensorRef, padding: Vec<(Expr, Expr)>, value: f32 },
    Slice { input: TensorRef, ranges: Vec<(usize, usize)> },
    Concat { inputs: Vec<TensorRef>, axis: usize },
}
```

### Compute演算の統一

全ての計算演算がCompute variantで統一的に表現されます。

| 旧表現 | 新表現（Compute） |
|--------|-------------------|
| `Elementwise { op }` | `reduce_op: None, axes: []` |
| `FusedElementwise { expr }` | `reduce_op: None, axes: []` |
| `Reduce { op, axes }` | `expr: Wildcard("0"), reduce_op: Some(op)` |
| `FusedElementwiseReduce` | `expr + reduce_op: Some(op)` |

## Eager Fusion

演算呼び出し時に即座に融合判定を行い、可能であれば融合演算に変換。

### 融合パターン

| 親演算 | 子演算 | 融合可能 |
|--------|--------|----------|
| Elementwise Compute | Elementwise Compute | ○ |
| Elementwise Compute | Reduce Compute | ○ |
| Reduce Compute | * | × |

### 所有権ベース設計

演算はselfを消費（move）する。分岐が必要な場合は明示的に`fork()`を使用。

```rust
let a = x + y;           // x, y消費
let b = a.sum();         // a消費 → 融合OK

// 分岐が必要な場合
let a = x + y;
let a2 = a.fork();       // Clone演算がグラフに追加
let b = a.sum();         // a消費 → 融合OK
let c = a2 * 2.0;        // a2は別パス → OK
```

## 主要コンポーネント

### Dimension トレイト

静的次元と動的次元を統一的に扱う。

- `Dim<N>`: 静的次元（コンパイル時に次元数が決定）
  - `Dim0` ~ `Dim6`: 0〜6次元テンソル用の型エイリアス
- `DimDyn`: 動的次元（実行時に次元数が決定）

### GradFn トレイト

勾配関数のインターフェース。

```rust
pub trait GradFn: Send + Sync {
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
| `ConstFill` | 定数値で埋める |
| `Rand` | 一様乱数 [0, 1) |
| `Arange` | 連番テンソル |

#### 要素ごとの演算（二項）
| 演算 | 説明 |
|------|------|
| `Add` | 加算 |
| `Mul` | 乗算 |
| `Max` | 最大値 |
| `Idiv` | 整数除算 |
| `Rem` | 剰余 |

#### 要素ごとの演算（単項）
| 演算 | 説明 |
|------|------|
| `Neg` | 否定 (-x) |
| `Recip` | 逆数 (1/x) |
| `Sqrt` | 平方根 |
| `Log2` | 2を底とする対数 |
| `Exp2` | 2のべき乗 |
| `Sin` | 正弦 |
| `Floor` | 床関数（非微分可能、勾配=0） |

#### 縮約演算
| 演算 | 説明 |
|------|------|
| `Reduce(Sum)` | 総和 |
| `Reduce(Prod)` | 総積 |
| `Reduce(Max)` | 最大値 |

#### 形状変更
| 演算 | 説明 |
|------|------|
| `View` | メモリコピーなしのView変更 |
| `Contiguous` | メモリレイアウト正規化・実行トリガー |
| `Pad` | パディング |
| `Slice` | スライス |

#### 特殊
| 演算 | 説明 |
|------|------|
| `Clone` | 分岐点（バッファコピー） |
| `Cast` | 型変換 |

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
| `Mean(x, axes)` | `Div(Reduce(Sum, x, axes), count)` |
| `Softmax(x)` | `Div(Exp(x - max), Reduce(Sum, Exp(x - max)))` |
| `MatMul(a, b)` | `Reduce(Sum, Mul(Unsqueeze(a), Unsqueeze(b)))` |
| `Conv2d` | Unfold + MatMul (im2col方式) |

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

// 分岐
let a2 = a.fork();  // Clone演算を追加
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
// realize()で実行トリガー（tinygradと同様）
let result = x.realize()?;

// 結果取得
let data: Vec<f32> = result.data().unwrap();

// contiguous()はメモリレイアウトの正規化（グラフノード作成のみ）
let contiguous = x.contiguous();

// from_data()で既存データからTensorを作成
let t = Tensor::<DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
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
| `z = Reduce(Sum)` | ∂L/∂a = expand(∂L/∂z) |
| `z = Reduce(Prod)` | ∂L/∂a = ∂L/∂z · z / a |
| `z = Reduce(Max)` | ∂L/∂a = ∂L/∂z · (a == max) |

### 融合演算の勾配

Compute演算の勾配はシンボリック微分により勾配を計算。

```rust
// Compute演算の勾配
// - AstNode式を各入力Wildcardに対してシンボリック微分
// - 導出した微分式をテンソル値で評価して勾配を計算

// Reduce付きCompute演算
// - まず勾配をReduce前の形状に展開（unsqueeze + expand）
// - 次にElementwise部分のシンボリック微分を適用
```

## モジュール構成

```
src/tensor/
├── mod.rs          # Tensor構造体、TensorInner、GradFn
├── dimension.rs    # Dimension トレイト
├── ops.rs          # TensorOp、ElementwiseOp、ReduceOp、TensorRef
├── fusion.rs       # Eager Fusion ロジック（can_fuse）
├── forward.rs      # forward()、realize() 実行
├── shape/          # 形状関連の型
│   ├── mod.rs
│   ├── expr.rs     # Expr（シンボリック式）
│   └── view.rs     # View（メモリレイアウト）
├── lowerer/        # TensorLowerer（Tensor → AST変換）
│   ├── mod.rs
│   ├── expr_builder.rs
│   └── helpers.rs
├── hlops/          # 高級演算
│   ├── activation.rs
│   ├── arithmetic.rs
│   ├── linalg.rs
│   ├── reduction.rs
│   └── transcendental.rs
└── primops/        # プリミティブ演算
    ├── binary.rs
    ├── grad.rs
    ├── init.rs
    ├── movement.rs
    ├── reduce.rs
    └── unary.rs
```

## TensorLowerer

TensorからASTへの変換を行うLowerer。

```rust
use harp::tensor::lowerer::{TensorLowerer, lower_tensor};

let a = Tensor::<Dim2>::input("a", [2, 3]);
let b = Tensor::<Dim2>::input("b", [2, 3]);
let c = &a + &b;

// TensorLowererを使用
let mut lowerer = TensorLowerer::new();
let ast = lowerer.lower(&c.clone().into_dyn());

// または簡易関数を使用
let ast = lower_tensor(&c.clone().into_dyn());
```

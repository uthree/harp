# Tensor モジュール仕様

**Tensorはharpの主要なAPIです。** 統合Tensor型を提供し、遅延評価、自動微分、Eager Fusionをサポートします。

## 設計思想

tinygrad/microgradの設計哲学に基づき、最小のプリミティブ演算の組み合わせで複雑な機能を実現。

## アーキテクチャ

### 内部構造

TensorはTensorInnerを内包し、TensorOpで演算を管理します。入力テンソルはTensorOp内に埋め込まれます。

```rust
pub struct Tensor<T: TensorDType = f32, D: Dimension = DimDyn> {
    inner: Arc<TensorInner>,      // 内部データ（Arc for sharing）
    _dtype: PhantomData<T>,       // データ型マーカー
    _dim: PhantomData<D>,         // 次元型マーカー
}

pub struct TensorInner {
    op: TensorOp,                             // 演算種類（入力テンソルを含む）
    view: View,                               // メモリレイアウト
    shape: Vec<usize>,                        // テンソル形状
    dtype: DType,                             // データ型
    name: Option<String>,                     // バッファ名（オプション）
    autograd: Option<AutogradStorage>,        // 勾配追跡データ（F32/F64）
    buffer: RwLock<Option<Box<dyn Buffer>>>, // 実行結果バッファ（GPU/ホスト）
}

// 型消去された自動微分ストレージ
pub enum AutogradStorage {
    F32(AutogradMeta<f32>),
    F64(AutogradMeta<f64>),
}

pub type TensorRef = Arc<Tensor<f32, DimDyn>>;
```

### TensorDType トレイト階層

型レベルでのデータ型制約を提供します。

```rust
pub trait TensorDType: Clone + Send + Sync + 'static {
    const DTYPE: DType;
}

// 数値型（算術演算可能）
pub trait NumericDType: TensorDType {}

// 浮動小数点型（sin, cos, sqrt等が利用可能）
pub trait FloatDType: NumericDType {}

// 整数型（ビット演算等が利用可能）
pub trait IntegerDType: NumericDType {}

// 符号付き整数型
pub trait SignedIntDType: IntegerDType {}

// 符号なし整数型
pub trait UnsignedIntDType: IntegerDType {}
```

| 型 | TensorDType | NumericDType | FloatDType | IntegerDType |
|----|-------------|--------------|------------|--------------|
| f32, f64 | ○ | ○ | ○ | × |
| i8, i16, i32, i64 | ○ | ○ | × | ○ (Signed) |
| u8, u16, u32, u64 | ○ | ○ | × | ○ (Unsigned) |
| bool | ○ | × | × | × |

**NumericInitDType**: 初期化操作のための共通トレイト

```rust
pub trait NumericInitDType: NumericDType {
    const ZERO: Self;
    const ONE: Self;
    fn to_literal(val: Self) -> Literal;
}
```

FloatDType と IntegerDType の両方がこのトレイトを実装しており、`zeros`, `ones`, `full`, `input` などの初期化メソッドが統一的に利用可能です。

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
    Clone { input: TensorRef },

    // 統一計算演算（MapReduce）
    // Cast演算もMapReduceとして表現（融合可能）
    MapReduce {
        inputs: Vec<TensorRef>,      // 入力テンソル群
        expr: AstNode,               // 計算式
        reduce_op: Option<ReduceOp>, // 縮約演算（オプション）
        axes: Vec<usize>,            // 縮約軸
        keepdim: bool,               // 次元を維持するか
    },

    // 構造演算
    Concat { inputs: Vec<TensorRef>, axis: usize },
}
```

**注意**: パディング演算はView機構に統合されています（`View::Padded`を参照）。

### MapReduce演算の統一

全ての計算演算がMapReduce variantで統一的に表現されます。

| 旧表現 | 新表現（MapReduce） |
|--------|-------------------|
| `Elementwise { op }` | `reduce_op: None, axes: []` |
| `FusedElementwise { expr }` | `reduce_op: None, axes: []` |
| `Cast { dtype }` | `expr: Cast(Wildcard("0"), dtype), reduce_op: None, axes: []` |
| `Reduce { op, axes }` | `expr: Wildcard("0"), reduce_op: Some(op)` |
| `FusedElementwiseReduce` | `expr + reduce_op: Some(op)` |

### View構造

メモリレイアウトを表現するenum。

```rust
pub enum View {
    // 線形アクセス（連続メモリ、ストライドアクセス）
    Linear {
        shape: Vec<Expr>,
        strides: Vec<Expr>,
        offset: Expr,
    },

    // インデックス式アクセス（複雑な変換）
    IndexExpr {
        shape: Vec<Expr>,
        index_expr: Expr,
    },

    // パディング付きView
    Padded {
        inner: Box<View>,              // 内側のView
        padding: Vec<(Expr, Expr)>,    // 各軸の(前, 後)パディング
        default_value: PadValue,       // 境界外の値（Zero, One, NegInf）
    },

    // 条件付きマスクView
    Masked {
        inner: Box<View>,              // 内側のView
        condition: Expr,               // 条件式（非0で有効、0でデフォルト値）
        default_value: PadValue,       // 条件が偽の場合の値
    },
}
```

**View::Padded**はパディング操作を他のView操作と統一的に扱うための設計です。
境界外アクセス時は`default_value`に応じた値が返されます:
- `PadValue::Zero`: 0.0（Sum演算の単位元）
- `PadValue::One`: 1.0（Prod演算の単位元）
- `PadValue::NegInf`: -∞（Max演算の単位元）

**View::Masked**は任意のExpr条件に基づいたマスク操作を表現します。
用途例:
- Attention mask（三角形マスク）: `Idx(0).le(Idx(1))`
- スパースパターン（偶数インデックスのみ）: `Idx(0).rem(2).eq_expr(0)`
- 任意の境界条件

### Expr（シェイプ式）

テンソルのシェイプ、ストライド、インデックス計算を表現する式。

**プリミティブ演算（AstNode同様の最小性原則）:**
- 算術: `Const`, `Idx`, `Add`, `Sub`, `Mul`, `Div`, `Rem`
- 比較: `Lt`（小なり）
- 論理: `Not`（否定）, `And`（論理積）

**導出演算（ヘルパーメソッド）:**
- 比較: `gt` = `b.lt(a)`, `le` = `!b.lt(a)`, `ge` = `!a.lt(b)`
- 等価: `eq_expr` = `a.le(b).and(b.le(a))`, `ne_expr` = `!a.eq_expr(b)`
- 論理: `or` = `!(!a).and(!b)`

比較・論理演算は0/1の整数値を返し、条件演算子として乗算で利用可能:
```rust
// 条件が真なら値を使用、偽なら0: value * condition
let result = value * condition;  // condition は 0 or 1
```

## Eager Fusion

演算呼び出し時に即座に融合判定を行い、可能であれば融合演算に変換。

### 融合パターン

| 親演算 | 子演算 | 融合可能 |
|--------|--------|----------|
| Elementwise MapReduce | Elementwise MapReduce | ○ |
| Elementwise MapReduce | Reduce MapReduce | ○ |
| Reduce MapReduce | * | × |

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

勾配関数のインターフェース。FloatDType（f32, f64）に対してジェネリックです。

```rust
pub trait GradFn<T: FloatDType>: Send + Sync {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>>;
    fn inputs(&self) -> Vec<Tensor<T, DimDyn>>;
    fn name(&self) -> &'static str;
}
```

### 型安全なGradFn構造体

GradFn構造体は`Tensor<T, D>`（型安全な次元）で入力を保持し、トレイト実装時のみ`DimDyn`に変換します。

```rust
// 例: PadBackward<T, D>
pub struct PadBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,           // 型安全に保持
    padding: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> GradFn<T> for PadBackward<T, D> {
    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone().into_dyn()]  // ここでのみ変換
    }
    // ...
}
```

**注意**: pad/slice/concat演算はFloatDType（f32, f64）専用です。整数型テンソルでは利用できません。

## 演算の分類

### primops（プリミティブ演算）

最小限の基本演算。これらの組み合わせで全ての演算を表現する。

**型制約**:
- 初期化演算: f32, f64をサポート
- 二項演算: NumericDType（f32, f64）をサポート
- 単項演算: FloatDType（f32, f64）をサポート
- 勾配追跡: f32, f64（FloatDTypeAutograd）

#### 初期化
| 演算 | 説明 | 型 |
|------|------|-----|
| `Const` | 定数テンソル | f32, f64 |
| `ConstFill` | 定数値で埋める | f32, f64 |
| `Rand` | 一様乱数 [0, 1) | f32, f64 |
| `Arange` | 連番テンソル | f32 |

#### 要素ごとの演算（二項）
| 演算 | 説明 | 型制約 |
|------|------|--------|
| `Add` | 加算 | NumericDType |
| `Mul` | 乗算 | NumericDType |
| `Max` | 最大値 | NumericDType |
| `Idiv` | 整数除算 | IntegerDType |
| `Rem` | 剰余 | IntegerDType |

#### ビット演算
| 演算 | 説明 | 型制約 |
|------|------|--------|
| `BitAnd` | ビットAND | IntegerDType |
| `BitOr` | ビットOR | IntegerDType |
| `BitXor` | ビットXOR | IntegerDType |
| `BitNot` | ビットNOT | IntegerDType |
| `Shl` | 左シフト | IntegerDType |
| `Shr` | 右シフト | IntegerDType |

#### 要素ごとの演算（単項）
| 演算 | 説明 | 型制約 |
|------|------|--------|
| `Neg` | 否定 (-x) | NumericDType |
| `Recip` | 逆数 (1/x) | FloatDType |
| `Sqrt` | 平方根 | FloatDType |
| `Log2` | 2を底とする対数 | FloatDType |
| `Exp2` | 2のべき乗 | FloatDType |
| `Sin` | 正弦 | FloatDType |
| `Floor` | 床関数（非微分可能、勾配=0） | FloatDType |

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
| `Pad` | パディング（View::Padded経由） |
| `Slice` | スライス（View::Linear経由、ゼロコピー） |
| `Concat` | テンソル結合（複数入力を条件分岐で処理） |
| `Unfold` | スライディングウィンドウ（im2col用、View::IndexExpr経由） |
| `Fold` | Unfoldの逆操作（col2im、勾配計算用、slice+pad+sumで実装） |

#### 特殊
| 演算 | 説明 |
|------|------|
| `Clone` | 分岐点（バッファコピー） |
| `Cast` | 型変換（MapReduceとして実装、融合可能） |

---

### hlops（高級演算）

primopsの組み合わせで表現される演算。f32, f64両方でサポート。

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
// 静的次元（f32がデフォルト）
let zeros = Tensor::<f32, Dim2>::zeros([3, 4]);
let ones = Tensor::<f32, Dim2>::ones([3, 4]);
let full = Tensor::<f32, Dim2>::full([3, 4], 2.5);
let input = Tensor::<f32, Dim2>::input("x", [3, 4]);

// 動的次元
let zeros = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4, 5]);
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

// 型変換（MapReduceとして実装、融合可能）
let f: Tensor<f32, Dim2> = i.cast();
let i: Tensor<i32, Dim2> = f.cast();

// 形状変更（型安全）
let b: Tensor<f32, Dim3> = a.unsqueeze(0);   // Dim2 → Dim3
let c: Tensor<f32, Dim1> = b.squeeze(0);     // Dim2 → Dim1
let d: Tensor<f32, Dim2> = a.pad(&[(1, 1), (2, 2)], PadValue::Zero); // Dim2 → Dim2

// スライス（ゼロコピー、View経由）
let sliced = a.slice(&[(1, 3), (0, 2)]);     // 範囲指定で部分テンソル取得

// 結合
let combined = Tensor::concat(&[&a, &b], 0); // axis=0で結合
```

### 型安全な形状操作

`Dimension`トレイトの関連型を使用した型安全なAPI:

| 演算 | 入力次元 | 出力次元 | 説明 |
|------|----------|----------|------|
| `squeeze(dim)` | `D` | `D::Smaller` | 指定次元を削除（size=1必須） |
| `unsqueeze(dim)` | `D` | `D::Larger` | 指定位置に次元追加 |
| `pad(padding, value)` | `D` | `D` | パディング（次元数保持） |
| `slice(ranges)` | `D` | `D` | スライス（次元数保持、ゼロコピー） |
| `concat(tensors, axis)` | `&[&Tensor<T, D>]` | `D` | 静的メソッド、テンソル結合 |
| `contiguous()` | `D` | `D` | メモリレイアウト正規化 |
| `flatten()` | `D` | `Dim<1>` | 1次元に展開 |
| `reshape([...])` | `D` | `Dim<M>` | 静的形状への変換 |

### 勾配追跡

勾配関連の操作はFloatDTypeAutograd（f32, f64）テンソルで利用可能です。

```rust
// 勾配追跡を有効化（f32）
let x = Tensor::<f32, Dim2>::ones([2, 2]).set_requires_grad(true);

// f64でも同様に使用可能
let x = Tensor::<f64, Dim2>::ones([2, 2]).set_requires_grad(true);

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
| `z = Pad(a, padding)` | ∂L/∂a = slice(∂L/∂z, padding位置) |
| `z = Slice(a, ranges)` | ∂L/∂a = pad_zero(∂L/∂z, 逆範囲) |
| `z = Concat([a,b,...], axis)` | ∂L/∂a = slice(∂L/∂z, aの範囲), ... |
| `z = Unfold(a)` | ∂L/∂a = fold(∂L/∂z) |
| `z = Fold(a)` | ∂L/∂a = unfold(∂L/∂z) |

### 融合演算の勾配

MapReduce演算の勾配はシンボリック微分により勾配を計算。

```rust
// MapReduce演算の勾配
// - AstNode式を各入力Wildcardに対してシンボリック微分
// - 導出した微分式をテンソル値で評価して勾配を計算

// Reduce付きMapReduce演算
// - まず勾配をReduce前の形状に展開（unsqueeze + expand）
// - 次にElementwise部分のシンボリック微分を適用
```

## モジュール構成

```
src/tensor/
├── mod.rs          # Tensor構造体、TensorInner、GradFn
├── dtype.rs        # TensorDType、NumericDType、FloatDType等のトレイト
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
    ├── binary.rs   # 二項演算（Add, Mul, Max, Idiv, Rem）
    ├── bitwise.rs  # ビット演算（IntegerDType専用）
    ├── grad.rs     # 勾配関数
    ├── init.rs     # 初期化（zeros, ones, full）
    ├── movement/   # 形状変更演算
    │   ├── mod.rs
    │   ├── core.rs     # pad, slice, squeeze, unsqueeze, reshape等
    │   ├── unfold.rs   # unfold1d/2d/3d
    │   ├── fold.rs     # fold1d/2d/3d（unfoldの逆操作）
    │   ├── backward.rs # 勾配関数
    │   └── tests.rs
    ├── reduce.rs   # 縮約演算
    └── unary.rs    # 単項演算
```

## TensorLowerer

TensorからASTへの変換を行うLowerer。

```rust
use harp::tensor::lowerer::{TensorLowerer, lower_tensor};

let a = Tensor::<f32, Dim2>::input("a", [2, 3]);
let b = Tensor::<f32, Dim2>::input("b", [2, 3]);
let c = &a + &b;

// TensorLowererを使用
let mut lowerer = TensorLowerer::new();
let ast = lowerer.lower(&c.clone().into_dyn());

// または簡易関数を使用
let ast = lower_tensor(&c.clone().into_dyn());
```

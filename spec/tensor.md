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

| 型 | TensorDType | NumericDType | FloatDType | IntegerDType | ComplexDType |
|----|-------------|--------------|------------|--------------|--------------|
| f32, f64 | ○ | ○ | ○ | × | × |
| i8, i16, i32, i64 | ○ | ○ | × | ○ (Signed) | × |
| u8, u16, u32, u64 | ○ | ○ | × | ○ (Unsigned) | × |
| bool | ○ | × | × | × | × |
| Complex32, Complex64 | ○ | ○ | × | × | ○ |

### ComplexDType トレイト

複素数型のためのトレイトです。

```rust
pub trait ComplexDType: NumericDType {
    type Real: FloatDType;
    const I: Self;  // 虚数単位
    fn new(re: Self::Real, im: Self::Real) -> Self;
    fn re(&self) -> Self::Real;
    fn im(&self) -> Self::Real;
    fn conj(&self) -> Self;
    fn abs(&self) -> Self::Real;
    fn arg(&self) -> Self::Real;
    fn norm_sqr(&self) -> Self::Real;
}
```

複素数は独自の `Complex<T>` 構造体として実装されています（orphan rules回避のため）:

```rust
#[repr(C)]  // Interleaved layout: [re, im, re, im, ...]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;
```

**NumericDType**: 数値型の基底トレイト（初期化定数を含む）

```rust
pub trait NumericDType: TensorDType {
    const ZERO: Self;
    const ONE: Self;
    fn to_literal(val: Self) -> Literal;
}
```

すべての数値型がこのトレイトを実装しており、`zeros`, `ones`, `full`, `input` などの初期化メソッドが統一的に利用可能です。

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

**注意**: パディング演算はView機構に統合されています（`View::Masked`を参照）。

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

    // 条件付きマスクView（パディングもこれで表現）
    Masked {
        inner: Box<View>,              // 内側のView
        condition: Expr,               // 条件式（非0で有効、0でデフォルト値）
        default_value: PadValue,       // 条件が偽の場合の値
    },
}
```

**View::Masked**は任意のExpr条件に基づいたマスク操作を表現します。
条件が非0のとき内側のViewからロードし、0のとき`default_value`に応じた値が返されます:
- `PadValue::Zero`: 0.0（Sum演算の単位元）
- `PadValue::One`: 1.0（Prod演算の単位元）
- `PadValue::NegInf`: -∞（Max演算の単位元）

**パディング**はMaskedで統一的に表現されます:
- `View::padded()`ヘルパーがMasked + IndexExprの組み合わせを生成
- 境界条件: `Idx(i) >= before[i] && Idx(i) < before[i] + inner_shape[i]`

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

### GradFn トレイト（レガシー）

勾配関数のインターフェース。FloatDType（f32, f64）に対してジェネリックです。

```rust
pub trait GradFn<T: FloatDType>: Send + Sync {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>>;
    fn inputs(&self) -> Vec<Tensor<T, DimDyn>>;
    fn name(&self) -> &'static str;
}
```

### GradFnTyped トレイト（新システム）

静的次元付きの勾配関数インターフェース。次元情報をコンパイル時に保持します。

```rust
pub trait GradFnTyped<T: FloatDType, D: Dimension>: Send + Sync {
    fn backward(&self, grad_output: &Tensor<T, D>);  // 戻り値なし、内部で伝播
    fn name(&self) -> &'static str;
}
```

**設計上の違い:**
- レガシー `GradFn<T>`: `backward()` がグラデーションの `Vec` を返し、トラバーサル側で伝播を管理
- 新 `GradFnTyped<T, D>`: 各実装が自身の入力テンソルを保持し、`backward()` 内で直接 `backward_with_typed()` を呼び出して勾配伝播

### AutogradMetaTyped（新システム）

静的次元付きの自動微分メタデータ。

```rust
pub struct AutogradMetaTyped<T: FloatDType, D: Dimension> {
    grad: RwLock<Option<Arc<Tensor<T, D>>>>,
    grad_fn: Option<Arc<dyn GradFnTyped<T, D>>>,
}
```

### Tensor構造体のautograd

`Tensor<T, D>` は両方のシステムをサポートします:

```rust
pub struct Tensor<T: TensorDType = f32, D: Dimension = DimDyn> {
    inner: Arc<TensorInner>,
    autograd_typed: Option<Arc<dyn Any + Send + Sync>>,  // AutogradMetaTyped<T, D>
    _dtype: PhantomData<T>,
    _dim: PhantomData<D>,
}
```

**新システムのメソッド:**
- `set_requires_grad_typed(bool)`: 型付き勾配追跡を有効化
- `requires_grad_typed()`: 型付き勾配追跡が有効かチェック
- `backward_typed()`: 型付き逆伝播開始
- `backward_with_typed(grad)`: 型付き勾配で逆伝播
- `grad_typed()`: 型付き勾配を取得

### 型安全なGradFn構造体（レガシー）

レガシーGradFn構造体は`Tensor<T, D>`で入力を保持し、トレイト実装時のみ`DimDyn`に変換します。

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

### 型安全なGradFnTyped構造体（新システム）

新システムでは次元変換が静的に追跡されます:

```rust
// 同次元演算: AddBackwardTyped<T, D>
pub struct AddBackwardTyped<T: FloatDType, D: Dimension> {
    lhs: Tensor<T, D>,
    rhs: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for AddBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.lhs.requires_grad_typed() {
            self.lhs.backward_with_typed(grad_output.clone());
        }
        if self.rhs.requires_grad_typed() {
            self.rhs.backward_with_typed(grad_output.clone());
        }
    }
}

// 次元減少演算: SumBackwardTyped<T, D>
// 入力: D, 出力: D::Smaller
pub struct SumBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    axis: usize,
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D::Smaller> for SumBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Smaller>) {
        // unsqueeze + expand で元の次元に戻す
        // ...
        self.input.backward_with_typed(grad_input);
    }
}
```

**注意**: pad/slice/concat演算はFloatDType（f32, f64）専用です。整数型テンソルでは利用できません。

### Autograd移行状況

新しい`GradFnTyped<T, D>`システムへの移行状況:

**新システムに接続済み:**
- Binary ops: Add, Mul（f32, f64）
- Scalar ops: ScalarAdd, ScalarMul（f32, f64）
- Unary ops: Neg
- Reduce ops: Sum, Mean, Max
- Movement ops: pad, slice, squeeze, unsqueeze, reshape, reshape_dyn, permute, transpose, expand, concat

**複素数自動微分（ComplexGradFn）:**
- 四則演算: Add, Mul, Neg, Recip（Sub, Divは合成経由）
- 複素共役: conj
- 超越関数: exp, ln, sqrt, sin, cos

**レガシーシステムのみ:**
- unfold1d, unfold2d, unfold3d（+ dilated versions）
- fold1d, fold2d, fold3d（+ dilated versions）
- maximum
- gather_with_grad（ScatterAddによるGatherBackward実装）

**次元変換の取り扱い:**
- `try_into_dim()`/`into_dimensioned()`: `autograd_typed`を保持
- `into_dyn()`: `autograd_typed`を保持しない（レガシーシステムとの互換性のため）
- `backward_with_typed()`: 次元型が一致しない場合、DimDynへのフォールバックを試行

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
| `Pad` | パディング（View::Masked経由） |
| `Slice` | スライス（View::Linear経由、ゼロコピー） |
| `Concat` | テンソル結合（複数入力を条件分岐で処理） |
| `Unfold` | スライディングウィンドウ（im2col用、View::IndexExpr経由、dilation対応） |
| `Fold` | Unfoldの逆操作（col2im、勾配計算用、slice+pad+sumで実装、dilation対応） |
| `Gather` | インデックステンソルに基づく要素収集（View::IndexExpr + Expr::LoadIndex経由） |
| `ScatterAdd` | インデックス指定位置への累積加算（AtomicAdd使用、GatherBackward用） |

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
| `nearest1d(size)` | arange + floor + gather（NCW形式、Dim3） |
| `nearest2d(size)` | arange + floor + gather（NCHW形式、Dim4） |
| `nearest3d(size)` | arange + floor + gather（NCDHW形式、Dim5） |
| `linear1d(size)` | arange + floor + gather×2 + 線形補間（NCW形式、Dim3） |
| `bilinear2d(size)` | arange + floor + gather×4 + 双線形補間（NCHW形式、Dim4） |
| `trilinear3d(size)` | arange + floor + gather×8 + 三線形補間（NCDHW形式、Dim5） |

### 複素数演算（hlops/complex）

複素数テンソル（Complex32, Complex64）用の演算です。

#### 複素数プリミティブ演算

| 演算 | 説明 | 戻り値型 |
|------|------|----------|
| `z.real()` | 実部抽出 | `Tensor<T::Real>` |
| `z.imag()` | 虚部抽出 | `Tensor<T::Real>` |
| `z.conj()` | 複素共役 | `Tensor<T>` |
| `complex(re, im)` | 実部と虚部から構成 | `Tensor<Complex<T>>` |

#### 複素数四則演算

| 演算 | 数学的表現 |
|------|-----------|
| `(a+bi) + (c+di)` | `(a+c) + (b+d)i` |
| `(a+bi) - (c+di)` | `(a-c) + (b-d)i` |
| `(a+bi) * (c+di)` | `(ac-bd) + (ad+bc)i` |
| `(a+bi) / (c+di)` | `((ac+bd) + (bc-ad)i) / (c²+d²)` |
| `-(a+bi)` | `(-a) + (-b)i` |
| `1/(a+bi)` | `(a - bi) / (a²+b²)` |

#### 複素数超越関数

| 演算 | 数学的表現 |
|------|-----------|
| `exp(a+bi)` | `exp(a) * (cos(b) + i*sin(b))` |
| `ln(z)` | `ln|z| + i*arg(z)` |
| `sqrt(z)` | `sqrt(|z|) * exp(i*arg(z)/2)` |
| `sin(a+bi)` | `sin(a)*cosh(b) + i*cos(a)*sinh(b)` |
| `cos(a+bi)` | `cos(a)*cosh(b) - i*sin(a)*sinh(b)` |

**自動微分**: 複素数テンソルはWirtinger微分に基づく自動微分をサポートしています。詳細は「複素数自動微分」セクションを参照してください。

#### メモリレイアウト

複素数は **Interleaved layout** を使用します:
```
[re0, im0, re1, im1, re2, im2, ...]
```

Lowering時に複素数は実数2つに分解されます。

### 複素数自動微分（Wirtinger Derivatives）

複素数テンソルの自動微分はWirtinger微分に基づいています。実数値損失関数 L に対する複素変数 z = x + iy の勾配は:

```
∂L/∂z* = (1/2)(∂L/∂x + i·∂L/∂y)
```

この勾配がbackward時に伝播されます。

#### 複素数自動微分トレイト

複素数テンソル用の自動微分システムは実数用とは別のトレイト体系を使用します:

```rust
/// 複素数テンソル用の勾配関数トレイト
pub trait ComplexGradFn<T: FloatDType, D: Dimension>: Send + Sync
where
    Complex<T>: TensorDType,
{
    fn backward(&self, grad_output: &Tensor<Complex<T>, D>);
    fn name(&self) -> &'static str;
}

/// 複素数テンソル用の自動微分メタデータ
pub struct ComplexAutogradMeta<T: FloatDType, D: Dimension>
where
    Complex<T>: TensorDType,
{
    grad: RwLock<Option<Arc<Tensor<Complex<T>, D>>>>,
    grad_fn: Option<Arc<dyn ComplexGradFn<T, D>>>,
}
```

#### 演算別の勾配公式

| 演算 | 正則性 | ∂f/∂z | backward実装 |
|------|--------|-------|--------------|
| `a + b` | ✅ | 1 | `grad_a = grad_b = grad_out` |
| `a * b` | ✅ | b | `grad_a = grad_out * conj(b)` |
| `a / b` | ✅ | 1/b | `grad_a = grad_out * conj(1/b)` |
| `conj(a)` | ❌ | 0 | `grad_a = conj(grad_out)` |
| `exp(a)` | ✅ | exp(a) | `grad_a = grad_out * conj(exp(a))` |
| `ln(a)` | ✅ | 1/a | `grad_a = grad_out / conj(a)` |
| `sqrt(a)` | ✅ | 1/(2√a) | `grad_a = grad_out / (2 * conj(sqrt(a)))` |
| `sin(a)` | ✅ | cos(a) | `grad_a = grad_out * conj(cos(a))` |
| `cos(a)` | ✅ | -sin(a) | `grad_a = -grad_out * conj(sin(a))` |

**注意**: 正則関数では ∂f/∂z* = 0 なので、勾配計算は実数の場合と類似しますが、共役が必要です。

#### 使用例

```rust
use harp::tensor::{Tensor, Dim2, Complex32, Complex};

let a = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32))
    .set_requires_grad(true);
let b = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(3.0f32, 4.0f32))
    .set_requires_grad(true);

let c = &a * &b;
assert!(c.requires_grad());

// backward()の呼び出しにより勾配が計算される
// a.grad()で勾配を取得可能
```

## API

### テンソル作成

```rust
// 静的次元（f32がデフォルト）
let zeros = Tensor::<f32, Dim2>::zeros([3, 4]);
let ones = Tensor::<f32, Dim2>::ones([3, 4]);
let full = Tensor::<f32, Dim2>::full([3, 4], 2.5);
let input = Tensor::<f32, Dim2>::input("x", [3, 4]);

// 連番（1D専用）
let seq = Tensor::<f32, Dim1>::arange(5);  // [0.0, 1.0, 2.0, 3.0, 4.0]

// 動的次元
let zeros = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4, 5]);

// 複素数テンソル
let z = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
let w = Tensor::<Complex64, Dim2>::zeros([2, 3]);
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

// 複素数演算
use harp::tensor::hlops::{ComplexExp, ComplexSin};
let z = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(1.0f32, 2.0f32));
let w = Tensor::<Complex32, Dim2>::full([2, 3], Complex::new(0.5f32, 1.0f32));
let sum = &z + &w;           // 複素数加算
let prod = &z * &w;          // 複素数乗算
let re = z.real();           // 実部抽出
let im = z.imag();           // 虚部抽出
let conj = z.conj();         // 複素共役
let exp_z = z.exp();         // 複素指数関数

// 形状変更（型安全）
let b: Tensor<f32, Dim3> = a.unsqueeze(0);   // Dim2 → Dim3
let c: Tensor<f32, Dim1> = b.squeeze(0);     // Dim2 → Dim1
let d: Tensor<f32, Dim2> = a.pad(&[(1, 1), (2, 2)], PadValue::Zero); // Dim2 → Dim2

// スライス（ゼロコピー、View経由）
let sliced = a.slice(&[(1, 3), (0, 2)]);     // 範囲指定で部分テンソル取得

// 結合
let combined = Tensor::concat(&[&a, &b], 0); // axis=0で結合

// Gather（インデックスに基づく要素収集）
let data = Tensor::<f32, Dim2>::ones([4, 5]);
let index = Tensor::<i64, Dim2>::zeros([3, 5]);
let gathered = data.gather(0, &index); // output[i][j] = input[index[i][j]][j]

// Gather with gradient（勾配追跡可能）
let data = Tensor::<f32, Dim2>::ones([4, 5]).set_requires_grad(true);
let gathered = data.gather_with_grad(0, &index); // 逆伝播でScatterAddを使用
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
| `gather(dim, index)` | `D` | `D` | インデックステンソルに基づく要素収集 |
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
| `z = Gather(a, dim, idx)` | ∂L/∂a = scatter_add(zeros, dim, idx, ∂L/∂z) |

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
├── complex.rs      # Complex<T>構造体、ComplexDTypeトレイト
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
│   ├── complex.rs      # 複素数分解ヘルパー
│   ├── expr_builder.rs
│   └── helpers.rs
├── hlops/          # 高級演算
│   ├── activation.rs
│   ├── arithmetic.rs
│   ├── complex_arithmetic.rs    # 複素数四則演算
│   ├── complex_transcendental.rs # 複素数超越関数
│   ├── interpolate/             # 補間
│   │   ├── mod.rs
│   │   ├── nearest.rs           # 最近傍補間（nearest1d/2d/3d）
│   │   └── linear.rs            # 線形補間（linear1d, bilinear2d, trilinear3d）
│   ├── linalg.rs
│   ├── reduction.rs
│   └── transcendental.rs
└── primops/        # プリミティブ演算
    ├── binary.rs   # 二項演算（Add, Mul, Max, Idiv, Rem）
    ├── bitwise.rs  # ビット演算（IntegerDType専用）
    ├── complex.rs  # 複素数演算（real, imag, conj, complex_from_parts）
    ├── grad.rs     # 勾配関数
    ├── init.rs     # 初期化（zeros, ones, full, arange）
    ├── movement/   # 形状変更演算
    │   ├── mod.rs
    │   ├── core.rs     # pad, slice, squeeze, unsqueeze, reshape等
    │   ├── gather.rs   # gather演算（インデックステンソルに基づく要素収集）
    │   ├── scatter.rs  # scatter_add演算（AtomicAdd使用、GatherBackward用）
    │   ├── unfold.rs   # unfold1d/2d/3d_dilated（stride, dilation対応）
    │   ├── fold.rs     # fold1d/2d/3d_dilated（unfoldの逆操作、stride, dilation対応）
    │   ├── backward.rs # 勾配関数（GatherBackward等）
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

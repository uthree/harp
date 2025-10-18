# AST (Abstract Syntax Tree) 仕様

## 概要

harpのASTは、数値計算の式を木構造で表現するためのデータ構造です。GPU/CPUでの並列計算に最適化されたコード生成を行うために、型情報やメモリアクセスパターンを明示的に保持します。

## データ構造

### AstNode

ASTの各ノードを表す構造体です。

```rust
pub struct AstNode {
    pub op: AstOp,
    pub dtype: DType,
}
```

- `op`: ノードが表す演算や値
- `dtype`: ノードの結果の型

#### メソッド

- `children(&self) -> Vec<&AstNode>`: 子ノードへの参照をVecで返す
- `with_children(&self, new_children: Vec<AstNode>) -> AstNode`: 子ノードを置き換えた新しいAstNodeを作成
- `new(op: AstOp) -> Self`: 型未定義の新しいノードを作成
- `with_dtype(mut self, dtype: DType) -> Self`: 型を設定

### AstOp

ASTノードが表す演算の種類を定義するenum。

```rust
pub enum AstOp {
    // 定数・変数
    Const(ConstValue),
    Var(String),

    // 二項演算
    Add(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Max(Box<AstNode>, Box<AstNode>),
    Idiv(Box<AstNode>, Box<AstNode>),  // 整数除算
    Rem(Box<AstNode>, Box<AstNode>),   // 剰余

    // 単項演算
    Neg(Box<AstNode>),
    Recip(Box<AstNode>),
    Sqrt(Box<AstNode>),
    Sin(Box<AstNode>),
    Log2(Box<AstNode>),
    Exp2(Box<AstNode>),
    Cast(Box<AstNode>, DType),

    // メモリ操作
    Load {
        target: Box<AstNode>,
        offset: Box<AstNode>,
        size: Box<AstNode>,
    },
    Store {
        target: Box<AstNode>,
        offset: Box<AstNode>,
        value: Box<AstNode>,
    },

    // パターンマッチング用
    Capture(isize),
}
```

## 型システム (DType)

### 基本型

```rust
pub enum DType {
    Unknown,  // 型未定義（プレースホルダー）
    None,     // void型
    Isize,    // 符号付き整数
    Usize,    // 符号なし整数
    F32,      // 32ビット浮動小数点数
    Bool,     // 真偽値

    Ptr {
        pointee: Box<DType>,
        mutable: bool,
    },
    Vec(Box<DType>, usize),
}
```

### Ptr型のメモリアクセス制御

GPU並列計算では、メモリアクセスパターンが重要です。`Ptr`型は`mutable`フラグを持ち、メモリ競合の可能性を型レベルで管理します。

#### mutableフラグの意味

- `mutable: false` (デフォルト)
  - **読み取り専用**ポインタ
  - 複数のGPUスレッドから**同時に読み込んでも安全**
  - メモリ競合が発生しない
  - OpenCL/CUDAの`const`修飾子に相当

- `mutable: true`
  - **書き込み可能**なポインタ
  - 排他制御が必要
  - 複数スレッドからの同時書き込みはメモリ競合を引き起こす可能性がある

#### 使用例

```rust
// 読み取り専用バッファ（複数GPUスレッドから安全に読み込み可能）
let readonly_buffer = DType::F32.ptr();

// 書き込み可能バッファ（排他制御が必要）
let writable_buffer = DType::F32.ptr_mut();

// パターンマッチで判定
match buffer_type {
    DType::Ptr { mutable: false, .. } => {
        // 並列読み込み可能 → 最適化可能
    }
    DType::Ptr { mutable: true, .. } => {
        // 排他制御が必要 → アトミック操作など
    }
    _ => {}
}
```

#### ヘルパーメソッド

```rust
impl DType {
    /// 読み取り専用ポインタ型を作成（デフォルト）
    /// GPU並列処理で安全に同時読み込みができる
    pub fn ptr(self) -> Self;

    /// 書き込み可能なポインタ型を作成
    /// 排他制御が必要
    pub fn ptr_mut(self) -> Self;

    /// ベクトル型を作成
    pub fn vec(self, size: impl Into<usize>) -> Self;
}
```

### Vec型

SIMD演算用の固定長ベクトル型。

```rust
DType::Vec(Box::new(DType::F32), 4)  // float4
DType::Vec(Box::new(DType::F32), 8)  // float8
```

## 定数値 (ConstValue)

```rust
pub enum ConstValue {
    Isize(isize),
    Usize(usize),
    F32(f32),
    Bool(bool),
}
```

## ヘルパー関数

### 二項演算子

```rust
pub fn add(lhs: AstNode, rhs: AstNode) -> AstNode;
pub fn mul(lhs: AstNode, rhs: AstNode) -> AstNode;
pub fn max(lhs: AstNode, rhs: AstNode) -> AstNode;
pub fn idiv(lhs: AstNode, rhs: AstNode) -> AstNode;
pub fn rem(lhs: AstNode, rhs: AstNode) -> AstNode;
```

### 単項演算子

```rust
pub fn neg(operand: AstNode) -> AstNode;
pub fn recip(operand: AstNode) -> AstNode;
pub fn sqrt(operand: AstNode) -> AstNode;
pub fn sin(operand: AstNode) -> AstNode;
pub fn log2(operand: AstNode) -> AstNode;
pub fn exp2(operand: AstNode) -> AstNode;
```

### 定数コンストラクタ

```rust
pub fn const_isize(value: isize) -> AstNode;
pub fn const_usize(value: usize) -> AstNode;
pub fn const_f32(value: f32) -> AstNode;
pub fn const_bool(value: bool) -> AstNode;
```

## 演算子オーバーロード

AstNodeは標準的な演算子をサポートしています。

```rust
use harp::ast::*;

let a = const_f32(1.0);
let b = const_f32(2.0);
let c = const_f32(3.0);

// 演算子を使った記述
let result = (a + b) * c - const_f32(1.0);

// ヘルパー関数を使った記述（上記と等価）
let result = add(mul(add(a, b), c), neg(const_f32(1.0)));
```

## パターンマッチング

### AstRewriteRule

AST変換ルールを定義する構造体。

```rust
pub struct AstRewriteRule {
    pattern: AstNode,
    rewriter: Box<dyn Fn(&[AstNode]) -> AstNode>,
    condition: Box<dyn Fn(&[AstNode]) -> bool>,
}
```

### ast_rule! マクロ

リライトルールを簡潔に記述するためのマクロ。

```rust
// 条件なしのルール
let rule = ast_rule!(|a, b| mul(capture(0), capture(1)) => mul(b.clone(), a.clone()));

// 条件付きのルール
let rule = ast_rule!(
    |a, b| add(capture(0), capture(1)) => const_f32(a_val + b_val),
    if |caps: &[AstNode]| {
        matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
    }
);
```

### AstRewriter

複数のリライトルールを管理し、順番に適用するための構造体。

```rust
pub struct AstRewriter {
    rules: Vec<Rc<AstRewriteRule>>,
}
```

#### メソッド

- `new() -> Self`: 新しいリライターを作成
- `add_rule(&mut self, rule: Rc<AstRewriteRule>)`: ルールを追加
- `apply(&self, node: &AstNode) -> AstNode`: 全ルールを順番に適用
- `apply_until_fixed(&self, node: &AstNode) -> AstNode`: 変化がなくなるまで繰り返し適用
- `merge(&self, other: &AstRewriter) -> AstRewriter`: 別のリライターと融合

### ast_rewriter! マクロ

複数のルールを持つリライターを作成するマクロ。

```rust
let rewriter = ast_rewriter! {
    // 二重否定を除去
    ast_rule!(|x| neg(neg(capture(0))) => x.clone()),

    // 定数の加算を畳み込む
    ast_rule!(
        |a, b| add(capture(0), capture(1)) => {
            if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                (&a.op, &b.op)
            {
                const_f32(av + bv)
            } else {
                add(a.clone(), b.clone())
            }
        },
        if |caps: &[AstNode]| {
            matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
        }
    ),
};
```

### リライターの合成

`+`演算子でリライターを合成できます。

```rust
let rewriter1 = ast_rewriter! { /* ... */ };
let rewriter2 = ast_rewriter! { /* ... */ };

// 2つのリライターを合成
let combined = &rewriter1 + &rewriter2;
```

## 最適化器 (Optimizer)

### AstOptimizer トレイト

AST最適化を抽象化するトレイト。

```rust
pub trait AstOptimizer {
    fn apply(&self, ast: &AstNode) -> AstNode;

    fn compose(self, other: impl AstOptimizer + 'static) -> ComposedOptimizer
    where
        Self: Sized + 'static;
}
```

#### メソッド

- `apply(&self, ast: &AstNode) -> AstNode`: ASTに最適化を適用
- `compose(self, other: impl AstOptimizer + 'static) -> ComposedOptimizer`: 他のOptimizerと合成して新しいOptimizerを作成

#### composeメソッド

`compose`メソッドは、複数の最適化器を合成するための便利なメソッドです。

**デフォルト実装**: selfとotherを含む新しいComposedOptimizerを作成します。

**ComposedOptimizerでのオーバーライド**: 既存のoptimizer群に新しいoptimizerを追加します（新しいComposedOptimizerを作成するのではなく）。

これにより、メソッドチェーン形式で複数の最適化器を組み合わせることができます：

```rust
use harp::opt::AstOptimizer;
use harp::opt::ast::{RewriterOptimizer, ComposedOptimizer};

let opt1 = RewriterOptimizer::new(/* ... */);
let opt2 = RewriterOptimizer::new(/* ... */);
let opt3 = RewriterOptimizer::new(/* ... */);

// メソッドチェーンで合成
let composed = opt1.compose(opt2).compose(opt3);

// 従来の方法
let composed = ComposedOptimizer::new()
    .add_optimizer(opt1)
    .add_optimizer(opt2)
    .add_optimizer(opt3);
```

### RewriterOptimizer

AstRewriterを使った最適化器の実装。

```rust
pub struct RewriterOptimizer {
    rewriter: AstRewriter,
    apply_until_fixed: bool,
}
```

#### メソッド

- `new(rewriter: AstRewriter) -> Self`: 新しい最適化器を作成（1回だけ適用）
- `with_fixed_point(self) -> Self`: 不動点に達するまで繰り返し適用するモードを有効化
- `from_rewriters(rewriters: Vec<AstRewriter>) -> Self`: 複数のリライターを統合して最適化器を作成

#### 使用例

```rust
use harp::opt::ast::RewriterOptimizer;
use harp::ast::*;

// 定数畳み込みリライター
let rewriter = ast_rewriter! {
    ast_rule!(
        |a, b| add(capture(0), capture(1)) => {
            if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                (&a.op, &b.op)
            {
                const_f32(av + bv)
            } else {
                add(a.clone(), b.clone())
            }
        },
        if |caps: &[AstNode]| {
            matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
        }
    ),
};

// 最適化器を作成し、不動点まで適用
let optimizer = RewriterOptimizer::new(rewriter).with_fixed_point();

// (1.0 + 2.0) + (3.0 + 4.0)
let ast = add(
    add(const_f32(1.0), const_f32(2.0)),
    add(const_f32(3.0), const_f32(4.0)),
);

let result = optimizer.apply(&ast);  // const_f32(10.0) になる
```

### ComposedOptimizer

複数の最適化器を順番に適用する合成最適化器。

```rust
pub struct ComposedOptimizer {
    optimizers: Vec<Box<dyn AstOptimizer>>,
}
```

#### メソッド

- `new() -> Self`: 新しい合成最適化器を作成
- `add_optimizer<O: AstOptimizer + 'static>(self, optimizer: O) -> Self`: 最適化器を追加
- `from_optimizers(optimizers: Vec<Box<dyn AstOptimizer>>) -> Self`: 最適化器のリストから作成
- `compose(self, other: impl AstOptimizer + 'static) -> ComposedOptimizer`: 既存のoptimizer群に新しいoptimizerを追加（オーバーライド）

#### 使用例

```rust
use harp::opt::ast::{RewriterOptimizer, ComposedOptimizer};

// 二重否定除去
let opt1 = RewriterOptimizer::new(ast_rewriter! {
    ast_rule!(|x| neg(neg(capture(0))) => x.clone()),
});

// 定数畳み込み（不動点まで）
let opt2 = RewriterOptimizer::new(ast_rewriter! {
    ast_rule!(
        |a, b| add(capture(0), capture(1)) => {
            if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                (&a.op, &b.op)
            {
                const_f32(av + bv)
            } else {
                add(a.clone(), b.clone())
            }
        },
        if |caps: &[AstNode]| {
            matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
        }
    ),
}).with_fixed_point();

// 方法1: add_optimizerを使う
let optimizer = ComposedOptimizer::new()
    .add_optimizer(opt1)
    .add_optimizer(opt2);

// 方法2: composeメソッドを使う（より簡潔）
let optimizer = opt1.compose(opt2);

// -(-1.0) + -(-2.0)
let ast = add(neg(neg(const_f32(1.0))), neg(neg(const_f32(2.0))));
let result = optimizer.apply(&ast);  // const_f32(3.0) になる
```

**composeメソッドの利点**:

```rust
// チェーン形式で読みやすい
let optimizer = opt1.compose(opt2).compose(opt3).compose(opt4);

// add_optimizerの場合
let optimizer = ComposedOptimizer::new()
    .add_optimizer(opt1)
    .add_optimizer(opt2)
    .add_optimizer(opt3)
    .add_optimizer(opt4);
```

### 最適化の順序

**重要**: ComposedOptimizerは各最適化器を順番に1回だけ適用します。最適化の順序によって結果が変わる場合があります。

```rust
// 例: 最適化の順序が重要なケース

// パターン1: 定数畳み込み → 二重否定除去
let opt1 = ComposedOptimizer::new()
    .add_optimizer(constant_folding_optimizer)
    .add_optimizer(double_negation_optimizer);

// パターン2: 二重否定除去 → 定数畳み込み
let opt2 = ComposedOptimizer::new()
    .add_optimizer(double_negation_optimizer)
    .add_optimizer(constant_folding_optimizer);

// -(-2.0) * 3.0 の場合:
// パターン1: 定数畳み込みできず → 二重否定除去 → Mul(2.0, 3.0) のまま
// パターン2: 二重否定除去 → 定数畳み込みできず → Mul(2.0, 3.0) のまま

// 両方の最適化を完全に適用するには、with_fixed_pointを使うか、
// より強力な最適化パイプラインを構築する必要があります
```

## 使用例

### 基本的な式の構築

```rust
use harp::ast::*;

// (x + 2.0) * 3.0
let x = AstNode::new(AstOp::Var("x".to_string())).with_dtype(DType::F32);
let expr = (x + 2.0f32) * 3.0f32;
```

### 最適化ルールの適用

```rust
// 定数畳み込み最適化
let optimizer = ast_rewriter! {
    ast_rule!(
        |a, b| add(capture(0), capture(1)) => {
            if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                (&a.op, &b.op)
            {
                const_f32(av + bv)
            } else {
                add(a.clone(), b.clone())
            }
        },
        if |caps: &[AstNode]| {
            matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
        }
    ),
};

let expr = const_f32(1.0) + const_f32(2.0);
let optimized = optimizer.apply(&expr);  // const_f32(3.0) になる
```

### GPU用のバッファ型定義

```rust
// 入力バッファ（読み取り専用）
let input_buffer = DType::F32.ptr();

// 出力バッファ（書き込み可能）
let output_buffer = DType::F32.ptr_mut();

// カーネル引数の型シグネチャ例
// kernel void process(
//     __global const float* input,   // DType::F32.ptr()
//     __global float* output          // DType::F32.ptr_mut()
// )
```

## 設計方針

### 演算子の最小化（Minimal Operator Set）

**harpのASTは、演算子の種類を必要最小限に抑える設計思想を採用しています。**

この設計により、以下のメリットが得られます：

1. **パターンマッチングの簡素化**: 最適化ルールが少ない演算子のパターンだけを考慮すれば良い
2. **最適化の網羅性向上**: すべての演算が正規化された形で表現されるため、最適化の適用漏れが減る
3. **コード生成の簡素化**: バックエンドが対応すべき演算子の種類が少なくて済む
4. **保守性の向上**: 新機能の追加時に考慮すべきケースが減る

#### 演算子の正規化ルール

複雑な演算は、より基本的な演算の組み合わせで表現されます：

| 演算 | 通常の表現 | harpでの正規化表現 | 理由 |
|------|-----------|------------------|------|
| 減算 | `sub(a, b)` | `add(a, neg(b))` | 加算と否定だけで表現可能 |
| 除算 | `div(a, b)` | `mul(a, recip(b))` | 乗算と逆数だけで表現可能 |
| 余弦 | `cos(x)` | `sin(add(x, const(π/2)))` | 正弦関数と定数オフセットで表現可能 |
| 正接 | `tan(x)` | `mul(sin(x), recip(cos(x)))` | 正弦と余弦の比で表現可能 |
| べき乗 | `pow(a, b)` | `exp2(mul(b, log2(a)))` | 対数と指数で表現可能 |

#### 実装例

演算子オーバーロードによって、ユーザーコードでは通常の演算子を使用できますが、内部的には正規化された形に変換されます：

```rust
// ユーザーが書くコード
let result = a - b;

// 内部的には以下のように展開される
impl<T: Into<AstNode>> Sub<T> for AstNode {
    type Output = AstNode;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs_node = rhs.into();
        // 減算は add(a, neg(b)) として実装
        add(self, neg(rhs_node))
    }
}

// 除算も同様
impl<T: Into<AstNode>> Div<T> for AstNode {
    type Output = AstNode;

    fn div(self, rhs: T) -> Self::Output {
        let rhs_node = rhs.into();
        // 除算は mul(a, recip(b)) として実装
        mul(self, recip(rhs_node))
    }
}
```

#### 最適化への影響

演算子の正規化により、最適化ルールの記述が簡潔になります：

```rust
// 悪い例: 減算と加算の両方に対応する必要がある
ast_rule!(|a| sub(a, a) => const_f32(0.0));  // a - a = 0
ast_rule!(|a| add(a, neg(a)) => const_f32(0.0));  // a + (-a) = 0

// 良い例: 加算のパターンだけで十分（減算は内部的に add(a, neg(b)) に正規化されているため）
ast_rule!(|a| add(a, neg(a)) => const_f32(0.0));  // a + (-a) = 0
```

#### トレードオフ

この設計にはトレードオフも存在します：

**メリット:**
- 最適化ルールの数が減少
- パターンマッチングが簡単
- AST構造の一貫性が向上

**デメリット:**
- AST構造がやや複雑になる（`a - b` が `add(a, neg(b))` という3ノード構造になる）
- 特定のハードウェア命令（例: FMA = Fused Multiply-Add）への直接マッピングが難しくなる場合がある
  - ただし、これは後段の最適化パスで検出可能（`add(mul(a, b), c)` → FMA命令）

### 不変性

ASTノードは基本的に不変です。変換は常に新しいノードを生成します。

### 型安全性

- 二項演算子は両オペランドの型が一致することをassertで検証
- ポインタのmutableフラグにより、メモリアクセスパターンをコンパイル時に判定可能

### GPU最適化

- Ptr型のmutableフラグにより、バックエンドは以下の最適化が可能：
  - 読み取り専用バッファはキャッシュ最適化
  - 並列読み込みの安全性保証
  - OpenCL/CUDAの`const`修飾子への自動マッピング

### 拡張性

- 新しい演算子の追加が容易
- パターンマッチングによる柔軟な変換ルール定義
- リライターの合成により、複雑な最適化パイプラインを構築可能

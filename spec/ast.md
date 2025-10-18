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

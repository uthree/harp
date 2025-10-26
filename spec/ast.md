# AST (抽象構文木) 仕様

## 概要

harpプロジェクトのAST（Abstract Syntax Tree）モジュールは、数値計算を表現するための抽象構文木を提供します。

## ファイル構成

- `src/ast/mod.rs` - 型定義、型推論、子ノード取得などの主要機能
- `src/ast/ops.rs` - 演算子オーバーロード (`+`, `*`, `%`)
- `src/ast/helper.rs` - ヘルパー関数（数学関数など）

## データ型

### AstNode

ASTのノードを表すenum型。計算式を木構造で表現します。

```rust
pub enum AstNode {
    // 定数
    Const(Literal),

    // 二項演算
    Add(Box<AstNode>, Box<AstNode>),      // 加算
    Mul(Box<AstNode>, Box<AstNode>),      // 乗算
    Max(Box<AstNode>, Box<AstNode>),      // 最大値
    Rem(Box<AstNode>, Box<AstNode>),      // 剰余
    Idiv(Box<AstNode>, Box<AstNode>),     // 整数除算

    // 単項演算
    Recip(Box<AstNode>),                  // 逆数
    Sqrt(Box<AstNode>),                   // 平方根
    Log2(Box<AstNode>),                   // 対数（底2）
    Exp2(Box<AstNode>),                   // 指数（底2）
    Sin(Box<AstNode>),                    // 正弦

    // 型変換
    Cast(Box<AstNode>, DType),            // 型キャスト
}
```

### DType

データ型を表すenum型。

```rust
pub enum DType {
    Isize,                  // 符号付き整数
    Usize,                  // 符号なし整数（配列インデックス用）
    F32,                    // 浮動小数点数
    Ptr(Box<DType>),        // ポインタ（メモリバッファ用）
    Vec(Box<DType>, usize), // 固定サイズベクトル（SIMD用）
    Tuple(Vec<DType>),      // タプル
    Unknown,                // 型不明（型推論失敗時）
}
```

### Literal

リテラル値を表すenum型。

```rust
pub enum Literal {
    Isize(isize),
    Usize(usize),
    F32(f32),
}
```

## 主要機能

### 型変換 (Into/From)

数値型からLiteralへの自動変換をサポート：

```rust
// f32, isize, usizeからLiteralへの変換
let lit = Literal::from(3.14f32);
let lit: Literal = 42isize.into();
```

### 型推論

`Literal::dtype()` - リテラルの型を取得：

```rust
let lit = Literal::F32(3.14);
assert_eq!(lit.dtype(), DType::F32);
```

### 子ノードの取得

`AstNode::children()` - ASTノードの直接の子ノードを取得：

```rust
let node = AstNode::Add(Box::new(a), Box::new(b));
let children = node.children(); // Vec<&AstNode> を返す
```

### 型推論（再帰的）

`AstNode::infer_type()` - ASTノードの型を再帰的に推論：

```rust
let expr = AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into());
assert_eq!(expr.infer_type(), DType::F32);
```

**推論ルール：**
- `Const`: リテラルの型を返す
- `Cast`: 指定された型を返す
- 二項演算 (`Add`, `Mul`, `Max`, `Rem`, `Idiv`): 両辺の型が一致すればその型、異なれば`Unknown`
- `Recip`: オペランドの型を保持
- 数学関数 (`Sqrt`, `Log2`, `Exp2`, `Sin`): 常に`F32`を返す

## 演算子オーバーロード

`src/ast/ops.rs`で定義：

```rust
impl Add for AstNode  // +演算子
impl Mul for AstNode  // *演算子
impl Rem for AstNode  // %演算子
```

使用例：

```rust
let a = AstNode::Const(1.0f32.into());
let b = AstNode::Const(2.0f32.into());
let sum = a + b;  // AstNode::Add(...)
let product = sum * AstNode::Const(3.0f32.into());
```

## ヘルパー関数

`src/ast/helper.rs`で定義：

### 数学関数

```rust
pub fn max(a: AstNode, b: AstNode) -> AstNode
pub fn idiv(a: AstNode, b: AstNode) -> AstNode
pub fn recip(a: AstNode) -> AstNode
pub fn sqrt(a: AstNode) -> AstNode
pub fn log2(a: AstNode) -> AstNode
pub fn exp2(a: AstNode) -> AstNode
pub fn sin(a: AstNode) -> AstNode
pub fn cast(a: AstNode, dtype: DType) -> AstNode
```

使用例：

```rust
use harp::ast::helper::*;

let a = AstNode::Const(4.0f32.into());
let result = sqrt(a);  // AstNode::Sqrt(...)
```

## 使用例

### 基本的な式の構築

```rust
use harp::ast::{AstNode, DType};
use harp::ast::helper::*;

// (a + b) * sqrt(c)
let a = AstNode::Const(1.0f32.into());
let b = AstNode::Const(2.0f32.into());
let c = AstNode::Const(4.0f32.into());

let expr = (a + b) * sqrt(c);

// 型推論
assert_eq!(expr.infer_type(), DType::F32);

// 子ノードの取得
let children = expr.children(); // [(a + b), sqrt(c)]
assert_eq!(children.len(), 2);
```

### 型キャスト

```rust
let int_val = AstNode::Const(42isize.into());
let float_val = cast(int_val, DType::F32);
assert_eq!(float_val.infer_type(), DType::F32);
```

### 複雑な式

```rust
// sqrt((a + b) * c) % d
let a = AstNode::Const(1.0f32.into());
let b = AstNode::Const(2.0f32.into());
let c = AstNode::Const(3.0f32.into());
let d = AstNode::Const(5.0f32.into());

let expr = sqrt((a + b) * c) % d;
```

## 設計上の注意点

1. **所有権**: ASTノードは`Clone`トレイトを実装しており、複製が可能
2. **型安全性**: `DType`により型情報を保持し、型推論により静的な型チェックが可能
3. **拡張性**: 新しい演算や関数を追加する場合は、`AstNode` enumに追加し、対応する`infer_type()`のケースを実装
4. **定数生成**: `AstNode::Const(value.into())`の形式で、数値型から直接定数ノードを生成可能

## テスト

各モジュールには包括的なテストが含まれています：

- `mod.rs`: 型推論、子ノード取得、リテラル変換のテスト
- `ops.rs`: 演算子オーバーロードのテスト
- `helper.rs`: ヘルパー関数のテスト

テスト実行：

```bash
cargo test --lib ast
```

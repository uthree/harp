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

## 設計思想と方針

### 演算の最小性原則

**重要:** harpのASTは、**既存の演算の組み合わせで表現可能な演算は原則としてASTノードに実装しない**という設計思想に基づいています。これは演算子の種類を最小限に抑え、以下のメリットを得るためです：

1. **ASTの単純性**: ノードの種類が少ないほど、変換・最適化・コード生成が容易
2. **保守性**: 演算が少ないほど、各種パスの実装が簡潔になる
3. **一貫性**: 同じ意味を持つ演算が複数の表現を持たない

#### 実装されている最小演算セット

- **加算 (`Add`)**: 基本演算
- **乗算 (`Mul`)**: 基本演算
- **逆数 (`Recip`)**: 除算の基礎（`a / b = a * recip(b)`）
- **剰余 (`Rem`)**: 整数演算
- **整数除算 (`Idiv`)**: 整数演算
- **最大値 (`Max`)**: 比較演算
- その他の数学関数 (`Sqrt`, `Log2`, `Exp2`, `Sin`)

#### 演算子オーバーロードによる派生演算

既存の演算を組み合わせて、ユーザーフレンドリーな演算子を提供：

```rust
// 減算: a - b = a + (-b) = a + (-1 * b)
impl Sub for AstNode {
    fn sub(self, rhs: T) -> AstNode {
        self + (-rhs.into())
    }
}

// 除算: a / b = a * recip(b)
impl Div for AstNode {
    fn div(self, rhs: T) -> AstNode {
        self * AstNode::Recip(Box::new(rhs.into()))
    }
}

// 単項マイナス: -x = -1 * x
impl Neg for AstNode {
    fn neg(self) -> AstNode {
        AstNode::Const(Literal::F32(-1.0)) * self
    }
}
```

#### 実装しない演算の例

以下のような演算は、既存の演算で表現可能なため、ASTノードとして追加しません：

- **減算 (`Sub`)**: `a - b = a + (-b)` で表現
- **除算 (`Div`)**: `a / b = a * recip(b)` で表現
- **余弦 (`Cos`)**: `cos(x) = sin(x + π/2)` で表現可能
- **正接 (`Tan`)**: `tan(x) = sin(x) / cos(x)` で表現可能
- **べき乗 (`Pow`)**: `a^b = exp2(b * log2(a))` で表現可能

ただし、パフォーマンス上の理由や数値安定性の観点から例外的に追加することは検討可能です。

### その他の設計上の注意点

1. **所有権**: ASTノードは`Clone`トレイトを実装しており、複製が可能
2. **型安全性**: `DType`により型情報を保持し、型推論により静的な型チェックが可能
3. **拡張性**: 新しい演算や関数を追加する場合は、`AstNode` enumに追加し、対応する`infer_type()`のケースを実装
4. **定数生成**: `AstNode::Const(value.into())`の形式で、数値型から直接定数ノードを生成可能
5. **マクロによる共通化**: `helper.rs`では、マクロを使用してボイラープレートコードを削減

## DType型変換メソッド

### Vec型への変換

`DType`はSIMD対応のため、スカラー型とベクトル型の相互変換メソッドを提供します：

```rust
// 型をVec型に変換
let vec_type = DType::F32.to_vec(4);  // Vec(F32, 4)

// Vec型から要素型とサイズを取得
if let Some((elem_type, size)) = vec_type.from_vec() {
    // elem_type: &DType::F32, size: 4
}

// Vec型の要素型を取得（Vec型でなければ自身を返す）
let elem = vec_type.element_type();  // &DType::F32

// Vec型かどうか判定
assert!(vec_type.is_vec());
```

### Ptr型への変換

メモリバッファ操作のため、ポインタ型の変換メソッドも提供します：

```rust
// 型をPtr型に変換
let ptr_type = DType::F32.to_ptr();  // Ptr(F32)

// Ptr型から参照先の型を取得
if let Some(pointee) = ptr_type.from_ptr() {
    // pointee: &DType::F32
}

// Ptr型の参照先型を取得（Ptr型でなければ自身を返す）
let pointee = ptr_type.deref_type();  // &DType::F32

// Ptr型かどうか判定
assert!(ptr_type.is_ptr());
```

### ネストした型

Vec型とPtr型は自由にネストできます：

```rust
// Vec<Ptr<F32>>
let vec_of_ptr = DType::F32.to_ptr().to_vec(4);

// Ptr<Vec<F32>>
let ptr_to_vec = DType::F32.to_vec(8).to_ptr();
```

## テスト

各モジュールには包括的なテストが含まれています：

- `mod.rs`: 型推論、子ノード取得、リテラル変換、DType型変換のテスト
- `ops.rs`: 演算子オーバーロードのテスト
- `helper.rs`: ヘルパー関数のテスト

テスト実行：

```bash
cargo test --lib ast
```

## TODO

### AstNode

- [ ] **メモリ操作**: `Load`と`Store`の実装
  - メモリアドレスからの読み込み
  - メモリアドレスへの書き込み
- [ ] **制御構文**: 条件分岐やループ構造の追加
  - `If`/`Else`
  - `Loop`/`While`
  - `Break`/`Continue`
- [ ] **関数と呼び出し**: 関数定義と呼び出しのサポート
  - 関数定義ノード
  - 関数呼び出しノード
  - 引数の受け渡し

### DType

- [ ] **Bool型の追加**: 条件分岐のための真偽値型
- [ ] **F16型の追加**: 半精度浮動小数点数のサポート
  - 省メモリ化
  - GPU互換性
- [ ] **その他の数値型**:
  - `I32`, `U32`: 32ビット整数
  - `I64`, `U64`: 64ビット整数
  - `F64`: 倍精度浮動小数点数

### Literal

- [ ] **Bool型リテラル**: `Literal::Bool(bool)`の追加
- [ ] **F16型リテラル**: 半精度浮動小数点数リテラルの追加

### その他

- [ ] **型推論の改善**: より高度な型推論アルゴリズム
  - 型の自動昇格（type promotion）
  - より詳細なエラーメッセージ
- [ ] **最適化パス**: ASTの最適化機能
  - 定数畳み込み
  - 代数的簡約化

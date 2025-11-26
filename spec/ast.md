# AST (抽象構文木)

## 概要

数値計算を表現するための抽象構文木を提供します。

## 設計思想と方針

### 演算の最小性原則

**既存の演算の組み合わせで表現可能な演算は、原則としてASTノードに実装しない**という設計方針です。演算子の種類を最小限に抑えることで、ASTの単純性・保守性・一貫性を確保します。

実装されている演算: `Add`, `Mul`, `Recip`, `Rem`, `Idiv`, `Max`, 数学関数（`Sqrt`, `Log2`, `Exp2`, `Sin`）

演算子オーバーロードで提供: 減算（`a - b = a + (-b)`）, 除算（`a / b = a * recip(b)`）

### ヘルパー関数とビルダー

`src/ast/helper.rs`にAST構築を簡潔にするヘルパー関数群を提供しています。

#### 演算ヘルパー（マクロ生成）
- **二項演算**: `max`, `idiv`, `rem`
- **単項演算**: `recip`, `sqrt`, `log2`, `exp2`, `sin`

#### 構造化ヘルパー
- **変数**: `var("x")` - 変数参照
- **定数**: `const_int(42)`, `const_f32(3.14)` - 型付き定数
- **制御構造**: `range`, `block`, `empty_block` - ループとブロック
- **メモリ操作**: `load`, `load_vec`, `store`, `assign` - メモリアクセス
- **型変換**: `cast`, `broadcast` - 型変換とSIMDブロードキャスト
- **プログラム構造**: `function`, `program`, `barrier` - 関数とプログラム

#### 演算子オーバーロード

`AstNode`に対する演算子オーバーロードと組み合わせることで、数式を直感的に記述できます：

```rust
use crate::ast::helper::{var, const_int, load};

// 複雑なAST構築を簡潔に
let expr = var("a") + var("b") * const_int(2);
let mem_op = store(var("out"), var("i"), load(var("in"), var("i"), DType::F32) * const_f32(2.0));
```

#### 数値型からの自動変換

`From`トレイトと双方向演算子により、数値型を直接演算に使用できます：

```rust
use crate::ast::helper::var;

// 右辺に数値（Into<AstNode>経由）
let expr = var("x") + 2;          // const_int不要
let expr = var("y") * 3.14f32;    // const_f32不要

// 左辺に数値（逆演算子）
let expr = 2.0 * var("x") + 1.0;  // 数式のように自然に記述
let expr = 10 - var("i");
let expr = 1.0 / var("y");

// 対応する数値型
// f32, isize, usize, i32, i64
```

冗長な`AstNode::Variant { field: Box::new(...) }`の直接初期化を避け、可読性と保守性を向上させます。

## Scopeと変数管理

変数の宣言と型管理を担当します。

### 並列アクセスの安全性

変数のアクセス安全性は`Mutability`によって管理されます：
- **Immutable**: 読み取り専用（複数スレッドから安全にアクセス可能）
- **Mutable**: 書き込み可能（単一スレッドのみ、排他制御が必要）

この単純な2値による管理により、並列実行時のデータ競合を防ぎます。

## 各ノードの責務

### Block
複数の文をグループ化し、独立したスコープを提供。型推論では最後の文の型を返します（空なら`Tuple(vec![])`）。

### Range
範囲ベースのループを表現。ループ変数は自動的にスコープに宣言され、親スコープの変数にもアクセス可能です。

### Function
`AstNode`の一つのバリアントとして実装されています。

通常の関数とGPUカーネルを統一的に表現：
- `FunctionKind::Normal`: CPU上で逐次実行
- `FunctionKind::Kernel(ndim)`: GPU上で並列実行（ndimは並列次元数）

組み込み変数（`ThreadId`, `GroupId`等）はスコープに登録せず、特別扱いします。

```rust
AstNode::Function {
    name: Option<String>,    // 関数名（匿名関数の場合はNone）
    params: Vec<VarDecl>,    // 引数リスト
    return_type: DType,      // 返り値の型
    body: Box<AstNode>,      // 関数本体
    kind: FunctionKind,      // 関数の種類
}
```

### Program
`AstNode`の一つのバリアントとして実装されています。

プログラム全体を表現し、複数の関数定義（`AstNode::Function`のリスト）を管理。エントリーポイント関数から実行が開始されます。

```rust
AstNode::Program {
    functions: Vec<AstNode>,  // Function ノードのリスト
    entry_point: String,      // エントリーポイント関数名
}
```

### Barrier
並列実行における同期点。GPU等で全スレッドがこの地点に到達するまで待機し、共有メモリアクセスのデータ競合を防ぎます。

### Allocate / Deallocate
動的メモリ管理のためのノード。中間バッファーの確保と解放に使用します。

```rust
AstNode::Allocate {
    dtype: Box<DType>,    // 要素の型（例: F32）
    size: Box<AstNode>,   // 要素数（式で指定可能）
}

AstNode::Deallocate {
    ptr: Box<AstNode>,    // 解放するポインタ
}
```

C言語へのレンダリング例:
```c
// Allocate
float* tmp0 = (float*)malloc(size * sizeof(float));

// Deallocate
free(tmp0);
```

## DType型変換

基本型として`Bool`（内部的には`unsigned char`/`uchar`として8ビット整数で表現）、`Int`（整数）、`F32`（浮動小数点）を提供。SIMD対応のベクトル型（`Vec<T, N>`）とメモリバッファ用のポインタ型（`Ptr<T>`）も提供。型変換メソッド（`to_vec`, `to_ptr`等）により、型を自由にネスト可能です（例: `Vec<Ptr<F32>>`, `Ptr<Vec<F32>>`）。

### Bool型
Attention maskなどの用途向けにブール型を提供。内部実装では8ビット整数（C/Metalでは`unsigned char`、OpenCLでは`uchar`）として表現されます。リテラルは`true`→`1`、`false`→`0`としてレンダリングされます。

## AST最適化

ASTノードに対する代数的最適化機能を`src/opt/ast/`に実装しています。

### パターンマッチングと書き換え

`src/ast/pat.rs`にパターンマッチングと書き換えの基礎機能を実装。`astpat!`マクロでパターン記述が可能。

### 最適化フレームワーク

詳細は[opt-ast.md](opt-ast.md)を参照。

# AST モジュール仕様

## 概要

ASTモジュールはC言語的な抽象構文木を提供します。計算グラフから変換された具体的なループ構造やメモリアクセスを表現します。

## 設計思想

ASTノードの設計では、**演算子の種類を最小限に抑える**という原則に従っています。これにより、最適化や変換処理の実装が簡潔になり、保守性が向上します。

### 逆元による演算の統一

基本的な演算を組み合わせることで、より複雑な演算を表現します：

- **減算**: `a - b` = `Add(a, Neg(b))` - 加算と符号反転の組み合わせ
- **除算**: `a / b` = `Mul(a, Recip(b))` - 乗算と逆数の組み合わせ

この設計により：
1. 最適化パスで扱う演算の種類が減少
2. パターンマッチングが簡潔になる
3. コード生成の実装が統一される
4. 代数的簡約化のルールが少なくて済む

例えば、`a + 0`を最適化するパターンを定義すれば、`a - 0`も自動的に`Add(a, Neg(0))`として最適化されます。

## 主要な型

### AstNode

抽象構文木のノード。

```rust
pub enum AstNode {
    // 値の表現
    Const(ConstLiteral),     // 定数値
    Var(String),             // 変数参照
    Cast { dtype: DType, expr: Box<Self> },  // 型変換

    // 数値演算
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    Neg(Box<Self>),
    Recip(Box<Self>),
    Sin(Box<Self>),
    Sqrt(Box<Self>),
    Log2(Box<Self>),
    Exp2(Box<Self>),
    Rand,  // 乱数生成

    // 比較演算（Bool型を返す）
    LessThan(Box<Self>, Box<Self>),
    Eq(Box<Self>, Box<Self>),

    // 条件選択
    Select {
        cond: Box<Self>,
        true_val: Box<Self>,
        false_val: Box<Self>,
    },

    // ビット演算
    BitAnd(Box<Self>, Box<Self>),
    BitOr(Box<Self>, Box<Self>),
    BitXor(Box<Self>, Box<Self>),
    Shl(Box<Self>, Box<Self>),
    Shr(Box<Self>, Box<Self>),
    BitNot(Box<Self>),

    // 文・制御構造
    Block {
        scope: Scope,
        statements: Vec<AstNode>,
    },
    Assign(String, Box<Self>),  // 変数への代入
    Store {
        target: Box<Self>,
        index: Box<Self>,
        value: Box<Self>,
    },  // メモリへの書き込み (target[index] = value)
    Deref(Box<Self>),  // ポインタの参照外し (*expr)

    // ループ
    Range {
        counter_name: String,
        start: Box<Self>,
        max: Box<Self>,
        step: Box<Self>,
        body: Box<Self>,
        unroll: Option<usize>,  // アンロールヒント
    },

    // その他
    Drop(String),  // 変数の明示的な削除
    Barrier,       // 同期バリア（並列実行の世代区切り）
    CallFunction { name: String, args: Vec<Self> },

    // 関数とプログラム定義
    Function {
        name: String,
        scope: Scope,
        statements: Vec<AstNode>,
        arguments: Vec<(String, DType)>,
        return_type: DType,
    },
    Program {
        functions: Vec<AstNode>,  // 各要素はAstNode::Function
        entry_point: String,
    },

    // パターンマッチング用
    Capture(usize),
}
```

## 演算子のオーバーロード

AstNodeは標準的な演算子をオーバーロードしています。

### 算術演算子
- `+`, `-`, `*`, `/`, `%`
- 減算: `a - b` = `a + (-b)`
- 除算: `a / b` = `a * (1/b)`

### ビット演算子
- `&`, `|`, `^`, `<<`, `>>`, `!`

### 代入演算子
- `+=`, `-=`, `*=`, `/=`, `%=`
- `&=`, `|=`, `^=`, `<<=`, `>>=`

### 単項演算子
- `-`: 符号反転（Neg）
- `!`: ビット否定（BitNot）

## データ型

### DType

```rust
pub enum DType {
    F32,    // float
    Usize,  // size_t
    Isize,  // ssize_t
    Bool,   // boolean
    Void,

    Ptr(Box<Self>),        // ポインタ
    Vec(Box<Self>, usize), // 固定長配列（SIMD用）
}
```

### ConstLiteral

```rust
pub enum ConstLiteral {
    F32(f32),
    Usize(usize),
    Isize(isize),
    Bool(bool),
}
```

## スコープと変数宣言

### Scope

```rust
pub struct Scope {
    pub declarations: Vec<VariableDecl>,
}
```

### VariableDecl

```rust
pub struct VariableDecl {
    pub name: String,
    pub dtype: DType,
    pub constant: bool,
    pub size_expr: Option<Box<AstNode>>,  // 動的配列のサイズ式
}
```

## 関数とプログラム

### Function

Functionは`AstNode`のバリアントとして定義されています。

```rust
AstNode::Function {
    name: String,
    scope: Scope,
    statements: Vec<AstNode>,
    arguments: Vec<(String, DType)>,
    return_type: DType,
}
```

### Program

Programも`AstNode`のバリアントとして定義されています。型エイリアス`Program = AstNode`として利用できます。

```rust
AstNode::Program {
    functions: Vec<AstNode>,  // 各要素はAstNode::Function
    entry_point: String,
}
```

## RangeBuilder

ループ構造を構築するためのビルダーパターン。

```rust
AstNode::range_builder("i", 10isize, body)
    .start(0isize)
    .step(1isize)
    .unroll()  // 完全アンロール
    .build()
```

**アンロールオプション:**
- `None`: アンロールなし
- `Some(0)`: 完全アンロール
- `Some(n)`: n回アンロール

## ヘルパー関数

### 値の構築
- `AstNode::var(name)`: 変数参照
- `AstNode::const_val(val)`: 定数
- `AstNode::capture(n)`: キャプチャノード

### 演算
- `AstNode::max(lhs, rhs)`: 最大値
- `AstNode::cast(dtype, expr)`: 型変換
- `AstNode::less_than(lhs, rhs)`: 比較
- `AstNode::eq(lhs, rhs)`: 等価比較
- `AstNode::select(cond, true_val, false_val)`: 条件選択

### 文
- `AstNode::assign(var_name, value)`: 代入
- `AstNode::store(target, index, value)`: メモリ書き込み
- `AstNode::deref(expr)`: 参照外し
- `AstNode::drop(var_name)`: 変数削除

### ブロック
- `AstNode::block(scope, statements)`: スコープ付きブロック
- `AstNode::block_with_statements(statements)`: 空スコープのブロック

### ループ
- `AstNode::range(counter_name, max, body)`: 基本ループ
- `AstNode::range_builder(...)`: ビルダーパターン

### 関数とプログラム
- `function(name, arguments, return_type, scope, statements)`: 関数定義
- `program(functions, entry_point)`: プログラム定義

### その他
- `AstNode::call(name, args)`: 関数呼び出し
- `AstNode::rand()`: 乱数生成
- `AstNode::barrier()`: バリア

## ノード操作

### children()
子ノードのリストを取得。

### replace_node(target, replacement)
特定のノードを別のノードに置き換え。

### replace_if(predicate, transform)
条件を満たすノードを変換。

```rust
ast.replace_if(
    |node| matches!(node, AstNode::Add(_, r) if **r == AstNode::Const(ConstLiteral::F32(0.0))),
    |node| if let AstNode::Add(l, _) = node { *l } else { node }
)
```

### replace_children(new_children)
子ノードを置き換えて新しいノードを構築。

## パターンマッチング

`Capture(n)`ノードを使用して、パターンマッチングをサポート。
詳細は`src/ast/pattern.rs`を参照。

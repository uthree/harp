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
    Load {
        target: Box<Self>,
        index: Box<Self>,
        vector_width: usize,  // ベクトル幅（1 = スカラー）
    },  // メモリからの読み込み (target[index])
    Store {
        target: Box<Self>,
        index: Box<Self>,
        value: Box<Self>,
        vector_width: usize,  // ベクトル幅（1 = スカラー）
    },  // メモリへの書き込み (target[index] = value)

    // ループ
    Range {
        counter_name: String,
        start: Box<Self>,
        max: Box<Self>,
        step: Box<Self>,
        body: Box<Self>,
        unroll: Option<usize>,  // アンロールヒント
    },

    // 条件分岐
    If {
        condition: Box<Self>,      // Bool型の条件式
        then_branch: Box<Self>,    // 条件が真の場合に実行
        else_branch: Option<Box<Self>>,  // 条件が偽の場合に実行（オプション）
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
    Kernel {
        name: String,
        scope: KernelScope,        // カーネル専用スコープ（スレッドID変数を含む）
        statements: Vec<AstNode>,
        arguments: Vec<(String, DType)>,
        return_type: DType,
        global_size: [Box<AstNode>; 3],  // グローバルワークサイズ（3次元）
        local_size: [Box<AstNode>; 3],   // ローカルワークサイズ（3次元）
    },
    CallKernel {
        name: String,
        args: Vec<Self>,
        global_size: [Box<AstNode>; 3],
        local_size: [Box<AstNode>; 3],
    },
    Program {
        functions: Vec<AstNode>,  // 各要素はAstNode::FunctionまたはAstNode::Kernel
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

関数のスコープを表します。

```rust
pub struct Scope {
    pub declarations: Vec<VariableDecl>,
}
```

### KernelScope

カーネル（GPU関数）のスコープを表します。通常の変数宣言に加えて、スレッドID変数を含みます。

```rust
pub struct KernelScope {
    pub declarations: Vec<VariableDecl>,
    pub thread_ids: Vec<ThreadIdDecl>,
}
```

### ThreadIdDecl

GPUスレッドIDを表す3次元ベクトル変数の宣言。

```rust
pub struct ThreadIdDecl {
    pub name: String,
    pub id_type: ThreadIdType,
}

pub enum ThreadIdType {
    GlobalId,   // グローバルスレッドID
    LocalId,    // ローカル（ワークグループ内）スレッドID
    GroupId,    // ワークグループID
}
```

**使用例:**
```rust
// グローバルスレッドIDの取得
// size_t global_id[3] = get_global_id();
// size_t i = global_id[0];
ThreadIdDecl {
    name: "global_id".to_string(),
    id_type: ThreadIdType::GlobalId,
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

Functionは`AstNode`のバリアントとして定義されています。ヘルパーメソッドを使って構築できます。

```rust
AstNode::Function {
    name: String,
    scope: Scope,
    statements: Vec<AstNode>,
    arguments: Vec<(String, DType)>,
    return_type: DType,
}

// ヘルパーメソッド
AstNode::function(name, arguments, return_type, scope, statements)
```

### Program

Programも`AstNode`のバリアントとして定義されています。ヘルパーメソッドを使って構築できます。

```rust
AstNode::Program {
    functions: Vec<AstNode>,  // 各要素はAstNode::Function
    entry_point: String,
}

// ヘルパーメソッド
AstNode::program(functions, entry_point)
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
- `AstNode::load(target, index)`: メモリ読み込み（スカラー）
- `AstNode::load_vec(target, index, vector_width)`: ベクトルメモリ読み込み
- `AstNode::store(target, index, value)`: メモリ書き込み（スカラー）
- `AstNode::store_vec(target, index, value, vector_width)`: ベクトルメモリ書き込み
- `AstNode::drop(var_name)`: 変数削除

### 制御構造
- `AstNode::if_then(condition, then_branch)`: if文
- `AstNode::if_then_else(condition, then_branch, else_branch)`: if-else文
- `AstNode::block(scope, statements)`: スコープ付きブロック
- `AstNode::block_with_statements(statements)`: 空スコープのブロック

### ループ
- `AstNode::range(counter_name, max, body)`: 基本ループ
- `AstNode::range_builder(...)`: ビルダーパターン

### 関数とプログラム
- `AstNode::function(name, arguments, return_type, scope, statements)`: 関数定義
- `AstNode::kernel(name, arguments, return_type, scope, statements, global_size, local_size)`: カーネル定義
- `AstNode::call_kernel(name, args, global_size, local_size)`: カーネル呼び出し
- `AstNode::program(functions, entry_point)`: プログラム定義
- `function(name, arguments, return_type, scope, statements)`: 関数定義（ヘルパー関数）
- `kernel(name, arguments, return_type, scope, statements, global_size, local_size)`: カーネル定義（ヘルパー関数）
- `call_kernel(name, args, global_size, local_size)`: カーネル呼び出し（ヘルパー関数）
- `program(functions, entry_point)`: プログラム定義（ヘルパー関数）
- `thread_id_decl(name, id_type)`: スレッドID変数宣言
- `kernel_scope(thread_ids, declarations)`: カーネルスコープ作成

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

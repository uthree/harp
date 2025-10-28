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
    // パターンマッチング用ワイルドカード
    Wildcard(String),

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

    // 変数
    Var(String),                          // 変数参照

    // メモリ操作（バッファー用）
    Load {
        ptr: Box<AstNode>,                // ポインタ（Ptr<T>型の式）
        offset: Box<AstNode>,             // オフセット（Usize型の式）
        count: usize,                     // 読み込む要素数（コンパイル時定数、1ならスカラー）
    },
    Store {
        ptr: Box<AstNode>,                // ポインタ（Ptr<T>型の式）
        offset: Box<AstNode>,             // オフセット（Usize型の式）
        value: Box<AstNode>,              // 書き込む値（スカラーまたはVec型）
    },

    // 変数への代入（スタック/レジスタ用）
    Assign {
        var: String,                      // 変数名
        value: Box<AstNode>,              // 代入する値
    },

    // 文のブロック
    Block {
        statements: Vec<AstNode>,         // 文のリスト
        scope: Box<Scope>,                // ブロックのスコープ
    },

    // 制御構文
    Range {
        var: String,          // ループ変数名
        start: Box<AstNode>,  // 開始値
        step: Box<AstNode>,   // ステップ
        stop: Box<AstNode>,   // 終了値
        body: Box<AstNode>,   // ループ本体（通常はBlockノード）
    },

    // 関数呼び出し
    Call {
        name: String,        // 関数名
        args: Vec<AstNode>,  // 引数リスト
    },

    // 返り値
    Return {
        value: Box<AstNode>, // 返す値
    },
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
- `Wildcard`: `Unknown`を返す（パターンマッチング用）
- `Const`: リテラルの型を返す
- `Cast`: 指定された型を返す
- `Var`: `Unknown`を返す（変数の型はコンテキストに依存）
- 二項演算 (`Add`, `Mul`, `Max`, `Rem`, `Idiv`): 両辺の型が一致すればその型、異なれば`Unknown`
- `Recip`: オペランドの型を保持
- 数学関数 (`Sqrt`, `Log2`, `Exp2`, `Sin`): 常に`F32`を返す
- `Load`: ポインタが指す型を返す。`count=1`ならスカラー、`count>1`なら`Vec<T, count>`型を返す
- `Store`: `Tuple(vec![])`を返す（unit型、値を返さない）
- `Assign`: 代入される値の型を返す
- `Range`: `Tuple(vec![])`を返す（unit型、値を返さない）
- `Call`: `Unknown`を返す（関数定義を参照して型を決定する必要があるため、Programコンテキストで解決）
- `Return`: 返す値の型を返す

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

### 変数とメモリ操作

```rust
pub fn var(name: impl Into<String>) -> AstNode
pub fn load(ptr: AstNode, offset: AstNode) -> AstNode
pub fn load_vec(ptr: AstNode, offset: AstNode, count: usize) -> AstNode
pub fn store(ptr: AstNode, offset: AstNode, value: AstNode) -> AstNode
pub fn assign(var: impl Into<String>, value: AstNode) -> AstNode
```

使用例：

```rust
use harp::ast::helper::*;

// 数学関数
let a = AstNode::Const(4.0f32.into());
let result = sqrt(a);  // AstNode::Sqrt(...)

// 変数参照
let x = var("x");
let buffer = var("input0");

// スカラーロード: input0[i]
let i = var("i");
let loaded = load(buffer, i.clone());

// ベクトルロード: 4要素まとめて読み込み
let ptr = cast(var("buffer"), DType::F32.to_ptr());
let vec_loaded = load_vec(ptr, var("offset"), 4);

// ストア: output[i] = value
let stored = store(var("output"), i, loaded);

// 代入: alu0 = a + b
let assigned = assign("alu0", var("a") + var("b"));
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

### 変数の使用

```rust
// 変数参照
let var_x = AstNode::Var("x".to_string());
let var_y = AstNode::Var("y".to_string());

// 変数を使った計算: x + y
let sum = var_x + var_y;
```

### メモリ操作

```rust
use harp::ast::{AstNode, DType};

// スカラーロード: input0[i]
let load_scalar = AstNode::Load {
    ptr: Box::new(AstNode::Var("input0".to_string())),
    offset: Box::new(AstNode::Var("i".to_string())),
    count: 1,  // スカラー読み込み
};

// ベクトルロード: 4要素まとめて読み込み
let load_vector = AstNode::Load {
    ptr: Box::new(AstNode::Cast(
        Box::new(AstNode::Var("buffer".to_string())),
        DType::F32.to_ptr(),
    )),
    offset: Box::new(AstNode::Const(0usize.into())),
    count: 4,  // Vec<F32, 4>を返す
};

// ストア: output0[i] = value
let store = AstNode::Store {
    ptr: Box::new(AstNode::Var("output0".to_string())),
    offset: Box::new(AstNode::Var("i".to_string())),
    value: Box::new(AstNode::Const(3.14f32.into())),
};

// 変数への代入: alu0 = a + b
let assign = AstNode::Assign {
    var: "alu0".to_string(),
    value: Box::new(
        AstNode::Var("a".to_string()) + AstNode::Var("b".to_string())
    ),
};
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

## Scope - スコープと変数管理

`Scope`は変数の宣言と型管理、並列アクセスの安全性チェックを行います。

### データ構造

```rust
pub struct Scope {
    variables: HashMap<String, VarDecl>,
    parent: Option<Box<Scope>>,
}

pub struct VarDecl {
    pub name: String,            // 変数名
    pub dtype: DType,            // 変数の型
    pub mutability: Mutability,  // 可変性
    pub region: AccessRegion,    // アクセス領域
}
```

### VarDecl - 変数宣言

`VarDecl`は変数の宣言情報を表します。変数名、型、可変性、アクセス領域を持ちます。

- **name**: 変数名
- **dtype**: 変数の型
- **mutability**: 変数が書き換え可能かどうか
- **region**: 変数がどのメモリ領域に配置されるか、並列実行時にどのようにアクセスされるか

`VarDecl`は以下の用途で使用されます：

1. **Scopeでの変数管理**: `HashMap<String, VarDecl>`として保存
2. **関数パラメータ**: `Function`の`params: Vec<VarDecl>`として使用

```rust
// 変数宣言の例
let var_decl = VarDecl {
    name: "x".to_string(),
    dtype: DType::F32,
    mutability: Mutability::Mutable,
    region: AccessRegion::ThreadLocal,
}

pub enum Mutability {
    Immutable,  // 読み取り専用（複数スレッドから安全にアクセス可）
    Mutable,    // 書き込み可能（単一スレッドのみ）
}

pub enum AccessRegion {
    ThreadLocal,           // スレッドローカル（競合なし）
    Shared,                // 共有メモリ（読み取り専用なら安全）
    ShardedBy(Vec<usize>), // 特定の軸でシャーディング（軸番号のリスト）
}
```

### 主要機能

#### 変数の宣言

```rust
let mut scope = Scope::new();

scope.declare(
    "input".to_string(),
    DType::F32.to_ptr(),
    Mutability::Immutable,
    AccessRegion::Shared,
).unwrap();
```

#### 読み取り・書き込みチェック

```rust
// 読み取り可能かチェック
scope.check_read("input")?;

// 書き込み可能かチェック（mutabilityと型をチェック）
scope.check_write("output", &DType::F32)?;
```

#### 並列アクセスの安全性チェック

```rust
// 2つの変数が並列にアクセス可能かチェック
if scope.can_access_parallel("input", "output") {
    // 並列実行可能
}
```

**並列アクセス可能な条件：**
- 両方が`Immutable`
- 両方が`ThreadLocal`
- 異なる軸で`ShardedBy`されている

#### ASTのスコープチェック

```rust
let ast = store(
    var("output"),
    var("i"),
    load(var("input"), var("i")) * AstNode::Const(2.0f32.into())
);

// ASTがスコープ内で有効かチェック
ast.check_scope(&scope)?;
```

### 使用例

```rust
use harp::ast::*;
use harp::ast::helper::*;

// スコープを作成
let mut scope = Scope::new();

// 入力バッファ（読み取り専用、共有）
scope.declare(
    "input".to_string(),
    DType::F32.to_ptr(),
    Mutability::Immutable,
    AccessRegion::Shared,
).unwrap();

// 出力バッファ（書き込み可能、軸0でシャーディング）
scope.declare(
    "output".to_string(),
    DType::F32.to_ptr(),
    Mutability::Mutable,
    AccessRegion::ShardedBy(vec![0]),
).unwrap();

// インデックス（読み取り専用、スレッドローカル）
scope.declare(
    "i".to_string(),
    DType::Usize,
    Mutability::Immutable,
    AccessRegion::ThreadLocal,
).unwrap();

// AST構築: output[i] = input[i] * 2.0
let ast = store(
    var("output"),
    var("i"),
    load(var("input"), var("i")) * AstNode::Const(2.0f32.into())
);

// スコープチェック
ast.check_scope(&scope).unwrap();

// 並列性チェック
assert!(scope.can_access_parallel("input", "output"));  // OK
```

### ネストしたスコープ

```rust
let mut parent_scope = Scope::new();
parent_scope.declare(
    "global_var".to_string(),
    DType::F32,
    Mutability::Immutable,
    AccessRegion::Shared,
).unwrap();

let child_scope = Scope::with_parent(parent_scope);

// 子スコープから親スコープの変数にアクセス可能
assert!(child_scope.check_read("global_var").is_ok());
```

## Block - 文のブロック

`Block`ノードは複数の文（statement）をグループ化し、それらに対して独立したスコープを提供します。

### データ構造

```rust
Block {
    statements: Vec<AstNode>,  // 文のリスト
    scope: Box<Scope>,         // ブロック内のスコープ
}
```

### 特徴

- **スコープ管理**: Blockは独自のスコープを持ち、ブロック内で宣言された変数はブロック外からアクセスできません
- **型推論**: Blockの型は最後の文の型として推論されます。空のBlockは`Tuple(vec![])`型になります
- **使用場面**: RangeのbodyやFunctionのbodyとして使用されます

### 使用例

```rust
let mut scope = Scope::new();
scope.declare(
    "x".to_string(),
    DType::Isize,
    Mutability::Immutable,
    AccessRegion::ThreadLocal,
)?;

let block = AstNode::Block {
    statements: vec![
        AstNode::Var("x".to_string()),
        AstNode::Const(42isize.into()),
    ],
    scope: Box::new(scope),
};

// ブロックの型は最後の文の型（Isize）
assert_eq!(block.infer_type(), DType::Isize);
```

## Range - ループ構造

`Range`ノードはforループのような範囲に基づくイテレーションを表現します。

### データ構造

```rust
Range {
    var: String,          // ループ変数名
    start: Box<AstNode>,  // 開始値（Usize型）
    step: Box<AstNode>,   // ステップ（Usize型）
    stop: Box<AstNode>,   // 終了値（Usize型）
    body: Box<AstNode>,   // ループ本体（通常はBlockノード）
}
```

### 意味論

- ループ変数`var`は`start`から`stop-1`まで`step`ずつ増加します
- `start`, `step`, `stop`はループ開始前の外側のスコープで評価されます
- ループ本体`body`は各イテレーションで順次実行されます
- `body`は通常`Block`ノードであり、そのスコープにループ変数が宣言されている必要があります

### スコープの扱い

- `body`は通常`Block`ノードです
- ループ変数`var`は`Block`のスコープに宣言されている必要があります
- 通常、ループスコープは外側のスコープを親として持ち、外側の変数にもアクセスできます

### 使用例

#### 基本的なループ

```rust
use harp::ast::*;
use harp::ast::helper::*;

// ループ変数のスコープを作成
let mut loop_scope = Scope::new();
loop_scope.declare(
    "i".to_string(),
    DType::Usize,
    Mutability::Immutable,
    AccessRegion::ThreadLocal,
).unwrap();

// for i in 0..10 step 1 { ... }
let loop_node = AstNode::Range {
    var: "i".to_string(),
    start: Box::new(AstNode::Const(0usize.into())),
    step: Box::new(AstNode::Const(1usize.into())),
    stop: Box::new(AstNode::Const(10usize.into())),
    body: Box::new(AstNode::Block {
        statements: vec![
            // ループ本体の処理
        ],
        scope: Box::new(loop_scope),
    }),
};
```

#### メモリ操作を含むループ

```rust
// 外側のスコープ（入出力バッファ）
let mut outer_scope = Scope::new();
outer_scope.declare(
    "input".to_string(),
    DType::F32.to_ptr(),
    Mutability::Immutable,
    AccessRegion::Shared,
).unwrap();
outer_scope.declare(
    "output".to_string(),
    DType::F32.to_ptr(),
    Mutability::Mutable,
    AccessRegion::ShardedBy(vec![0]),
).unwrap();

// ループ変数のスコープ
let mut loop_scope = Scope::with_parent(outer_scope.clone());
loop_scope.declare(
    "i".to_string(),
    DType::Usize,
    Mutability::Immutable,
    AccessRegion::ThreadLocal,
).unwrap();

// for i in 0..n { output[i] = input[i] * 2.0 }
let loop_node = AstNode::Range {
    var: "i".to_string(),
    start: Box::new(AstNode::Const(0usize.into())),
    step: Box::new(AstNode::Const(1usize.into())),
    stop: Box::new(var("n")),
    body: Box::new(AstNode::Block {
        statements: vec![
            store(
                var("output"),
                var("i"),
                load(var("input"), var("i")) * AstNode::Const(2.0f32.into())
            ),
        ],
        scope: Box::new(loop_scope),
    }),
};
```

#### ネストしたループ

```rust
// 外側のループのスコープ
let mut outer_loop_scope = Scope::new();
outer_loop_scope.declare(
    "i".to_string(),
    DType::Usize,
    Mutability::Immutable,
    AccessRegion::ThreadLocal,
).unwrap();

// 内側のループのスコープ
let mut inner_loop_scope = Scope::with_parent(outer_loop_scope.clone());
inner_loop_scope.declare(
    "j".to_string(),
    DType::Usize,
    Mutability::Immutable,
    AccessRegion::ThreadLocal,
).unwrap();

// for i in 0..10 { for j in 0..10 { ... } }
let nested_loop = AstNode::Range {
    var: "i".to_string(),
    start: Box::new(AstNode::Const(0usize.into())),
    step: Box::new(AstNode::Const(1usize.into())),
    stop: Box::new(AstNode::Const(10usize.into())),
    body: Box::new(AstNode::Block {
        statements: vec![
            AstNode::Range {
                var: "j".to_string(),
                start: Box::new(AstNode::Const(0usize.into())),
                step: Box::new(AstNode::Const(1usize.into())),
                stop: Box::new(AstNode::Const(10usize.into())),
                body: Box::new(AstNode::Block {
                    statements: vec![
                        // 内側のループ本体
                    ],
                    scope: Box::new(inner_loop_scope),
                }),
            },
        ],
        scope: Box::new(outer_loop_scope),
    }),
};
```

### スコープチェック

`check_scope()`の動作:

1. `start`, `step`, `stop`を外側のスコープでチェック
2. `body`をチェック（通常はBlockのcheck_scopeが呼ばれる）
3. `body`がBlockの場合、ループ変数`var`がそのスコープに宣言されているかチェック

```rust
// スコープチェックの実行
loop_node.check_scope(&outer_scope)?;
```

Blockを使用することで、以下のメリットがあります：

- **型安全性の向上**: ループは常にスコープを持つことが保証される
- **コードの統一性**: 文のグループとスコープが常にセットで扱われる
- **明示的な設計**: ループ変数の管理が明確になり、バグを防ぎやすくなった

## Function - 関数定義

`Function`構造体は関数の定義を表現します。

### データ構造

```rust
pub struct Function {
    pub params: Vec<VarDecl>,   // 引数リスト
    pub return_type: DType,     // 返り値の型
    pub body: Box<AstNode>,     // 関数本体（通常はBlockノード、引数を含むスコープを持つ）
}
```

### 主要メソッド

#### 関数の作成

```rust
let params = vec![
    VarDecl {
        name: "a".to_string(),
        dtype: DType::F32,
        mutability: Mutability::Immutable,
        region: AccessRegion::ThreadLocal,
    },
    VarDecl {
        name: "b".to_string(),
        dtype: DType::F32,
        mutability: Mutability::Immutable,
        region: AccessRegion::ThreadLocal,
    },
];
let return_type = DType::F32;
let body = vec![
    AstNode::Return {
        value: Box::new(var("a") + var("b")),
    }
];

let func = Function::new(params, return_type, body)?;
```

`Function::new()`は自動的にスコープを作成し、各パラメータをそれぞれの`mutability`と`region`で宣言します。

#### 関数本体の検証

```rust
// 関数本体がスコープ内で有効かチェック
func.check_body()?;
```

#### 返り値の型推論

```rust
// Return文から返り値の型を推論
let inferred_type = func.infer_return_type();
```

## Program - プログラム全体

`Program`構造体はプログラム全体の構造を管理します。

### データ構造

```rust
pub struct Program {
    pub functions: HashMap<String, Function>, // 関数定義の集合
    pub entry_point: String,                   // エントリーポイントの関数名
}
```

### 主要メソッド

#### プログラムの作成

```rust
let mut program = Program::new("main".to_string());
```

#### 関数の追加

```rust
let func = Function::new(params, return_type, body)?;
program.add_function("main".to_string(), func)?;
```

#### 関数の取得

```rust
if let Some(func) = program.get_function("main") {
    // 関数を使用
}

// エントリーポイントの取得
if let Some(entry) = program.get_entry() {
    // エントリーポイントを使用
}
```

#### プログラムの検証

```rust
// エントリーポイントの存在と全関数本体を検証
program.validate()?;
```

`validate()`は以下をチェックします：
- エントリーポイント関数が存在するか
- 全ての関数本体がスコープ内で有効か

### 使用例

```rust
use harp::ast::*;
use harp::ast::helper::*;

let mut program = Program::new("main".to_string());

// helper関数: double(x) = x * 2
let double_func = Function::new(
    vec![("x".to_string(), DType::Isize)],
    DType::Isize,
    vec![AstNode::Return {
        value: Box::new(var("x") * AstNode::Const(2isize.into())),
    }],
)?;
program.add_function("double".to_string(), double_func)?;

// main関数: Call double(5)
let main_func = Function::new(
    vec![],
    DType::Isize,
    vec![
        AstNode::Call {
            name: "double".to_string(),
            args: vec![AstNode::Const(5isize.into())],
        }
    ],
)?;
program.add_function("main".to_string(), main_func)?;

// プログラム全体の検証
program.validate()?;
```

## Call - 関数呼び出し

`Call`ノードは関数呼び出しを表現します。

### データ構造

```rust
Call {
    name: String,        // 関数名
    args: Vec<AstNode>,  // 引数リスト
}
```

### 使用例

```rust
// add(x, y)を呼び出し
let call = AstNode::Call {
    name: "add".to_string(),
    args: vec![var("x"), var("y")],
};
```

### スコープチェック

`check_scope()`は引数のスコープチェックのみを行います。関数名の存在確認は`Program::validate()`で行います。

## Return - 返り値

`Return`ノードは関数からの返り値を表現します。

### データ構造

```rust
Return {
    value: Box<AstNode>, // 返す値
}
```

### 使用例

```rust
// 計算結果を返す
let ret = AstNode::Return {
    value: Box::new(var("result")),
};
```

### 型推論

`Return`の型は、返す値の型と同じです：

```rust
let ret = AstNode::Return {
    value: Box::new(AstNode::Const(42isize.into())),
};
assert_eq!(ret.infer_type(), DType::Isize);
```

## 設計思想

### インライン展開を前提とした設計

関数呼び出しは最終的にインライン展開されることを前提としています。これにより：

- **GPU互換性**: GPUカーネルでは関数呼び出しコストを避けるため、インライン展開が一般的
- **最適化の余地**: 最適化パスでインライン展開を行うことで、より高度な最適化が可能
- **シンプルな実装**: スタック操作やコールフレームの管理が不要

### 静的型チェック

- 関数のパラメータと返り値は明示的に型を持つ
- `Program::validate()`で全体の整合性をチェック
- スコープシステムと統合され、変数の型安全性も保証

### 今後の拡張性

現在は基本的な関数機能のみを実装していますが、以下の拡張が可能です：

- 関数呼び出しの存在チェック（`validate()`で実装予定）
- 関数間の型チェック
- 再帰関数のサポート（必要に応じて）
- 高階関数やクロージャ（必要に応じて）

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

- [x] **メモリ操作**: `Load`と`Store`の実装
  - メモリアドレスからの読み込み（SIMD対応）
  - メモリアドレスへの書き込み
  - `Var`による変数参照
  - `Assign`による変数への代入
- [x] **制御構文**: ループ構造の追加
  - `Range` - 範囲ベースのループ（スコープサポート付き）
- [x] **関数と呼び出し**: 関数定義と呼び出しのサポート
  - `Function`構造体 - 関数定義（パラメータ、返り値、本体、スコープ）
  - `Program`構造体 - プログラム全体の管理（関数の集合、エントリーポイント）
  - `Call`ノード - 関数呼び出し
  - `Return`ノード - 返り値
  - インライン展開を前提とした設計
- [x] スコープの概念
  - 変数の読み取り専用/書き込み可能（`Mutability`）
  - 並列アクセスの安全性チェック（`AccessRegion`）
  - 型環境による変数の型管理
  - ネストしたスコープのサポート
- [ ] ASTのパターンマッチングによる置き換え
  - AstRewriteRule
    - 初期化: Rcで自身を初期化して返す。
    - ASTを再起的に探索し、項の書き換えを行う。
  - AstRewriter
    - 複数のAstRewriteRuleを持つ。applyで順番に全部適用。変化がなくなるか所定の回数に達するまで繰り返す。
  - マクロでの初期化
    大体こんな感じの構文で使えるように
    ```rs
    ast_rewriter!{
      astpat!(|a, b| a + b => b + a),
      astpat!(|a| a / a => 1 if a != 0)
    }
    ```

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

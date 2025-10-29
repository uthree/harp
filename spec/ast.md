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

    // 同期バリア
    Barrier,                              // 並列実行の同期点
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
- `Barrier`: `Tuple(vec![])`を返す（unit型、値を返さない）

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
pub fn barrier() -> AstNode
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

// バリア: 並列実行の同期点
let sync = barrier();
```

## 使用例

```rust
use harp::ast::{AstNode, DType};
use harp::ast::helper::*;

// 基本的な式: (a + b) * sqrt(c)
let expr = (AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into()))
           * sqrt(AstNode::Const(4.0f32.into()));
assert_eq!(expr.infer_type(), DType::F32);

// 型キャスト
let casted = cast(AstNode::Const(42isize.into()), DType::F32);

// メモリ操作: output[i] = input[i] * 2
let computation = store(
    var("output"),
    var("i"),
    load(var("input"), var("i")) * AstNode::Const(2.0f32.into())
);
```

## 設計思想と方針

### 演算の最小性原則

**重要:** harpのASTは、**既存の演算の組み合わせで表現可能な演算は原則としてASTノードに実装しない**という設計思想に基づいています。これは演算子の種類を最小限に抑え、以下のメリットを得るためです：

1. **ASTの単純性**: ノードの種類が少ないほど、変換・最適化・コード生成が容易
2. **保守性**: 演算が少ないほど、各種パスの実装が簡潔になる
3. **一貫性**: 同じ意味を持つ演算が複数の表現を持たない

**実装されている最小演算**: `Add`, `Mul`, `Recip`（除算の基礎）, `Rem`, `Idiv`, `Max`, 数学関数（`Sqrt`, `Log2`, `Exp2`, `Sin`）

**演算子オーバーロードで提供**: 減算（`a - b = a + (-b)`）, 除算（`a / b = a * recip(b)`）, 単項マイナス（`-x = -1 * x`）

**実装しない演算**: 余弦・正接・べき乗などは既存演算で表現可能（パフォーマンスや数値安定性で必要なら追加可能）

### その他の設計上の注意点

- ASTノードは`Clone`可能、型推論により静的な型チェックが可能
- 新演算追加時は`AstNode`に追加し`infer_type()`を実装
- `helper.rs`でマクロによりボイラープレートを削減

## DType型変換メソッド

SIMD対応のベクトル型とメモリバッファ用のポインタ型の変換メソッドを提供：

- `to_vec(size)`, `from_vec()`, `element_type()`, `is_vec()`: Vec型の操作
- `to_ptr()`, `from_ptr()`, `deref_type()`, `is_ptr()`: Ptr型の操作
- Vec型とPtr型は自由にネスト可能（例: `Vec<Ptr<F32>>`, `Ptr<Vec<F32>>`）

## Scope - スコープと変数管理

**責務**: 変数の宣言と型管理、並列アクセスの安全性チェック

### VarDecl - 変数宣言

変数の宣言情報を保持：

- **mutability**: `Immutable`（読み取り専用）/ `Mutable`（書き込み可能）
- **region**: `ThreadLocal`（競合なし）/ `Shared`（共有）/ `ShardedBy(axes)`（軸でシャーディング）
- **kind**: `Normal` / `ThreadId(axis)` / `GroupId(axis)` / `GroupSize(axis)` / `GridSize(axis)`

用途: Scopeでの変数管理と関数パラメータとして使用

### 主要機能

- `declare()`: 変数を宣言
- `check_read()`, `check_write()`: 読み取り・書き込み可能性をチェック
- `can_access_parallel()`: 並列アクセスの安全性をチェック（両方が`Immutable` / 両方が`ThreadLocal` / 異なる軸で`ShardedBy`）
- `AstNode::check_scope()`: ASTがスコープ内で有効かチェック

### 使用例

```rust
use harp::ast::*;

let mut scope = Scope::new();
scope.declare("input".to_string(), DType::F32.to_ptr(),
              Mutability::Immutable, AccessRegion::Shared).unwrap();
scope.declare("output".to_string(), DType::F32.to_ptr(),
              Mutability::Mutable, AccessRegion::ShardedBy(vec![0])).unwrap();

// 並列性チェック
assert!(scope.can_access_parallel("input", "output"));
```

## Block - 文のブロック

**責務**: 複数の文をグループ化し、独立したスコープを提供する

- 独自のスコープを持ち、ブロック内変数のライフタイムを管理
- 型推論では最後の文の型を返す（空なら`Tuple(vec![])`）
- RangeやFunctionのbodyとして使用

## Range - ループ構造

**責務**: 範囲ベースのイテレーションを表現

- ループ変数`var`を`start`から`stop-1`まで`step`ずつ増加させる
- `body`は通常Blockノードで、ループ変数がそのスコープに宣言されている必要がある
- ループ変数は親スコープの変数にもアクセス可能

## Function - 関数定義

**責務**: 通常の関数とGPUカーネルの定義を表現

- `FunctionKind::Normal`: CPU上で逐次実行される関数
- `FunctionKind::Kernel(ndim)`: 並列実行されるGPUカーネル（ndimは並列次元数）
- `body`は通常Blockノードで、引数を含むスコープを持つ
- `Function::new()`は自動的にスコープを作成し、`VarKind::Normal`のパラメータをスコープに宣言
- `ThreadId`や`GroupId`などの組み込み変数はスコープには登録されない

## Program - プログラム全体

**責務**: 複数の関数定義を管理し、プログラム全体の整合性を検証

- 関数の集合を`HashMap<String, Function>`で管理
- エントリーポイント関数を指定
- `validate()`でエントリーポイントの存在と全関数本体の妥当性をチェック

## Call / Return

- **Call**: 関数呼び出しを表現。`check_scope()`は引数のみチェック、関数名の存在確認は`Program::validate()`で実施
- **Return**: 関数からの返り値を表現。型推論では返す値の型と同じ型を返す

## Barrier - 同期バリア

**責務**: 並列実行における同期点を表現

- GPU等で全スレッドがこの地点に到達するまで待機
- 同期バリア前の処理完了を保証し、共有メモリアクセスのデータ競合を防ぐ
- 型は`Tuple(vec![])`（unit型）

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

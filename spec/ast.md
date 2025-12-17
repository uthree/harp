# AST (抽象構文木)

## 概要

数値計算を表現するための抽象構文木を提供します。

## 設計思想と方針

### 演算の最小性原則

**既存の演算の組み合わせで表現可能な演算は、原則としてASTノードに実装しない**という設計方針です。演算子の種類を最小限に抑えることで、ASTの単純性・保守性・一貫性を確保します。

実装されている演算: `Add`, `Mul`, `Recip`, `Rem`, `Idiv`, `Max`, 数学関数（`Sqrt`, `Log2`, `Exp2`, `Sin`）, 比較演算（`Lt`, `Le`, `Gt`, `Ge`, `Eq`, `Ne`）

演算子オーバーロードで提供: 減算（`a - b = a + (-b)`）, 除算（`a / b = a * recip(b)`）

### ヘルパー関数とビルダー

`src/ast/helper.rs`にAST構築を簡潔にするヘルパー関数群を提供しています。

#### 演算ヘルパー（マクロ生成）
- **二項演算**: `max`, `idiv`, `rem`
- **単項演算**: `recip`, `sqrt`, `log2`, `exp2`, `sin`
- **比較演算**: `lt`, `le`, `gt`, `ge`, `eq`, `ne`（Bool型を返す）

#### 構造化ヘルパー
- **変数**: `var("x")` - 変数参照
- **定数**: `const_int(42)`, `const_f32(3.14)` - 型付き定数
- **制御構造**: `range`, `block`, `empty_block`, `if_then`, `if_then_else` - ループ、ブロック、条件分岐
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

### If
条件分岐を表現。GPUカーネルの境界チェック等に使用されます。

```rust
AstNode::If {
    condition: Box<AstNode>,        // 条件式（Bool型）
    then_body: Box<AstNode>,        // 条件が真の場合の処理
    else_body: Option<Box<AstNode>>, // 条件が偽の場合の処理（オプション）
}
```

GPUのSIMTアーキテクチャでは、スレッドグループ内でプログラムカウンタが共通のため、境界チェックによるオーバーヘッドは最後のグループのみに影響し、全体的なパフォーマンスへの影響は最小限です。

### Function
`AstNode`の一つのバリアントとして実装されています。

通常のCPU関数を表現します。

```rust
AstNode::Function {
    name: Option<String>,    // 関数名（匿名関数の場合はNone）
    params: Vec<VarDecl>,    // 引数リスト
    return_type: DType,      // 返り値の型
    body: Box<AstNode>,      // 関数本体
}
```

### Kernel
GPUカーネル関数を表現します。GPU上で並列実行されます。

組み込み変数（`GroupId`, `LocalId`等）はスコープに登録せず、特別扱いします。

```rust
AstNode::Kernel {
    name: Option<String>,                     // カーネル名
    params: Vec<VarDecl>,                     // 引数リスト
    return_type: DType,                       // 返り値の型
    body: Box<AstNode>,                       // カーネル本体
    default_grid_size: [Box<AstNode>; 3],     // 推奨グリッド数 (x, y, z)
    default_thread_group_size: [Box<AstNode>; 3], // 推奨スレッドグループサイズ (x, y, z)
}
```

dispatch設定（`default_grid_size`, `default_thread_group_size`）は3次元で表現されます。1次元/2次元のみ使用する場合は、使わない軸を`1`に設定します。これらの値は`CallKernel`を生成する際のデフォルト/ヒントとして使用されます。

### CallKernel
GPUカーネルの呼び出しを表現します。実際のdispatch情報を指定します。

```rust
AstNode::CallKernel {
    name: String,                       // 呼び出すカーネル名
    args: Vec<AstNode>,                 // 引数（バッファポインタ等）
    grid_size: [Box<AstNode>; 3],       // グリッド数 (x, y, z)
    thread_group_size: [Box<AstNode>; 3], // スレッドグループサイズ (x, y, z)
}
```

`Kernel`が定義、`CallKernel`が呼び出しを表現します。同一カーネルを異なるdispatch設定で複数回呼び出すことが可能です。`grid_size`と`thread_group_size`は`AstNode`として表現されるため、実行時に決まる動的な値（例: 入力サイズに依存する計算）も記述できます。

#### ヘルパー関数
```rust
// 3D dispatch
kernel(name, params, return_type, body, default_grid_size, default_thread_group_size)
call_kernel(name, args, grid_size, thread_group_size)

// 1D dispatch（y, z軸を1に設定）
kernel_1d(name, params, return_type, body, default_grid_size, default_thread_group_size)
call_kernel_1d(name, args, grid_size, thread_group_size)
```

### Program
`AstNode`の一つのバリアントとして実装されています。

プログラム全体を表現し、複数のカーネル関数（`AstNode::Kernel`のリスト）と実行順序情報を管理します。

```rust
AstNode::Program {
    functions: Vec<AstNode>,  // AstNode::Kernel のリスト
    execution_order: Option<Vec<KernelExecutionInfo>>,  // 実行順序情報（オプション）
}
```

#### KernelExecutionInfo

カーネルの実行順序を管理するための構造体です。

```rust
pub struct KernelExecutionInfo {
    pub kernel_name: String,    // カーネル名
    pub inputs: Vec<String>,    // 入力バッファ名
    pub outputs: Vec<String>,   // 出力バッファ名
    pub wave_id: usize,         // 実行波ID（同じwave_idは並列実行可能）
}
```

- **wave_id**: 同じ`wave_id`を持つカーネルは依存関係がなく、並列実行可能です
- **依存関係**: `wave_id = 0`のカーネルが先に実行され、その後`wave_id = 1`、`wave_id = 2`...と順に実行されます

#### 実行順序の生成と利用

1. **生成**: `KernelMergeSuggester`がカーネルをマージする際に、Producer/Consumer関係から`execution_order`を生成
2. **収集**: `Lowerer`が複数のKernelノードを収集する際に、`execution_order`をマージ
3. **変換**: `Backend`の`compile_program()`が`execution_order`を`execution_waves`に変換

**注意**: `execution_order`は後方互換性のため`Option`型です。`None`の場合は、各カーネルを順次実行（sequential）として扱います。

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

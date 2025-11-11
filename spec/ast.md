# AST (抽象構文木)

## 概要

数値計算を表現するための抽象構文木を提供します。

## 設計思想と方針

### 演算の最小性原則

**既存の演算の組み合わせで表現可能な演算は、原則としてASTノードに実装しない**という設計方針です。演算子の種類を最小限に抑えることで、ASTの単純性・保守性・一貫性を確保します。

実装されている演算: `Add`, `Mul`, `Recip`, `Rem`, `Idiv`, `Max`, 数学関数（`Sqrt`, `Log2`, `Exp2`, `Sin`）

演算子オーバーロードで提供: 減算（`a - b = a + (-b)`）, 除算（`a / b = a * recip(b)`）

## Scopeと変数管理

変数の宣言と型管理を担当します。

### 並列アクセスの安全性

変数のアクセス安全性は`Mutability`によって管理されます：
- **Immutable**: 読み取り専用（複数スレッドから安全にアクセス可能）
- **Mutable**: 書き込み可能（単一スレッドのみ、排他制御が必要）

この単純な2値による管理により、並列実行時のデータ競合を防ぎます。

## 主要コンポーネントの責務

### Block
複数の文をグループ化し、独立したスコープを提供。型推論では最後の文の型を返します（空なら`Tuple(vec![])`）。

### Range
範囲ベースのループを表現。ループ変数は自動的にスコープに宣言され、親スコープの変数にもアクセス可能です。

### Function
**v0.2.0以降**: `AstNode`の一つのバリアントとして実装されています。

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
**v0.2.0以降**: `AstNode`の一つのバリアントとして実装されています。

プログラム全体を表現し、複数の関数定義（`AstNode::Function`のリスト）を管理。エントリーポイント関数から実行が開始されます。

```rust
AstNode::Program {
    functions: Vec<AstNode>,  // Function ノードのリスト
    entry_point: String,      // エントリーポイント関数名
}
```

`get_function(name)` と `get_entry()` メソッドで関数を取得できます。

## AST統一化のメリット

**v0.2.0の重要な変更**: `Function`と`Program`が`AstNode`のバリアントになったことで、以下が可能になりました：

1. **関数を跨いだ最適化**
   - インライン展開: 小さな関数を呼び出し元に展開
   - 関数融合: 連続した関数呼び出しを1つに統合
   - 未使用関数の削除: エントリーポイントから到達不可能な関数を除去

2. **統一的なAST操作**
   - `children()`, `map_children()`, `infer_type()`, `check_scope()` が全てのノードに適用可能
   - パターンマッチング（`ast/pat.rs`）が関数・プログラムレベルでも機能

3. **コードの簡略化**
   - 異なる型を扱う必要がなくなり、LowererとRendererがシンプルに

### Barrier
並列実行における同期点。GPU等で全スレッドがこの地点に到達するまで待機し、共有メモリアクセスのデータ競合を防ぎます。

## DType型変換

SIMD対応のベクトル型（`Vec<T, N>`）とメモリバッファ用のポインタ型（`Ptr<T>`）を提供。型変換メソッド（`to_vec`, `to_ptr`等）により、型を自由にネスト可能です（例: `Vec<Ptr<F32>>`, `Ptr<Vec<F32>>`）。

## AST最適化

ASTノードに対する代数的最適化機能を`src/opt/ast/`に実装しています。詳細は[最適化仕様書](opt.md#ast最適化)を参照してください。

### パターンマッチングと書き換え

`src/ast/pat.rs`にASTパターンマッチングと書き換えルールの基礎機能を実装：
- `AstPattern`: ワイルドカードを含むパターン表現
- `AstRewriteRule`: パターンマッチと書き換えのルール
- `AstRewriter`: ルール集を適用する書き換え器
- `astpat!`マクロ: パターンマッチング用の便利マクロ

### 最適化フレームワーク

`src/opt/ast/`に体系的な最適化フレームワークを実装：
- **CostEstimator**: ASTの実行コストを推定
- **Optimizer**: ASTを最適化（RuleBaseOptimizer、BeamSearchOptimizer）
- **Suggester**: 複数の書き換え候補を提案（ビームサーチ用）
- **ルール集**: 代数的書き換えルール（単位元、交換則、結合則、分配則、定数畳み込みなど）

**GenericPipelineでの最適化**:
- プログラム全体（`AstNode::Program`）を一つの単位として最適化
- 2段階最適化: (1) ルールベース最適化 → (2) ビームサーチ最適化
- ビームサーチでは`all_rules_with_search()`を使用し、交換則・分配則を含む完全なルール集で探索
- 最適化履歴は`"program"`キーで保存され、可視化ツールで表示可能

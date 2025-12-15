# 最適化

## 概要

Harpの最適化システムは、GraphレベルとASTレベルの2階層で最適化を行います。

## 階層構造

```
Graph (高レベル表現)
  ↓ Lowerer
AST (低レベル表現)
  ↓ Backend Compiler
機械語
```

### Graph最適化（高レベル）
- 演算の融合（Fusion）
- メモリレイアウトの最適化（View操作）
- 並列化戦略の選択

### AST最適化（低レベル）
GenericPipelineでは以下の2段階で実行される：
1. **ルールベース最適化**: 代数的書き換え（定数畳み込み、単位元除去など）
2. **ビームサーチ最適化**: ループ最適化（タイル化、インライン展開、交換など）と関数インライン展開

## 設計思想

### ビームサーチベースの最適化

両階層ともビームサーチを用いた探索ベースの最適化を採用:

1. **Suggester** - 書き換え候補を生成（`SuggestResult`/`AstSuggestResult`を返す）
2. **Selector** - 候補の評価と選択を担当（CostEstimatorを内包）
3. **Optimizer** - ビームサーチで最良の変換列を探索

### SuggestResult構造

Suggesterは単なるGraph/ASTではなく、メタ情報を含む構造体を返す：

```rust
// Graph用
pub struct SuggestResult {
    pub graph: Graph,
    pub suggester_name: String,  // 提案元のSuggester名
    pub description: String,     // 変換内容の説明
}

// AST用
pub struct AstSuggestResult {
    pub ast: AstNode,
    pub suggester_name: String,
    pub description: String,
}
```

`description`フィールドには具体的な変換内容を記述できる（例: "Fuse Add and Mul nodes"）。Visualizerで候補を比較する際に有用。

### Selector設計

OptimizerはSelectorを通じて候補選択を行います：

```
Optimizer
  └── Selector
        └── CostEstimator
```

- Graph用とAST用で別々のtraitを提供（`GraphSelector`、`AstSelector`）
- デフォルトは静的コストベースの選択器（`GraphCostSelector`、`AstCostSelector`）
- 実測値ベースの選択器も利用可能（`GraphRuntimeSelector`、`RuntimeSelector`）

### プラグイン可能な設計

- 新しいルールやSuggesterを簡単に追加可能
- CompositeSuggesterで複数の最適化を組み合わせ可能

## ファイル構成

### 共通
- `src/opt/selector.rs` - Selector実装
  - `GraphSelector`, `AstSelector` - 型安全な選択器trait
  - `GraphCostSelector`, `AstCostSelector` - 静的コストベース選択器
  - `GraphRuntimeSelector`, `RuntimeSelector` - 実測値ベース選択器

### AST最適化
- `src/opt/ast/` - AST最適化の実装
  - `rules.rs` - 代数的書き換えルール集
  - `optimizer.rs` - RuleBaseOptimizer、BeamSearchOptimizer
  - `suggesters/` - 各種Suggester実装
  - `transforms.rs` - ループ変換関数

### Graph最適化
- `src/opt/graph/` - Graph最適化の実装
  - `optimizer.rs` - BeamSearchGraphOptimizer、ChainedGraphOptimizer
  - `suggesters/` - 各種Suggester実装

## 詳細仕様

- [AST最適化の詳細](opt-ast.md)
- [Graph最適化の詳細](opt-graph.md)

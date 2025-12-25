# 最適化

## 概要

Harpの最適化システムは、ASTレベルで最適化を行います。

## 階層構造

```
Tensor (高レベル表現)
  ↓ TensorLowerer
AST (低レベル表現)
  ↓ AST Optimizer
AST (最適化済み)
  ↓ Backend Compiler
機械語
```

### AST最適化
Pipelineでは以下の2段階で実行される：
1. **ルールベース最適化**: 代数的書き換え（定数畳み込み、単位元除去など）
2. **ビームサーチ最適化**: ループ最適化（タイル化、インライン展開、交換など）と関数インライン展開

## 設計思想

### ビームサーチベースの最適化

ビームサーチを用いた探索ベースの最適化を採用:

1. **Suggester** - 書き換え候補を生成（`AstSuggestResult`を返す）
2. **Selector** - 候補の評価と選択を担当（CostEstimatorを内包）
3. **Optimizer** - ビームサーチで最良の変換列を探索

### SuggestResult構造

Suggesterは単なるASTではなく、メタ情報を含む構造体を返す：

```rust
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

- デフォルトは静的コストベースの選択器（`AstCostSelector`）
- 実測値ベースの選択器も利用可能（`RuntimeSelector`）

### プラグイン可能な設計

- 新しいルールやSuggesterを簡単に追加可能
- CompositeSuggesterで複数の最適化を組み合わせ可能

## ファイル構成

### AST最適化
- `src/opt/ast/` - AST最適化の実装
  - `rules.rs` - 代数的書き換えルール集
  - `optimizer.rs` - RuleBaseOptimizer、BeamSearchOptimizer
  - `suggesters/` - 各種Suggester実装
  - `transforms.rs` - ループ変換関数

## 詳細仕様

- [AST最適化の詳細](opt-ast.md)

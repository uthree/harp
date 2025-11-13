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
- 代数的書き換え（定数畳み込み、単位元除去など）
- ループ最適化（タイル化、インライン展開）

## 設計思想

### ビームサーチベースの最適化

両階層ともビームサーチを用いた探索ベースの最適化を採用:

1. **Suggester** - 書き換え候補を生成
2. **CostEstimator** - 各候補のコストを推定
3. **Optimizer** - ビームサーチで最良の変換列を探索

### プラグイン可能な設計

- 新しいルールやSuggesterを簡単に追加可能
- CompositeSuggesterで複数の最適化を組み合わせ可能

## ファイル構成

### AST最適化
- `src/opt/ast/` - AST最適化の実装
  - `rules.rs` - 代数的書き換えルール集
  - `optimizer.rs` - RuleBaseOptimizer、BeamSearchOptimizer
  - `suggester.rs` - 各種Suggester実装
  - `transforms.rs` - ループ変換関数

### Graph最適化
- `src/opt/graph/` - Graph最適化の実装
  - `optimizer.rs` - BeamSearchGraphOptimizer
  - `suggesters/` - 各種Suggester実装
    - `fusion.rs` - 演算融合
    - `view.rs` - View挿入最適化
    - `parallel.rs` - 並列化戦略変更

## 詳細仕様

- [AST最適化の詳細](opt-ast.md)
- [Graph最適化の詳細](opt-graph.md)

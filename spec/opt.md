# 最適化

## ファイル構成
- `src/opt/mod.rs` - 最適化モジュールの定義
- `src/opt/ast/mod.rs` - AST最適化（未実装）
- `src/opt/graph/mod.rs` - グラフ最適化（未実装）

## 実装状況

### グラフ最適化
未実装。将来的に実装予定。

### AST最適化
未実装。以下のトレイトを実装予定:

- `trait opt::ast::Optimizer` - ASTを書き換える
- `trait opt::ast::Suggester` - 複数の書き換え候補を提案する（ビームサーチ用）
- `trait opt::ast::CostEstimator` - ASTの実行コストを推定する

### 注記
現在、AST最適化の基礎機能として`src/ast/pat.rs`にAstRewriteRuleとAstRewriterが実装されていますが、`opt`モジュールとしての体系的な最適化フレームワークは未実装です。
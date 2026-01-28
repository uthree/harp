# Eclat 仕様書

Eclatはテンソル計算グラフからGPU/CPUカーネルを生成するトランスパイラです。

## ドキュメント一覧

- [アーキテクチャ概要](./architecture.md) - システム全体の構成
- [データ型](./types.md) - DType、Literal、Expr等の型定義
- [計算グラフ](./graph.md) - GraphNode、GraphOp、View
- [AST](./ast.md) - AstNode、カーネル表現
- [最適化](./optimization.md) - 最適化パス、サジェスター
- [バックエンド](./backends.md) - 各バックエンドの仕様
- [自動微分](./autograd.md) - 勾配計算

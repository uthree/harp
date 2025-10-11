# Harp 仕様書

このディレクトリには、Harpプロジェクトの詳細な仕様書が含まれています。

## 仕様書の構成

### [00_overview.md](00_overview.md)
プロジェクト全体の概要、アーキテクチャ、設計原則を説明します。

**内容:**
- プロジェクト概要
- アーキテクチャ図
- モジュール構成
- データフロー
- 設計原則

### [01_graph.md](01_graph.md)
計算グラフの仕様を説明します。

**内容:**
- GraphNode, GraphOp, Graph の型定義
- 演算の種類（Elementwise, Reduce, Cumulative）
- シェイプとビューの表現
- ビュー操作とContiguous操作の違い
- グラフシグネチャ

### [02_ast.md](02_ast.md)
抽象構文木（AST）の仕様を説明します。

**内容:**
- ASTノードの設計思想（演算子の最小化）
- AstNode, DType, ConstLiteral の型定義
- 演算子のオーバーロード
- スコープと変数宣言
- 関数とプログラム
- ノード操作とパターンマッチング

### [03_lowerer.md](03_lowerer.md)
計算グラフをASTに変換するLowererの仕様を説明します。

**内容:**
- Lowererの構造
- 変換プロセス（トポロジカルソート、ノード変換）
- 各演算のループ生成方法
- メモリアクセスパターン
- コピーループとFoldループの生成
- サブモジュール（elementwise, reduce, cumulative, fused）

### [04_backend.md](04_backend.md)
バックエンドの抽象化と実装の仕様を説明します。

**内容:**
- トレイト階層（Buffer, Kernel, Compiler, Renderer, Backend）
- Cバックエンド実装
  - CBuffer: メモリ管理
  - CRenderer: C言語コード生成
  - CCompiler: コンパイルと動的ロード
  - CKernel: カーネル実行
- バックエンドの選択と検証

### [05_tensor.md](05_tensor.md)
ユーザー向けTensor APIの仕様を説明します。

**内容:**
- TensorType, Dimension トレイト
- Tensor<T, D> の型パラメータ
- テンソルの作成と操作
- 勾配管理と自動微分
- 型安全性とコンパイル時チェック

### [06_optimization.md](06_optimization.md)
最適化モジュールの仕様を説明します。

**内容:**
- グラフレベル最適化
  - 演算融合（Elementwise, Reduce, Cumulative）
  - 融合の制約とアルゴリズム
- ASTレベル最適化
  - 定数畳み込み
  - 代数的簡約化
  - ヒューリスティック最適化（ビームサーチ）
  - コスト推定
  - 各種変換提案（代数的、ループ、ビット演算など）

## 読み進め方

### 初めて読む場合

1. **[00_overview.md](00_overview.md)** - 全体像を把握
2. **[05_tensor.md](05_tensor.md)** - ユーザーAPIを理解
3. **[01_graph.md](01_graph.md)** - 計算グラフの表現を理解
4. その他のドキュメントを必要に応じて参照

### 実装者向け

1. **[00_overview.md](00_overview.md)** - アーキテクチャ理解
2. **[01_graph.md](01_graph.md)** - グラフ表現の詳細
3. **[02_ast.md](02_ast.md)** - AST設計思想
4. **[03_lowerer.md](03_lowerer.md)** - 変換プロセス
5. **[06_optimization.md](06_optimization.md)** - 最適化手法
6. **[04_backend.md](04_backend.md)** - バックエンド実装

### 特定の機能を実装する場合

#### 新しい演算の追加
1. [01_graph.md](01_graph.md) - GraphOpに追加
2. [03_lowerer.md](03_lowerer.md) - Lowererで変換実装
3. [05_tensor.md](05_tensor.md) - Tensor APIに追加

#### 新しい最適化の追加
1. [06_optimization.md](06_optimization.md) - 最適化手法を理解
2. [02_ast.md](02_ast.md) - ASTパターンマッチング
3. opt/ast/heuristic/suggester/ に実装

#### 新しいバックエンドの追加
1. [04_backend.md](04_backend.md) - トレイト要件を理解
2. Buffer, Renderer, Compiler, Kernel を実装
3. Backend トレイトを実装

## 仕様書の更新

コードを変更した際は、対応する仕様書も更新してください。

### 更新手順

1. コードの変更を実施
2. 影響を受ける仕様書を特定
3. 仕様書を更新
4. コードと仕様書の整合性を確認

### 不整合を見つけた場合

- Issue を作成するか、直接修正してPull Requestを送ってください
- 仕様書とコードのどちらが正しいか明記してください

## その他のドキュメント

- **note/** - 開発時のメモ（デザインノート、実装メモなど）
- **README.md** - プロジェクトの概要とクイックスタート
- **examples/** - サンプルコードとユースケース
- **tests/** - テストケースとその説明

## 参考資料

### 関連プロジェクト
- [tinygrad](https://github.com/tinygrad/tinygrad) - インスピレーション元
- [luminal](https://github.com/luminal-ai/luminal) - インスピレーション元

### 関連技術
- JITコンパイル
- 自動微分
- テンソル計算
- グラフ最適化
- ループ最適化

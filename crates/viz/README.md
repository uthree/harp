# Harp Visualizer

Harpの計算グラフとAST最適化の各ステップを可視化するためのGUIツールです。

## 機能

### グラフビューア
- **計算グラフの構造を可視化**: ノードとエッジの視覚的表示
- **グラフ最適化履歴**: ビームサーチ最適化の各ステップを閲覧
- **ステップナビゲーション**: 前後のステップを移動して最適化の過程を確認
- **コスト遷移グラフ**: 最適化の各ステップでのコストの変化を折れ線グラフで可視化
- **DOTテキスト表示**: Graphviz DOT形式でのグラフ表現をリアルタイム表示

### ASTビューア
- **AST最適化履歴の可視化**: ビームサーチ最適化の各ステップを閲覧
- **ビーム内の候補表示**: 各ステップのビーム内の全候補をランク付きで表示
- **レンダリングされたコード表示**: 選択したASTを読みやすいコード形式で表示
- **コスト遷移グラフ**: 最適化の各ステップでのコストの変化を可視化

## デモの実行方法

### 統合最適化可視化デモ

グラフ最適化とAST最適化の両方を可視化する統合デモを実行するには：

```bash
cargo run --package harp-viz --example optimization_demo
```

このデモでは以下のことが行われます：

**グラフ最適化**:
1. サンプルの計算グラフを作成（`y = ((a + b) * c) - d`, `z = reduce_sum(y, axis=0)`）
2. ビームサーチ最適化器で最適化を実行
3. 最適化の各ステップを記録

**AST最適化**:
1. サンプルのASTを作成（`((2 + 3) * 1) + ((a + 0) * (b + c))`）
2. 代数的書き換えルールを適用してビームサーチ最適化を実行
3. 最適化の各ステップを記録

**可視化**:
- Graph Viewerタブ: グラフ最適化の履歴を表示
- AST Viewerタブ: AST最適化の履歴を表示
- タブを切り替えて両方の最適化過程を確認可能

### 操作方法

#### グラフビューア
- **◀ Prev / Next ▶**: 前後のステップに移動
- **Step表示**: 現在のステップ番号
- **Description**: 各ステップの説明
- **Cost**: 推定実行コスト
- **Show/Hide DOT Text**: DOT形式のテキストを画面に表示/非表示
  - DOTテキスト表示中は「Copy to Clipboard」ボタンでクリップボードにコピー可能
- **Show/Hide Cost Graph**: 最適化ステップごとのコスト遷移を折れ線グラフで表示/非表示
  - 現在のステップが赤い縦線で表示されます

#### ASTビューア
- **◀ Prev / Next ▶**: 前後のステップに移動
- **Stepスライダー**: 任意のステップに直接ジャンプ
- **Beam Candidates**: ビーム内の候補リスト（ランクとコストを表示）
  - クリックして候補を選択すると、右側にそのASTのコードが表示されます
- **AST Code**: 選択したASTのレンダリングされたコード
  - 構文ハイライト付きのモノスペースフォントで表示
- **Show/Hide Cost Graph**: コスト遷移の折れ線グラフを表示/非表示

### DOT形式での出力

グラフをGraphviz DOT形式で利用する方法：

1. **画面に表示**: 「Show DOT Text」ボタンをクリックして、右側にDOTテキストを表示
2. **クリップボードにコピー**: DOTテキスト表示中に「Copy to Clipboard」ボタンをクリック

コピーしたDOTテキストは外部ツールで可視化できます：

```bash
# クリップボードからファイルに保存
pbpaste > graph.dot  # macOS
# または
xclip -o > graph.dot  # Linux

# Graphvizでレンダリング
dot -Tpng graph.dot -o graph.png

# または SVG形式で
dot -Tsvg graph.dot -o graph.svg
```

プログラムから直接エクスポートすることもできます：

```rust
use harp::graph::Graph;

let mut graph = Graph::new();
// ... グラフを構築 ...

// DOT形式で出力
let dot_string = graph.to_dot();
println!("{}", dot_string);

// ファイルに保存
graph.save_dot("my_graph.dot").unwrap();
```

## ライブラリとして使用する

### グラフ最適化履歴の可視化

```rust
use harp_viz::HarpVizApp;
use harp::opt::graph::OptimizationHistory;

// グラフ最適化履歴を作成
let (optimized_graph, history) = optimizer.optimize_with_history(graph);

// 可視化アプリケーションで表示
let mut app = HarpVizApp::new();
app.load_graph_optimization_history(history);
```

### AST最適化履歴の可視化

```rust
use harp_viz::HarpVizApp;
use harp::opt::ast::OptimizationHistory;

// AST最適化履歴を作成
let (optimized_ast, history) = optimizer.optimize_with_history(ast);

// 可視化アプリケーションで表示
let mut app = HarpVizApp::new();
app.load_ast_optimization_history(history);
```

### カスタムレンダラーを使用したAST可視化

`AstViewerApp`はジェネリックで、任意のバックエンドレンダラーを使用できます：

```rust
use harp_viz::AstViewerApp;
use harp::backend::metal::MetalRenderer;  // または他のレンダラー
use harp::opt::ast::OptimizationHistory;

// MetalRendererを使用してAST Viewerを作成
let renderer = MetalRenderer::new();
let mut ast_viewer = AstViewerApp::with_renderer(renderer);

// 最適化履歴を読み込む
ast_viewer.load_history(history);

// または、デフォルトのCRendererを使用
let mut ast_viewer = AstViewerApp::new();  // デフォルトでCRendererを使用
```

### GenericPipelineとの統合

`GenericPipeline`は最適化履歴を自動的に記録できます。記録された履歴を可視化するには：

```rust
use harp::backend::GenericPipeline;
use harp_viz::HarpVizApp;

// パイプラインを作成してグラフをコンパイル
let mut pipeline = GenericPipeline::new(renderer, compiler);

// 最適化を実行して履歴を記録
let (optimized_graph, graph_history) = graph_optimizer.optimize_with_history(graph);
pipeline.set_graph_optimization_history(graph_history);

let (optimized_ast, ast_history) = ast_optimizer.optimize_with_history(ast);
pipeline.set_ast_optimization_history(ast_history);

// 可視化アプリで表示
let mut viz_app = HarpVizApp::new();

// 方法1: 履歴を参照として読み込む（Pipelineに履歴が残る）
viz_app.load_from_pipeline(&pipeline);

// 方法2: 履歴を移動する（Pipelineから履歴がクリアされる）
viz_app.take_from_pipeline(&mut pipeline);
```

## 依存関係

- `egui`: GUIフレームワーク
- `eframe`: eGUIのネイティブバックエンド
- `egui-snarl`: グラフ構造の可視化
- `egui_plot`: グラフのプロット表示
- `harp`: Harpライブラリ本体

## タブの切り替え

アプリケーション上部のメニューバーから以下のタブを切り替えできます：

- **Graph Viewer**: 計算グラフの可視化
- **AST Viewer**: AST最適化の可視化
- **Performance**: パフォーマンス統計（将来の拡張）

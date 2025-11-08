# Harp Visualizer

Harpの計算グラフとグラフ最適化の各ステップを可視化するためのGUIツールです。

## 機能

- **グラフビューア**: 計算グラフの構造を可視化
- **最適化履歴**: ビームサーチ最適化の各ステップを閲覧
- **ステップナビゲーション**: 前後のステップを移動して最適化の過程を確認
- **コスト遷移グラフ**: 最適化の各ステップでのコストの変化を折れ線グラフで可視化
- **DOTテキスト表示**: Graphviz DOT形式でのグラフ表現をリアルタイム表示

## デモの実行方法

### 最適化可視化デモ

グラフ最適化の各ステップを可視化するデモを実行するには：

```bash
cargo run --package harp-viz --example optimization_demo
```

このデモでは以下のことが行われます：

1. サンプルの計算グラフを作成（`y = ((a + b) * c) - d`, `z = reduce_sum(y, axis=0)`）
2. ビームサーチ最適化器で最適化を実行
3. 最適化の各ステップを記録
4. GUIで各ステップを可視化
   - グラフ構造の視覚的表示
   - コスト遷移の折れ線グラフ
   - DOT形式テキストの表示とコピー

### 操作方法

- **◀ Prev**: 前のステップに戻る
- **Next ▶**: 次のステップに進む
- **Step表示**: 現在のステップ番号
- **Description**: 各ステップの説明
- **Cost**: 推定実行コスト
- **Show/Hide DOT Text**: DOT形式のテキストを画面に表示/非表示
  - DOTテキスト表示中は「Copy to Clipboard」ボタンでクリップボードにコピー可能
- **Show/Hide Cost Graph**: 最適化ステップごとのコスト遷移を折れ線グラフで表示/非表示
  - 現在のステップが赤い縦線で表示されます

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

```rust
use harp_viz::HarpVizApp;
use harp::opt::graph::OptimizationHistory;

// 最適化履歴を作成
let (optimized_graph, history) = optimizer.optimize_with_history(graph);

// 可視化アプリケーションで表示
let mut app = HarpVizApp::new();
app.load_optimization_history(history);
```

## 依存関係

- `egui`: GUIフレームワーク
- `eframe`: eGUIのネイティブバックエンド
- `egui-snarl`: グラフ構造の可視化
- `harp`: Harpライブラリ本体

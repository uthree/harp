# Harp Visualizer

Harpの計算グラフとグラフ最適化の各ステップを可視化するためのGUIツールです。

## 機能

- **グラフビューア**: 計算グラフの構造を可視化
- **最適化履歴**: ビームサーチ最適化の各ステップを閲覧
- **ステップナビゲーション**: 前後のステップを移動して最適化の過程を確認

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

### 操作方法

- **◀ Prev**: 前のステップに戻る
- **Next ▶**: 次のステップに進む
- **Step表示**: 現在のステップ番号
- **Description**: 各ステップの説明
- **Cost**: 推定実行コスト
- **Export to DOT**: グラフをGraphviz DOT形式でエクスポート

### DOT形式でのエクスポート

グラフをGraphviz DOT形式でエクスポートできます。これにより、外部ツールでグラフを可視化したり、デバッグに利用できます。

```bash
# デモアプリケーションで「Export to DOT」ボタンをクリック
# カレントディレクトリに graph.dot または graph_step_N.dot が生成されます

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

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

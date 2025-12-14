# Harp Visualizer

Harpの計算グラフ最適化の各ステップを可視化するためのGUIツールです。

## 機能

### グラフビューア
- **計算グラフの構造を可視化**: ノードとエッジの視覚的表示
- **グラフ最適化履歴**: ビームサーチ最適化の各ステップを閲覧
- **ステップナビゲーション**: 前後のステップを移動して最適化の過程を確認
- **コスト遷移グラフ**: 最適化の各ステップでのコストの変化を折れ線グラフで可視化
- **DSLテキスト表示**: Harp DSL形式でのグラフ表現をリアルタイム表示
- **Kernelノードのコード表示**: 各ノードの詳細パネルで生成されたコードを確認

### コードビューア
- **最終的な生成コードを表示**: グラフ最適化後のKernel(Program)のコードを表示
- **シンタックスハイライト**: C言語風の構文ハイライト付きで表示
- **コピー機能**: クリップボードにワンクリックでコピー
- **統計情報**: 最適化ステップ数、最終コスト、コード行数を表示

### 最適化アーキテクチャ
グラフ最適化とAST最適化は分離されています：
- **グラフ最適化**: 計算グラフの構造変換（融合、View挿入、Lowering）
- **AST最適化**: 生成されたASTのループ変換、代数的簡約などの後処理

この分離により、各最適化フェーズが独立して改善可能な設計となっています。

## デモの実行方法

### クイックスタート（最も簡単）

Pipelineから直接visualizerを起動する最も簡単な例：

```bash
cargo run --package harp-viz --example quick_viz
```

このデモでは：
- GenericPipelineで最適化を有効化
- サンプルグラフをコンパイル（最適化履歴を自動記録）
- `HarpVizApp::run_from_pipeline()`でワンライナーで可視化

### GenericPipeline統合デモ（推奨）

GenericPipelineを使った複雑な計算グラフの最適化と可視化のデモ：

```bash
cargo run --package harp-viz --example pipeline_demo
```

このデモでは：
- 複雑なテンソル演算を含む計算グラフを構築（128x256の行列演算）
- GenericPipelineで最適化履歴を自動記録
- 統合最適化で大幅なコスト削減を実現
- 最適化過程を可視化UIで確認

### 統合最適化可視化デモ

グラフ最適化を可視化する基本デモを実行するには：

```bash
cargo run --package harp-viz --example optimization_demo
```

このデモでは以下のことが行われます：

**グラフ最適化**:
1. サンプルの計算グラフを作成（`y = ((a + b) * c) - d`, `z = reduce_sum(y, axis=0)`）
2. ビームサーチ最適化器でグラフ最適化を実行
3. 最適化の各ステップを記録

**可視化**:
- Graph Viewerタブ: グラフ最適化の履歴を表示
- Code Viewerタブ: 最終的な生成コードを表示
- タブを切り替えて最適化過程と最終結果を確認

### 操作方法

#### グラフビューア
- **◀ Prev / Next ▶**: 前後のステップに移動
- **Step表示**: 現在のステップ番号
- **Description**: 各ステップの説明
- **Cost**: 推定実行コスト
- **Show/Hide DSL Text**: DSL形式のテキストを画面に表示/非表示
  - DSLテキスト表示中は「Copy to Clipboard」ボタンでクリップボードにコピー可能
- **Show/Hide Cost Graph**: 最適化ステップごとのコスト遷移を折れ線グラフで表示/非表示
  - 現在のステップが赤い縦線で表示されます

#### コードビューア
- **Copy to Clipboard**: 生成コードをクリップボードにコピー
- **コード表示**: シンタックスハイライト付きで最終的な生成コードを表示
- **統計情報**: 最適化ステップ数、最終コスト、コード行数を表示

### DSL形式での出力

グラフをHarp DSL形式で利用する方法：

1. **画面に表示**: 「DSL Format」セクションを展開してDSLテキストを表示
2. **クリップボードにコピー**: 「Copy to Clipboard」ボタンをクリック

プログラムから直接DSL形式に変換することもできます：

```rust
use harp::graph::Graph;
use harp_dsl::decompiler::decompile;

let mut graph = Graph::new();
// ... グラフを構築 ...

// DSL形式で出力
let dsl_string = decompile(&graph, "my_graph");
println!("{}", dsl_string);
```

DSL形式はパース可能なため、他のツールとの連携や再コンパイルに利用できます。

## ライブラリとして使用する

### 最も簡単な使い方（GenericPipelineから直接起動）

```rust
use harp::backend::GenericPipeline;
use harp_viz::HarpVizApp;

let mut pipeline = GenericPipeline::new(renderer, compiler);

// 最適化を有効化
pipeline.enable_graph_optimization = true;
pipeline.collect_histories = true;  // 履歴を記録

// グラフをコンパイル（最適化履歴が自動的に記録される）
let kernel = pipeline.compile_graph(graph)?;

// ワンライナーで可視化ウィンドウを起動
HarpVizApp::run_from_pipeline(&pipeline)?;
```

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

### レンダラーの切り替え

レンダラーは実行時に切り替え可能です。利用可能なレンダラータイプ：
- `RendererType::C` - C言語（CPU、デフォルト）
- `RendererType::OpenCL` - OpenCL（GPU）
- `RendererType::Metal` - Metal（Apple GPU）

```rust
use harp_viz::{HarpVizApp, RendererType};

// 特定のレンダラータイプで起動
let mut app = HarpVizApp::with_renderer_type(RendererType::OpenCL);

// 実行時にレンダラーを切り替え
app.set_renderer_type(RendererType::Metal);
```

UIからも切り替え可能です：
- **Graph Viewer**: サイドパネルのノード詳細でRendererドロップダウンを使用
- **Code Viewer**: 上部のRendererドロップダウンを使用

### GenericPipelineとの統合

`GenericPipeline`は最適化履歴を自動的に記録できます。記録された履歴を可視化するには：

```rust
use harp::backend::GenericPipeline;
use harp_viz::HarpVizApp;

// パイプラインを作成してグラフをコンパイル
let mut pipeline = GenericPipeline::new(renderer, compiler);
pipeline.enable_graph_optimization = true;
pipeline.collect_histories = true;

// グラフをコンパイル（最適化履歴が自動記録される）
let kernel = pipeline.compile_graph(graph)?;

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
- `harp-dsl`: Harp DSL（グラフのテキスト表現）

## タブの切り替え

アプリケーション上部のメニューバーから以下のタブを切り替えできます：

- **Graph Viewer**: 計算グラフの可視化と最適化履歴
- **Code Viewer**: 最終的な生成コードの表示
- **Performance**: パフォーマンス統計（将来の拡張）

## 実装ノート

### egui-snarlのノードクリック検出について

**問題**: egui-snarl 0.6ではノードのクリックイベントが内部のドラッグ・複数選択機能によって消費されるため、`SnarlViewer::show_header`内で`ui.response().clicked()`を使用してもクリックを検出できない。

**解決策**: Kernelノード（生成コードを持つノード）には明示的な`📝`ボタンを追加し、このボタンのクリックで選択を行う実装とした。

**バージョン情報**:
- egui-snarl 0.6: `SnarlState`の`selected_nodes()`メソッドは内部のみで使用され、外部からアクセスするAPIがない
- egui-snarl 0.8.0: `get_selected_nodes()`関数が追加されているが、APIの破壊的変更があるためアップグレードは見送り

**将来の改善**: egui-snarl 0.8.0以降にアップグレードする場合、`get_selected_nodes()`を使用することでノードのクリック選択を実現できる可能性がある。

# 可視化ツール (crates/viz)

## 概要

eclat-vizはグラフ最適化とAST最適化の履歴を可視化するGUIツールで、egui/eframe/egui-snarlをベースに実装されている。

## 機能

- **AST可視化**: 最適化ステップごとの生成コードを構文ハイライト付きで表示
- **グラフ可視化**: 計算グラフをノード・エッジ形式で表示（egui-snarl使用）
- **ステップナビゲーション**: 最適化履歴を前後にスクロール
- **候補比較**: 各ステップで選択された候補と代替候補の比較
- **ビュー切替**: ASTとグラフを個別に表示

## ファイル構成

```
crates/viz/
├── Cargo.toml
└── src/
    ├── lib.rs           # パブリックAPI
    ├── app.rs           # eframe::App実装
    ├── state.rs         # アプリケーション状態
    ├── highlight.rs     # 構文ハイライト
    ├── convert.rs       # GraphNode → Snarl変換
    ├── graph_history.rs # GraphOptimizationHistory再エクスポート
    └── panels/
        ├── mod.rs
        ├── timeline.rs      # ステップナビゲーション
        ├── graph_panel.rs   # グラフ描画
        ├── code_panel.rs    # コード表示
        └── candidates.rs    # 候補リスト
```

## 使用方法

```rust
use eclat::opt::ast::history::OptimizationHistory;
use eclat::opt::graph::{GraphBeamSearchOptimizer, GraphOptimizer};
use eclat_viz::{run, run_with_both};

// AST履歴のみを可視化（デフォルトレンダラー）
let ast_history = OptimizationHistory::new();
run(ast_history)?;

// カスタムレンダラーで可視化（Metal等）
use eclat_backend_metal::MetalRenderer;
let renderer = MetalRenderer::new();
run_with_renderer(ast_history, renderer)?;

// グラフ履歴を記録して可視化
let mut optimizer = GraphBeamSearchOptimizer::new(suggester)
    .with_history();
let result = optimizer.optimize(roots);
let graph_history = optimizer.take_history().unwrap();

run_with_both(ast_history, graph_history, renderer)?;
```

## レンダラーサポート

`run_with_renderer`と`run_with_both`は`CLikeRenderer`トレイトを実装する任意のレンダラーを受け付ける:

- `GenericRenderer` - 汎用Cライクコード
- `MetalRenderer` - Metal Shading Language (macOS)
- `OpenCLRenderer` - OpenCL
- `CUDARenderer` - CUDA

## 依存関係

- egui 0.30
- eframe 0.30
- egui-snarl 0.6
- syntect 5.2

## キーボードショートカット

| キー | 操作 |
|------|------|
| ←/H | 前のステップ |
| →/L | 次のステップ |
| ↑/K | 前の候補 |
| ↓/J | 次の候補 |
| 1 | ASTビュー |
| 2 | グラフビュー |

## ターゲットバックエンド

最適化履歴は`TargetBackend`を保持し、どのバックエンドを対象として最適化が行われたかを記録する。
可視化ツールはこの情報を元に適切なレンダラーを自動選択できる。

```rust
use eclat::backend::TargetBackend;

// オプティマイザにターゲットバックエンドを設定
let mut optimizer = GraphBeamSearchOptimizer::new(suggester)
    .with_target_backend(TargetBackend::Metal)
    .with_history();

// 履歴からターゲットバックエンドを取得
let history = optimizer.take_history().unwrap();
println!("Target: {}", history.target_backend());  // "Metal"
```

利用可能なバックエンド:
- `TargetBackend::Generic` - 汎用C (デフォルト)
- `TargetBackend::Metal` - Metal (macOS GPU)
- `TargetBackend::Cuda` - CUDA (NVIDIA GPU)
- `TargetBackend::OpenCL` - OpenCL
- `TargetBackend::OpenMP` - OpenMP (CPU並列)

## GraphBeamSearchOptimizerの履歴記録

`with_history()`メソッドを呼び出すと、最適化中に以下の情報が記録される:

- 各ステップの選択されたグラフ
- 代替候補（beam内の他の候補）
- コスト、サジェスター名、説明
- ターゲットバックエンド

## AST BeamSearchOptimizerの履歴記録

`optimize_with_history()`メソッドでAST最適化履歴を取得できる。

- `with_record_all_steps(true)` - コスト改善がなくても全ステップを記録（可視化用）
- `with_target_backend(backend)` - ターゲットバックエンドを設定

```rust
let mut optimizer = AstBeamSearchOptimizer::new(suggester)
    .with_target_backend(TargetBackend::Metal)
    .with_record_all_steps(true)
    .with_max_steps(15);

let (result, history) = optimizer.optimize_with_history(ast);
println!("Optimized for: {}", history.target_backend());
```

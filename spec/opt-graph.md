# Graph最適化

ビームサーチベースの最適化フレームワーク。

## トレイト

- **GraphCostEstimator**: Graphの実行コスト推定
- **GraphOptimizer**: Graphの最適化
- **GraphSuggester**: 書き換え候補を提案（ビームサーチ用）

## コスト推定器

| 推定器 | 用途 |
|--------|------|
| SimpleCostEstimator | ノード数とメモリアクセスベースの簡易推定 |
| LoweringCostEstimator | Loweringフェーズ専用（ProgramRoot集約を促進） |
| KernelMergeCostEstimator | カーネルマージ最適化専用 |
| GraphRuntimeCostEstimator | 実測値ベースのコスト評価（Lowering→コンパイル→実行） |

SimpleCostEstimatorは複数のKernel(Function)にペナルティを付与し、単一のKernel(Program)への収束を促進する。

GraphRuntimeCostEstimatorはグラフを簡易Loweringした後、コンパイル・実行して実行時間を計測する。
AST版よりも計測コストが高いため、足切り候補数を少なめに設定することを推奨（デフォルト: 5件）。

## Suggester一覧

### 最適化系
- **FusionSuggester**: 連続するElementwise演算を融合
- **ViewMergeSuggester**: Viewノードを上流ノードにマージ
- **ViewInsertionSuggester**: メモリレイアウト最適化のためのView挿入
- **ContiguousInsertionSuggester**: 非contiguousなViewを実体化
- **TilingSuggester**: ループタイリング適用

### Lowering系
- **LoweringSuggester**: GraphOpをKernel(Function)に変換
  - デフォルトでSequential戦略のみで候補を生成
  - 並列化はAST最適化フェーズ（Global/LocalParallelizationSuggester）で行う
  - テスト用に`with_parallel_strategies()`で複数戦略を有効化可能
- **BufferAbsorptionSuggester**: KernelのsrcにあるBufferを`input_buffers`に取り込む
- **KernelPartitionSuggester**: 1D FlatParallel Kernelを多次元グリッドに分割

#### KernelPartitionSuggester

LoweringSuggesterが生成した1D FlatParallel Kernelを、より効率的な多次元グリッド構成に変換する。

**変換例:**
```text
// 変換前 (1D FlatParallel)
Kernel {
    params: [gidx: GroupId(0), ...],
    body: { if (gidx < total) { ... } },
    grid_size: [ceil_div(N, 256) * 256, 1, 1],
    thread_group_size: [256, 1, 1],
}

// 変換後 (2D Grid)
Kernel {
    params: [gidx0: GroupId(0), gidx1: GroupId(1), ...],
    body: { if (gidx0 < shape_0 && gidx1 < shape_1) { ... } },
    grid_size: [ceil_div(shape_0, 16) * 16, ceil_div(shape_1, 16) * 16, 1],
    thread_group_size: [16, 16, 1],
}
```

**設計方針:**
- Loweringフェーズ後、Absorptionフェーズ前に実行
- GraphOp::Kernelノードに対して直接操作することで、dispatch設定の一貫性を保証
- `parallel_dims_options`: 並列化する軸数の候補（デフォルト: [2, 3]）
- `thread_group_sizes`: スレッドグループサイズの候補（デフォルト: [64, 128, 256]）

**関数:**
- `distribute_thread_group_size(total_size, dims)`: スレッドグループサイズを指定した次元数に均等分配（2のべき乗を維持）

#### ParallelizationStrategy

LoweringSuggesterが生成する並列化戦略：

| 戦略 | 説明 | パラメータ |
|------|------|-----------|
| Sequential | 逐次実行（CPU向け） | なし |
| FlatParallel | 全要素を線形インデックスで1D並列処理（境界チェック付き） | thread_group_size, vector_width |

対応演算:
- **Elementwise/FusedElementwise**: 両戦略対応（ベクトル化含む）
- **Reduce**: Sequential, FlatParallel対応（ベクトル化なし）
- **その他**: Sequentialのみ

多次元グリッドへの変換は`KernelPartitionSuggester`が担当する（Loweringフェーズ後、Absorptionフェーズ前に実行）。

#### LoweringSuggesterの設定

デフォルトではSequential戦略のみで候補を生成する。並列化はAST最適化フェーズで行うことを推奨。

```rust
// デフォルト設定（Sequential戦略のみ）
let suggester = LoweringSuggester::new();

// 後方互換性のためのエイリアス（new()と同じ）
let suggester = LoweringSuggester::sequential_only();

// テスト・ベンチマーク用: 並列戦略を有効化
// - thread_group_sizes: [64, 128, 256, 512]
// - vector_widths: [2, 4, 8]
let suggester = LoweringSuggester::with_parallel_strategies();

// カスタム設定（並列戦略有効時のみ意味がある）
let suggester = LoweringSuggester::with_parallel_strategies()
    .with_thread_group_sizes(vec![128, 256])  // サイズを制限
    .with_vector_widths(vec![4]);             // float4のみ
```

FlatParallel戦略は境界チェック（`if (tid < total_elements)`）を含むため、総要素数がスレッドグループサイズで割り切れなくても適用可能。

### ProgramRoot関連
- **ProgramRootAbsorptionSuggester**: Kernel(Function)をProgramRootに吸収
- **ProgramRootBufferAbsorptionSuggester**: ProgramRootのsrcから入力Bufferを除去

### マージ・統合系
- **KernelMergeSuggester**: 依存関係のあるKernelをペアワイズでマージ
- **CompositeSuggester**: 複数Suggesterを組み合わせ

## Suggesterの連携

```
ViewMergeSuggester → ProgramRootAbsorptionSuggester
```
ProgramRootAbsorptionSuggesterはViewを透過的に扱わないため、ViewMergeSuggesterが先に適用される必要がある。

## 最適化フロー

```
FusionSuggester        : Elementwise演算の融合
       ↓
LoweringSuggester      : GraphOp → Kernel(Function) [Sequential]
       ↓
BufferAbsorptionSuggester : 入力Bufferの取り込み
       ↓
ProgramRootAbsorptionSuggester : Kernel(Function) → ProgramRoot(Program)
       ↓
ProgramRootBufferAbsorptionSuggester : 入力Bufferの除去
       ↓
=== AST最適化フェーズ ===
       ↓
Group/LocalParallelizationSuggester : Function → Kernel (並列化)
       ↓
LoopInterchangeSuggester + Group/LocalParallelizationSuggester : 追加並列化
```

## 最適化モード

### SuggesterFlags

| フラグ | kernel_merge | 説明 |
|--------|--------------|------|
| `new()` | false | 基本最適化のみ |
| `single_stage()` | true | 単一ステージ最適化（推奨） |

### 単一ステージ最適化（推奨）

`SuggesterFlags::single_stage()`でKernelMergeSuggesterを含めた最適化を実行。

```rust
let flags = SuggesterFlags::single_stage();
let (graph, history) = optimize_graph_with_history(
    graph, flags, SimpleCostEstimator::new(), 8, 200, true
);
```

### グラフ最適化とAST最適化の分離

グラフ最適化はグラフ構造の変換（融合、View挿入、Lowering）を担当し、
AST最適化（ループ変換、代数的簡約など）はLowering完了後に独立して実行される。
この分離により、各最適化フェーズが独立して改善可能な設計となっている。

### Lowering Optimizer

グラフからProgramへの変換は2種類のOptimizerで行う。

| 関数 | 説明 |
|------|------|
| `create_lowering_optimizer(beam_width, max_steps)` | マルチフェーズ最適化（ビームサーチ、複数の並列化戦略） |
| `create_simple_lowering_optimizer(max_steps)` | 貪欲法で高速（ビーム幅=1、Sequential戦略のみ） |

```rust
use harp::lowerer::{create_lowering_optimizer, create_simple_lowering_optimizer};
use harp::opt::graph::GraphOptimizer;

// 通常のOptimizer（最適化あり）
let optimizer = create_lowering_optimizer(8, 3000);
let (optimized, history) = optimizer.optimize_with_history(graph);

// 高速Optimizer（実測用）
let optimizer = create_simple_lowering_optimizer(5000);
let (optimized, history) = optimizer.optimize_with_history(graph);
```

`create_simple_lowering_optimizer`の用途:
- 実行時間の実測によるコスト評価（`MultiStageSelector`との組み合わせ）
- 最適化の初期候補生成
- デバッグ・テスト用途

## Factory モジュール (`opt/graph/factory.rs`)

グラフオプティマイザのファクトリ関数を提供。マルチフェーズ最適化や各種Suggesterを組み合わせたオプティマイザを作成する。

### 主要関数

| 関数 | 説明 |
|------|------|
| `create_multi_phase_optimizer(config)` | 7フェーズの最適化パイプラインを作成 |
| `create_multi_phase_optimizer_with_selector(config, selector)` | カスタムSelectorを使用するパイプライン |
| `create_greedy_optimizer(config)` | 貪欲法オプティマイザ（高速、ビーム幅=1） |
| `optimize_graph_multi_phase(graph, config)` | マルチフェーズ最適化を直接実行 |
| `optimize_graph_greedy(graph, max_steps)` | 貪欲法最適化を直接実行 |

### Suggesterファクトリ

| 関数 | 説明 |
|------|------|
| `create_subgraph_inlining_suggester()` | サブグラフインライン展開用 |
| `create_view_merge_only_suggester()` | ViewMergeのみ |
| `create_graph_optimization_suggester()` | グラフ構造最適化（View挿入、タイリング等） |
| `create_lowering_only_suggester()` | Lowering用（Sequentialのみ） |
| `create_lowering_only_suggester_with_simd(widths)` | SIMD幅指定付きLowering |
| `create_kernel_partition_suggester()` | カーネル分割用 |
| `create_fusion_suggester()` | 吸収・マージ用 |
| `create_ast_loop_suggester()` | ASTループ最適化用 |

### 設定 (`MultiPhaseConfig`)

```rust
let config = MultiPhaseConfig::new()
    .with_beam_width(4)           // ビーム幅
    .with_max_steps(5000)         // 最大ステップ数
    .with_progress(true)          // プログレス表示
    .with_collect_logs(true)      // ログ収集
    .with_early_termination_threshold(Some(10))  // 早期終了
    .with_subgraph_mode(SubgraphMode::Inline)    // サブグラフ処理モード
    .with_simd_widths(vec![4, 8]); // SIMD幅候補
```

**注意**: これらの関数は後方互換性のため `harp::backend` からも re-export されている。

## BeamSearchGraphOptimizer

- `beam_width`: ビーム幅（デフォルト: 10）
- `max_steps`: 最大探索ステップ（デフォルト: 10000）
- `early_termination_threshold`: 早期終了の閾値（改善なしステップ数）
  - `Some(n)`: n回連続で改善がなければ終了（デフォルト: `Some(10)`）
  - `None`: 早期終了を無効化
- `selector`: 候補選択器（デフォルト: `StaticCostSelector`）

詳細は`src/opt/graph/`を参照。

## 履歴・パス追跡

`OptimizationHistory`は最適化の各ステップを記録し、可視化や分析に使用する。

### OptimizationSnapshot

各ステップの状態を記録：
- `graph`: その時点でのグラフ
- `cost`: コスト推定値
- `suggester_name`: 適用されたSuggesterの名前
- `path`: このグラフに至るまでの完全なパス（各ステップの`(suggester_name, description)`）
- `alternatives`: 選択されなかった代替候補

### パス追跡

ビームサーチでは各ステップの最良候補が最終結果に至るとは限らない。`BeamEntry`構造体を使用してパスを追跡し、最終結果に至った実際のパスを`set_final_path()`で記録する。

```rust
// 最終結果に至るパスを取得
let final_path = history.final_path();
for (suggester_name, description) in final_path {
    println!("{}: {}", suggester_name, description);
}
```

### フェーズ結合

複数の最適化フェーズの履歴を結合する場合、`extend_with_phase()`を使用：

```rust
let mut combined = OptimizationHistory::new();
combined.extend_with_phase(phase1_history, "Lowering");
combined.extend_with_phase(phase2_history, "Absorption");
```

詳細は`src/opt/graph/history.rs`を参照。

## Selector

候補選択処理を抽象化するトレイト。ビームサーチにおいて、コスト付き候補から上位n件を選択する処理をカスタマイズ可能にする。

### 設計意図

tinygradのような多段階評価を実現するための抽象化：
1. 静的評価で明らかに悪い候補を足切り
2. 中間的なヒューリスティクスで絞り込み
3. 実行時間の実測値で精密に評価

### 実装

| 選択器 | 説明 |
|--------|------|
| StaticCostSelector | コストでソートして上位n件を選択（デフォルト） |
| MultiStageSelector | メソッドチェーンでn段階の選択パイプラインを構築 |
| GraphRuntimeSelector | グラフ用の2段階選択（静的コスト足切り→実測値） |

GraphRuntimeSelectorは以下の2段階で候補を選択:
1. 静的コスト（SimpleCostEstimator）で`pre_filter_count`件に足切り
2. GraphRuntimeCostEstimatorで実行時間を計測し、最終選択

### 使用例

```rust
// デフォルト（静的コスト選択、内部でSimpleCostEstimatorを使用）
let optimizer = BeamSearchGraphOptimizer::new(suggester);

// 多段階選択（dagoptスタイル）
let selector = MultiStageSelector::new()
    .then(|c| static_cost(c), 1000)   // 静的コストで1000件に足切り
    .then(|c| memory_cost(c), 100)    // メモリコストで100件に絞り込み
    .then(|c| measure_runtime(c), 10); // 実測で10件を最終選択

let optimizer = BeamSearchGraphOptimizer::new(suggester)
    .with_selector(selector);

// GraphRuntimeSelectorを使用した実測値ベース選択
let graph_selector = GraphRuntimeSelector::new(
    renderer,
    compiler,
    |sig| create_buffers(sig),
)
.with_pre_filter_count(5)   // 静的コストで5件に足切り
.with_measurement_count(5); // 5回計測して平均

let optimizer = BeamSearchGraphOptimizer::new(suggester)
    .with_selector(graph_selector);
```

詳細は`src/opt/selector.rs`を参照。設計は[dagopt](https://github.com/uthree/dagopt)を参考にしている。

## Pipelineでの統合

Pipelineは`set_runtime_buffer_factory()`を呼び出すことで、
グラフ最適化とAST最適化の両方で自動的にRuntimeSelectorを使用する。

### 型制約

```rust
pub struct Pipeline<R, Ctx, Comp>
where
    R: KernelSourceRenderer + Clone,
    Ctx: Context,
    Comp: Compiler<Context = Ctx>,
```

RuntimeSelector使用時にRendererとCompilerのクローンが必要なため、`Clone + 'static`制約が必要。

```rust
use harp::backend::{Pipeline, Context, Compiler};
use harp::backend::opencl::{OpenCLRenderer, OpenCLCompiler, OpenCLContext, OpenCLBuffer};

let context = OpenCLContext::new()?;
let mut pipeline = Pipeline::new(OpenCLRenderer::new(), OpenCLCompiler::new(), context);

// RuntimeSelectorを有効化（グラフとAST両方に適用）
pipeline.set_runtime_buffer_factory(|sig| {
    sig.inputs.iter()
        .chain(sig.outputs.iter())
        .map(|buf_sig| OpenCLBuffer::new(/* ... */))
        .collect()
});

// 通常のAPIで自動的にRuntimeSelectorが使用される
let (program, histories) = pipeline.optimize_graph_with_all_histories(graph)?;
```

### パラメータ

| 設定 | フィールド | デフォルト | 説明 |
|------|-----------|-----------|------|
| グラフ最適化 | `graph_config.pre_filter_count` | 10 | 静的コストでの足切り候補数 |
| グラフ最適化 | `graph_config.measurement_count` | 30 | 実行時間計測の回数 |
| AST最適化 | `ast_config.pre_filter_count` | 10 | 静的コストでの足切り候補数 |
| AST最適化 | `ast_config.measurement_count` | 30 | 実行時間計測の回数 |

### 注意点

- キャッシュなしの設計のため、毎回再計測される
- グラフ最適化ではLoweringを含むため、AST最適化より計測コストが高い
- `runtime_buffer_factory`が未設定の場合は静的コストにフォールバック

## 可視化

Graph ViewerはGraph最適化の各ステップを可視化する機能を提供する。

### 機能

- ステップナビゲーション（前へ/次へ、矢印キー操作）
- グラフのノードグラフ表示（egui_snarl使用）
- コスト遷移グラフ
- **パス表示**: 各スナップショットに至るまでのSuggester適用パスを表示（直近3件 + 省略記号）
- **候補セレクタ**: ビーム内の全候補を閲覧可能
  - 選択された候補（rank 0）とその他の候補を切り替え表示
  - 各候補のコスト、Suggester名、description（変換内容の説明）を表示
  - ↑/↓キーで候補を切り替え

### 使用方法

```rust
use harp_viz::HarpVizApp;

let mut app = HarpVizApp::new();
app.load_from_pipeline(&pipeline);  // Graph履歴も自動で読み込まれる
```

詳細は`crates/viz/src/graph_viewer.rs`を参照。

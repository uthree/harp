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
- **LoweringSuggester**: GraphOpをKernel(Function/Kernel)に変換
  - 各演算に対して複数の並列化戦略（`ParallelizationStrategy`）で候補を生成
  - ビームサーチのコスト評価により最適な戦略が選択される
  - `sequential_only()`モードでSequential戦略のみに制限可能
  - パラメータで候補数をチューニング可能
- **BufferAbsorptionSuggester**: KernelのsrcにあるBufferを`input_buffers`に取り込む

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

多次元グリッドへの変換は、AST最適化フェーズの`ThreadPartitionSuggester`が担当する。

#### LoweringSuggesterの設定

デフォルト値:
- **thread_group_sizes**: [64, 128, 256, 512]
- **vector_widths**: [2, 4, 8]（float2/float4/float8に相当）

ベクトル化は総要素数がベクトル幅で割り切れる場合のみ候補に追加される。

FlatParallel戦略は境界チェック（`if (tid < total_elements)`）を含むため、総要素数がスレッドグループサイズで割り切れなくても適用可能。グリッドサイズは自動的にスレッドグループサイズの倍数に切り上げられる。GPUのSIMTアーキテクチャでは境界チェックのオーバーヘッドは最後のグループのみに影響し、パフォーマンスへの影響は最小限。

```rust
// デフォルト設定
let suggester = LoweringSuggester::new();

// カスタム設定
let suggester = LoweringSuggester::new()
    .with_thread_group_sizes(vec![128, 256])  // サイズを制限
    .with_vector_widths(vec![4]);             // float4のみ

// ベクトル化を無効化
let suggester = LoweringSuggester::new()
    .without_vectorization();

// 高速化のためSequentialのみ
let suggester = LoweringSuggester::sequential_only();
```

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
LoweringSuggester      : GraphOp → Kernel(Function)
       ↓
BufferAbsorptionSuggester : 入力Bufferの取り込み
       ↓
ProgramRootAbsorptionSuggester : Kernel(Function) → ProgramRoot(Program)
       ↓
ProgramRootBufferAbsorptionSuggester : 入力Bufferの除去
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

## BeamSearchGraphOptimizer

- `beam_width`: ビーム幅（デフォルト: 10）
- `max_steps`: 最大探索ステップ（デフォルト: 10000）
- `early_termination_threshold`: 早期終了の閾値（改善なしステップ数）
  - `Some(n)`: n回連続で改善がなければ終了（デフォルト: `Some(10)`）
  - `None`: 早期終了を無効化
- `selector`: 候補選択器（デフォルト: `StaticCostSelector`）

詳細は`src/opt/graph/`を参照。

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

## GenericPipelineでの統合

GenericPipelineは`set_runtime_buffer_factory()`を呼び出すことで、
グラフ最適化とAST最適化の両方で自動的にRuntimeSelectorを使用する。

### 型制約

```rust
pub struct GenericPipeline<R, C>
where
    R: Renderer + Clone + 'static,
    C: Compiler<CodeRepr = R::CodeRepr> + Clone + 'static,
    C::Buffer: 'static,
```

RuntimeSelector使用時にRendererとCompilerのクローンが必要なため、`Clone + 'static`制約が必要。

```rust
use harp::backend::{GenericPipeline, OptimizationConfig};
use harp::backend::opencl::{OpenCLRenderer, OpenCLCompiler, OpenCLBuffer};

let mut pipeline = GenericPipeline::new(OpenCLRenderer::new(), OpenCLCompiler::new());

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

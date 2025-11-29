# Graph最適化

## トレイト定義

### GraphCostEstimator
Graphの実行コストを推定。`estimate(&self, graph: &Graph) -> f64`

### GraphOptimizer
Graphを最適化。`optimize(&self, graph: Graph) -> Graph`

### GraphSuggester
書き換え候補を提案（ビームサーチ用）。

**メソッド:**
- `name(&self) -> &'static str` - Suggesterの名前を返す
- `suggest(&self, graph: &Graph) -> Vec<Graph>` - 候補グラフを提案
- `suggest_named(&self, graph: &Graph) -> Vec<SuggestResult>` - Suggester名付きで候補を提案（デフォルト実装あり）

**SuggestResult構造体:**
- `graph: Graph` - 提案されたグラフ
- `suggester_name: String` - 提案したSuggesterの名前

## コスト推定器

### SimpleCostEstimator
ノード数とメモリアクセスベースの簡易推定器。
- ノード数ペナルティ項で無限のノード増加を防止
- `with_node_count_penalty(penalty: f32)`でペナルティ係数を調整可能（デフォルト: 0.01）

### AstBasedCostEstimator（推奨）
- GraphをASTに変換してコストを推定
- より正確だが計算コストが高い
- `ast_optimizer`で使用するAST最適化を設定可能

### KernelMergeCostEstimator
カーネルマージ最適化専用の推定器。
- Custom(Program)が存在する場合は低コスト
- Custom(Function)の数が多いほど高コスト
- 2段階最適化のPhase 2で使用

## Optimizer実装

### BeamSearchGraphOptimizer
- ビームサーチで最適な変換列を探索
- パラメータ:
  - `beam_width` - ビーム幅（デフォルト: 5）
  - `max_depth` - 最大探索深さ（デフォルト: 10）
  - `suggester` - 使用するSuggester
  - `estimator` - 使用するCostEstimator
- `optimize_with_history()`で最適化履歴を取得可能
- 各ステップで最良候補を提案したSuggester名を記録

### OptimizationSnapshot
最適化の各ステップのスナップショット。

**主要フィールド:**
- `step: usize` - ステップ番号
- `graph: Graph` - この時点でのグラフ
- `cost: f32` - コスト推定値
- `description: String` - ステップの説明
- `suggester_name: Option<String>` - このグラフを提案したSuggesterの名前
- `num_candidates: Option<usize>` - 候補数

## Suggester実装

### FusionSuggester
演算の融合を提案。

**融合パターン:**
- 連続するelementwise演算 → FusedElementwise
- elementwise → reduce → FusedElementwiseReduce

### CustomFusionSuggester
連続するElementwise演算をGraphOp::Customに融合。さらにElementwise→Reduce、Elementwise→Cumulativeパターンも融合。

**融合パターン:**
- Elementwiseチェーン → Custom(Elementwise)
- Elementwise → Reduce → Custom(Reduce)
- Elementwise → Cumulative → Custom(Cumulative)
- Custom(Elementwise) → Reduce → Custom(Reduce)
- Custom(Elementwise) → Cumulative → Custom(Cumulative)

**融合条件:**
- 現在のノードがElementwise/FusedElementwise/Custom(Elementwise)のいずれか
- 入力の少なくとも1つがElementwise/FusedElementwise/Custom(Elementwise)
- 融合対象のノードの被参照数が1（複数回参照されるノードは融合しない）

**効果:**
- 中間バッファの削減
- カーネル呼び出し回数の削減
- 任意のAST式による柔軟な演算表現
- Reduce/Cumulative演算前のElementwise演算を融合してカーネル効率を向上

### ViewInsertionSuggester
- メモリレイアウト最適化のためのView操作挿入
- 連続するelementwise演算間にpermute/flipを挿入

### ContiguousInsertionSuggester
- 非contiguousなViewを持つノードの前にContiguousノードを挿入
- View操作（permute、flipなど）で非連続になったメモリレイアウトを実体化
- メモリコピーのコストと、非連続アクセスのコストのトレードオフを探索

**挿入条件:**
- 入力ノードが非contiguousなView（転置、反転など）を持つ
- ノード自体がViewまたはContiguousノードでない

### LoweringSuggester
GraphOpをCustomノード（`AstNode::Function`を保持）に変換する。グラフ最適化の必須コンポーネント。

**変換対象:**
- Elementwise → Custom
- Reduce → Custom
- Cumulative → Custom
- Contiguous → Custom
- FusedElementwise → Custom
- FusedElementwiseReduce → Custom
- FusedElementwiseCumulative → Custom
- Pad/Slice/Concat/Arange/Cast/RandInit → Custom

**変換条件:**
- 入力のViewが連続（contiguous）であること（Elementwiseの場合）
- 既にCustomノードでないこと

### KernelMergeSuggester
複数のCustomノード（Function/Program）を1つのCustomノード（Program）にマージする。

**サポートするマージパターン:**
- Custom(Function) × N → Custom(Program)
- Custom(Program) + Custom(Function) → Custom(Program)（増分マージ）
- Custom(Program) + Custom(Program) → Custom(Program)（Program融合）

**効果:**
- グラフ全体を1つのProgram ASTとして表現
- 中間バッファの管理を明示的に制御
- カーネル呼び出し間にバリアを自動挿入（メモリ同期の保証）
- Lowererはカーネル関数を`AstNode::Kernel`として生成
- LowererはCustom(Program)を検出した場合、直接返す（パススルー）

**マージ条件:**
- Custom(Function)が2つ以上、または
- Custom(Program)が1つ以上かつCustom(Function)が1つ以上、または
- Custom(Program)が2つ以上

**生成するProgram構造:**
- 各カーネル関数（kernel_0, kernel_1, ...）
- main関数（harp_main）
  - 中間バッファの確保（Allocate）
  - カーネル呼び出し + バリア挿入
  - 中間バッファの解放（Deallocate）

**バリア挿入:**
カーネル呼び出し間に`AstNode::Barrier`を挿入し、前のカーネルの書き込み完了を保証する。

### CompositeSuggester
複数のSuggesterを組み合わせる。

- `suggest_named()`をオーバーライドし、各内部Suggesterの名前を正確に追跡
- 各Suggesterが提案したグラフには、そのSuggesterの名前が関連付けられる

## 最適化アーキテクチャ

### 単一ステージ最適化（推奨）

KernelMergeSuggesterがCustom(Program)の増分マージをサポートするため、
単一のビームサーチでloweringからマージまで統合的に最適化できる。

**メリット:**
- lowering途中の状態でも増分マージが可能
- 探索空間がより柔軟
- コスト関数が一貫

**API:**
```rust
// SuggesterFlags::single_stage()でKernelMergeSuggesterを含む
let (graph, history) = optimize_graph_single_stage(
    graph,
    SimpleCostEstimator::new(),
    8,     // beam_width
    200,   // max_steps
    true,  // show_progress
);
```

### 2段階最適化（従来方式、非推奨）

従来は2段階に分けて実行していた:

1. **Phase 1**: 一般的なグラフ最適化（fusion, lowering等）
2. **Phase 2**: カーネルマージ（複数Custom(Function)→Custom(Program)）

`backend::pipeline::optimize_graph_two_phase()`は非推奨。
`optimize_graph_single_stage()`または`SuggesterFlags::single_stage()`を使用推奨。

## 使用例

```rust
use harp::opt::graph::*;

// Suggesterの組み合わせ（推奨順序）
let suggester = CompositeSuggester::new(vec![
    // 1. 最適化系
    Box::new(CustomFusionSuggester::new()),
    Box::new(ContiguousInsertionSuggester::new()),
    // 2. GraphOp → Custom(Function)への変換
    Box::new(LoweringSuggester::new()),
    // 3. 複数Custom → Custom(Program)へのマージ
    Box::new(KernelMergeSuggester::new()),
]);

// 最適化器の構築
let estimator = SimpleCostEstimator::new();
let optimizer = BeamSearchGraphOptimizer::new(suggester, estimator)
    .with_beam_width(10)
    .with_max_steps(100);

// 最適化実行
let optimized_graph = optimizer.optimize(graph);
```

詳細は`src/opt/graph/`以下のソースコードを参照。

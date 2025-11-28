# Graph最適化

## トレイト定義

### GraphCostEstimator
Graphの実行コストを推定。`estimate(&self, graph: &Graph) -> f64`

### GraphOptimizer
Graphを最適化。`optimize(&self, graph: Graph) -> Graph`

### GraphSuggester
書き換え候補を提案（ビームサーチ用）。`suggest(&self, graph: &Graph) -> Vec<Graph>`

## コスト推定器

### SimpleCostEstimator
ノード数とメモリアクセスベースの簡易推定器。
- ノード数ペナルティ項で無限のノード増加を防止
- `with_node_count_penalty(penalty: f32)`でペナルティ係数を調整可能（デフォルト: 0.01）

### AstBasedCostEstimator（推奨）
- GraphをASTに変換してコストを推定
- より正確だが計算コストが高い
- `ast_optimizer`で使用するAST最適化を設定可能

## Optimizer実装

### BeamSearchGraphOptimizer
- ビームサーチで最適な変換列を探索
- パラメータ:
  - `beam_width` - ビーム幅（デフォルト: 5）
  - `max_depth` - 最大探索深さ（デフォルト: 10）
  - `suggester` - 使用するSuggester
  - `estimator` - 使用するCostEstimator

## Suggester実装

### FusionSuggester
演算の融合を提案。

**融合パターン:**
- 連続するelementwise演算 → FusedElementwise
- elementwise → reduce → FusedElementwiseReduce

### CustomFusionSuggester
連続するElementwise演算をGraphOp::Customに融合。

**融合条件:**
- 現在のノードがElementwise/FusedElementwise/Customのいずれか
- 入力の少なくとも1つがElementwise/FusedElementwise/Custom
- 融合対象のノードの被参照数が1（複数回参照されるノードは融合しない）

**効果:**
- 中間バッファの削減
- カーネル呼び出し回数の削減
- 任意のAST式による柔軟な演算表現

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

### ParallelStrategyChanger
並列化戦略を変更した候補を生成。

**戦略:**
- Sequential（逐次実行）
- Thread（スレッド並列）
- ThreadGroup（スレッドグループ並列、GPU向け）

**バックエンド実装状況:**
- C backend: 並列化未対応（Sequential戦略のみ）
- OpenCL backend: Thread/ThreadGroup戦略のコード生成は未実装（TODO）
- Metal backend: Thread/ThreadGroup戦略をサポート

### CompositeSuggester
複数のSuggesterを組み合わせる。

## 使用例

```rust
use harp::opt::graph::*;

// Suggesterの組み合わせ
let suggester = CompositeSuggester::new(vec![
    Box::new(FusionSuggester::new()),
    Box::new(CustomFusionSuggester::new()),
    Box::new(ContiguousInsertionSuggester::new()),
    Box::new(ParallelStrategyChanger::new()),
]);

// 最適化器の構築
let optimizer = BeamSearchGraphOptimizer::builder()
    .beam_width(10)
    .max_depth(15)
    .suggester(suggester)
    .estimator(AstBasedCostEstimator::new())
    .build();

// 最適化実行
let optimized_graph = optimizer.optimize(graph);
```

詳細は`src/opt/graph/`以下のソースコードを参照。

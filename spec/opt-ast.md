# AST最適化

## 2段階最適化

1. **ルールベース最適化**: 確定的な代数的簡約・定数畳み込み（最大100回反復）
2. **ビームサーチ最適化**: ループ変換などの構造変換を探索的に適用

## トレイト

- **CostEstimator**: ASTの実行コスト推定
- **Optimizer**: ASTの最適化
- **Suggester**: 書き換え候補を提案（ビームサーチ用）

## Optimizer実装

| Optimizer | 説明 |
|-----------|------|
| RuleBaseOptimizer | パターンマッチングベースの書き換え |
| BeamSearchOptimizer | ビームサーチで最適な変換列を探索 |

## Suggester実装

| Suggester | 説明 |
|-----------|------|
| RuleBaseSuggester | 書き換えルールを1ステップ適用 |
| LoopTilingSuggester | ループタイル化 |
| LoopInliningSuggester | 小さいループをインライン展開 |
| LoopInterchangeSuggester | ループ順序の入れ替え |
| LoopFusionSuggester | 同一境界のループをマージ |
| FunctionInliningSuggester | 小さい関数をインライン展開 |
| CseSuggester | 共通部分式除去 |
| VariableExpansionSuggester | 変数展開（CSEの逆操作） |
| CompositeSuggester | 複数Suggesterを組み合わせ |

## 代数的書き換えルール

- **単位元・零元**: `x + 0 → x`, `x * 1 → x`, `x * 0 → 0`
- **逆演算**: `recip(recip(x)) → x`, `log2(exp2(x)) → x`
- **定数畳み込み**: コンパイル時に計算可能な式を事前計算
- **ビット演算**: `x * 2^n → x << n`

ルール集: `constant_folding_rules()`, `simplification_rules()`, `normalization_rules()`, `all_algebraic_rules()`

## コスト推定

### SimpleCostEstimator

静的コスト推定。対数スケール（log(CPUサイクル数)）で計算：
- 各演算にCPUサイクル数を割り当て
- **ループ融合ボーナス**: 境界が揃ったループに減点
- **ノード数ペナルティ**: ノード爆発を抑制

### RuntimeCostEstimator

実行時間の実測値をコストとして使用する評価器。

- ASTをコンパイル・実行して実行時間（μs）を計測
- コンパイル結果と実行時間をキャッシュ
- ジェネリクスでRenderer/Compilerを保持（C, Metal, OpenCL対応）

**パラメータ:**
- `measurement_count`: 計測回数（デフォルト: 10回、平均を使用）

```rust
let estimator = RuntimeCostEstimator::new(
    renderer, compiler, signature,
    |sig| create_buffers(sig),  // バッファファクトリ（ユーザー定義）
).with_measurement_count(10);
```

## Selector（候補選択）

ビームサーチの候補選択を抽象化。

| Selector | 説明 |
|----------|------|
| StaticCostSelector | 静的コストでソート（デフォルト） |
| MultiStageSelector | 多段階選択（足切り→精密評価） |
| RuntimeSelector | 静的コスト足切り→実行時間計測 |

### RuntimeSelector

2段階評価を行う選択器：

1. **Stage 1**: SimpleCostEstimatorで`pre_filter_count`件に足切り
2. **Stage 2**: RuntimeCostEstimatorで実行時間計測、`n`件を選択

**パラメータ:**
- `pre_filter_count`: 足切り候補数（デフォルト: 10件）
- `measurement_count`: 計測回数（デフォルト: 10回）

```rust
let selector = RuntimeSelector::new(
    CRenderer::new(), CCompiler::new(), signature,
    |sig| create_buffers(sig),
)
.with_pre_filter_count(10)
.with_measurement_count(10);

let optimizer = BeamSearchOptimizer::new(suggester)
    .with_selector(selector);
```

## Barrierによる依存関係保証

ループ融合では、Barrierノードを同期点として扱い、Barrierを跨ぐループは融合しない。

```
for i { a[i] = ... }
for i { b[i] = ... }  // 融合OK（同じBarrier区間）

barrier

for i { c[i] = ... }  // 上のループとは融合しない
```

Kahnのアルゴリズムで世代間にBarrierが挿入されるため、同じ区間内のループは依存関係がないことが保証される。

詳細は`src/opt/ast/`を参照。

## 可視化

Code ViewerはAST最適化の各ステップを可視化する機能を提供する。

### 機能

- ステップナビゲーション（前へ/次へ、最初/最後にジャンプ、矢印キー操作）
- コスト・候補数遷移グラフ
- 適用されたルール名の表示
- 各ステップのデバッグログ
- コード差分表示（前のステップとの比較）
- シンタックスハイライト付きコード表示

### 使用方法

```rust
use harp_viz::HarpVizApp;

// 方法1: GenericPipelineから読み込む
let mut app = HarpVizApp::new();
app.load_from_pipeline(&pipeline);  // AST履歴も自動で読み込まれる

// 方法2: AST履歴を直接読み込む
let (optimized_ast, ast_history) = ast_optimizer.optimize_with_history(ast);
app.load_ast_optimization_history(ast_history);
```

詳細は`crates/viz/src/code_viewer.rs`を参照。

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
| FunctionMergeSuggester | 複数FunctionをProgram内で1つに統合 |
| FunctionInliningSuggester | 小さい関数をインライン展開 |
| CseSuggester | 共通部分式除去 |
| VariableExpansionSuggester | 変数展開（CSEの逆操作） |
| GroupParallelizationSuggester | ループ並列化（GroupId使用、動的分岐チェックあり） |
| LocalParallelizationSuggester | ループ並列化（LocalId使用、動的分岐チェックなし） |
| VectorizationSuggester | 連続メモリアクセスのSIMD化 |
| CompositeSuggester | 複数Suggesterを組み合わせ |

## 並列化Suggester

AST段階でRangeループをKernelに変換して並列化を行う。2つのSuggesterを提供：

### GroupParallelizationSuggester

GroupId（get_group_id）を使用してワークグループ間並列化を行う。
**動的分岐チェック: あり** - ループ内にIf文があると並列化しない。

### LocalParallelizationSuggester

LocalId（get_local_id）を使用してワークグループ内並列化を行う。
**動的分岐チェック: なし** - If文を含むループも並列化対象。

### 対応する変換

両Suggesterとも以下の2種類の変換に対応。
ホスト側でスレッド数・グループ数を正確に設定するため、**境界チェック（if文）は生成しない**。

**Function → Kernel変換:**

```
// 変換前
Function { body: Range { var: "i", start: 0, stop: N, body: ... } }

// 変換後（Group: GroupId使用、grid_size=N）
Kernel { params: [gidx0: GroupId(0), ...], grid_size: [N, 1, 1], body: ... }

// 変換後（Local: LocalId使用、thread_group_size=N）
Kernel { params: [lidx0: LocalId(0), ...], thread_group_size: [N, 1, 1], body: ... }
```

**Kernel内ループ追加並列化:**

```
// 変換前
Kernel { params: [gidx0: GroupId(0)], body: Range { var: "j", stop: M, ... } }

// 変換後（Group: 追加GroupId、grid_size[1]=M）
Kernel { params: [gidx0: GroupId(0), gidx1: GroupId(1)], grid_size: [.., M, ..], body: ... }

// 変換後（Local: 追加LocalId、thread_group_size[1]=M、GroupId(0)が軸0を使用しているため軸1を使用）
Kernel { params: [gidx0: GroupId(0), lidx1: LocalId(1)], thread_group_size: [.., M, ..], body: ... }
```

**並列化可否の判定:**
- ループ外変数への書き込みがないこと
- Store先オフセットがループ変数に依存していること
- GroupParallelizationSuggester: 動的分岐（If文）を含まないこと

**LoopInterchangeSuggesterとの組み合わせ:**
内側ループを並列化したい場合は、LoopInterchangeSuggesterで外側に持ってきてから並列化する。

## VectorizationSuggester

ループ展開後の連続メモリアクセスパターンを検出し、SIMD（ベクトル）命令に変換する。

### 処理フロー

1. Block内のStore文をグループ化（同一ポインタ、同一ベースオフセット）
2. 定数オフセットが連続（0, 1, 2, ...）かチェック
3. 式構造の同一性を検証
4. Load → load_vec、Const → broadcast に変換
5. Store → store_vec（count > 1）に変換

### 推奨パイプライン

```
LoopTilingSuggester (タイル化)
    ↓
LoopInliningSuggester (展開)
    ↓ 連続アクセスが露出
VectorizationSuggester (SIMD化)
    ↓
GroupParallelizationSuggester (並列化)
```

### 設計上の決定

**グラフLoweringではなくAST最適化で行う理由:**
- ループ展開後にのみ連続アクセスパターンが検出可能
- 全演算（四則演算、超越関数含む）を統一的に処理
- LoweringSuggesterはスカラー版のみ生成し、役割を明確化

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
- **候補セレクタ**: ビーム内の全候補を閲覧可能
  - 選択された候補（rank 0）とその他の候補を切り替え表示
  - 各候補のコスト、Suggester名、description（変換内容の説明）を表示
  - ↑/↓キーで候補を切り替え

### 使用方法

```rust
use harp_viz::HarpVizApp;

// 方法1: Pipelineから読み込む
let mut app = HarpVizApp::new();
app.load_from_pipeline(&pipeline);  // AST履歴も自動で読み込まれる

// 方法2: AST履歴を直接読み込む
let (optimized_ast, ast_history) = ast_optimizer.optimize_with_history(ast);
app.load_ast_optimization_history(ast_history);
```

詳細は`crates/viz/src/code_viewer.rs`を参照。

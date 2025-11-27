# AST最適化

## 概要

AST最適化はGenericPipelineにおいて2段階で実行される：

1. **ルールベース最適化**: 確定的な代数的簡約・定数畳み込みを先に適用（最大100回反復）
2. **ビームサーチ最適化**: ループ変換などのより複雑な構造変換を探索的に適用

この2段階構成により、簡単な最適化で正規化されたASTに対してビームサーチを実行することで、探索空間を削減し最適化品質を向上させる。

## トレイト定義

### CostEstimator
ASTの実行コストを推定。`estimate(&self, ast: &AstNode) -> f64`

### Optimizer
ASTを最適化。`optimize(&self, ast: AstNode) -> AstNode`

### Suggester
書き換え候補を提案（ビームサーチ用）。`suggest(&self, ast: &AstNode) -> Vec<AstNode>`

## 主な実装

### Optimizer実装

#### RuleBaseOptimizer
- パターンマッチングベースの書き換えルールを適用
- 変化がなくなるまで繰り返し適用（最大反復回数設定可能）

#### BeamSearchOptimizer
- ビームサーチで最適な変換列を探索
- Suggesterで候補生成、CostEstimatorで評価

### Suggester実装

#### RuleBaseSuggester
- 書き換えルールを1ステップ適用した候補を生成

#### LoopTilingSuggester
- ループをタイル化した候補を生成（複数のタイルサイズを試行）

#### LoopInliningSuggester
- 小さいループをインライン展開
- `innermost_only`オプション（デフォルト: true）で最内側ループのみを展開
- これによりネストループの外側が展開されてノード数が爆発する問題を防止

#### LoopInterchangeSuggester
- ネストしたループの順序を入れ替え

#### LoopFusionSuggester
- 連続するループで境界（start, step, stop）が同じ場合にボディをマージ
- ループオーバーヘッドを削減し、データ局所性を向上
- ループ変数名が異なる場合も自動的に置換

#### FunctionInliningSuggester
- 小さい関数をインライン展開
- Return文が1つだけの単純な関数が対象
- デフォルトで50ノードまでの関数を展開

#### CseSuggester（共通部分式除去）
- 共通する部分式を一時変数に抽出（Common Subexpression Elimination）
- 例: `y = (a * b) + (a * b)` → `cse_tmp_0 = a * b; y = cse_tmp_0 + cse_tmp_0`
- `min_expr_cost`で抽出対象の最小複雑度を設定（デフォルト: 2）
- `with_prefix`で生成する一時変数名のプレフィックスを指定（デフォルト: "cse_tmp_"）

#### VariableExpansionSuggester（変数展開）
- 一時変数をその定義式で置換（CSEの逆操作）
- 例: `x = a * b; y = x + x` → `y = (a * b) + (a * b)`
- `with_prefix`で展開対象の変数名プレフィックスを指定可能（例: "cse_tmp_"）
- `with_max_usage`で使用回数が指定値以下の変数のみ展開対象に
- 未使用変数の定義を削除する候補も生成

#### CompositeSuggester
- 複数のSuggesterを組み合わせる

## 代数的書き換えルール

### ルール分類

**単位元・零元ルール:**
- `x + 0 → x`、`x * 1 → x`、`x * 0 → 0`

**冪等則:**
- `max(x, x) → x`

**逆演算:**
- `recip(recip(x)) → x`、`log2(exp2(x)) → x`

**交換則・結合則・分配則:**
- 演算の順序や結合を最適化

**定数畳み込み:**
- コンパイル時に計算可能な式を事前計算

**その他の最適化:**
- `x * 2^n → x << n`（2のべき乗の乗算をシフトに）
- `sqrt(x * x) → x`

### ルール集生成関数

- `constant_folding_rules()` - 定数畳み込みルール
- `simplification_rules()` - 簡約化ルール（単位元除去など）
- `normalization_rules()` - 正規化ルール（交換則など）
- `all_algebraic_rules()` - すべてのルール

詳細は`src/opt/ast/rules.rs`を参照。

## ループ変換

### inline_small_loop
小さいループ（反復回数が定数）をインライン展開。

### tile_loop
ループをタイル化（ブロック化）してキャッシュ効率を向上。

詳細は`src/opt/ast/transforms.rs`を参照。

## コスト推定

SimpleCostEstimatorは対数スケール（log(CPUサイクル数)）でコストを計算。主な特性:

- 各演算にCPUサイクル数を割り当て（加算:3、乗算:4、除算:25、メモリアクセス:4等）
- ループコスト = ループ回数 × (ボディコスト + オーバーヘッド)
- **ループ融合ボーナス**: 連続するループの境界が揃っている場合、コストを減算（融合可能性への報酬）
- **ノード数ペナルティ**: ノード数に比例するペナルティを対数スケールで直接加算
  - `with_node_count_penalty(penalty: f32)`で係数を設定可能（デフォルト: 0.01）
  - ループ展開などでノード数が爆発する変換を抑制

このボーナスにより、LoopInterchangeSuggesterがループ境界を揃える変換を優先し、その後LoopFusionSuggesterが融合を適用する流れを促進。

## 依存関係の保証

### Barrierによる同期

ループ融合では、Barrierノードを同期点として扱い、Barrierを跨ぐループは融合しない。

Kahnのアルゴリズムによるトポロジカルソートで生成されたASTでは、世代間にBarrierが挿入される。これにより:
- 同じBarrier区間内の連続ループは依存関係がないことが保証
- Barrierを跨ぐ融合を禁止することで、データ依存関係の問題を回避

```
// 融合可能（同じBarrier区間内）
for i in 0..100 { a[i] = ... }
for i in 0..100 { b[i] = ... }  // 融合OK

barrier  // 同期点

for i in 0..100 { c[i] = ... }  // 上のループとは融合しない
```

この設計により、複雑なメモリアクセスパターンの解析なしに安全なループ融合が可能。

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

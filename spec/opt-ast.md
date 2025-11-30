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

SimpleCostEstimatorは対数スケール（log(CPUサイクル数)）で計算：
- 各演算にCPUサイクル数を割り当て
- **ループ融合ボーナス**: 境界が揃ったループに減点
- **ノード数ペナルティ**: ノード爆発を抑制

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

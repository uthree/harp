# 最適化
## グラフ最適化

## AST最適化
`trait AstOptimizer`によってASTを書き換える。

### ヒューリスティックな最適化
ビームサーチによって最適化したい

`trait AstOptSuggester`
複数の書き換え候補を提案する

`trait CostEstimator`
ASTの実行コストを推定する
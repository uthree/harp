# 最適化
## グラフ最適化

## AST最適化
`trait opt::ast::Optimizer`によってASTを書き換える。

### ヒューリスティックな最適化
ビームサーチによって最適化したい

`trait opt::ast::Suggester`
複数の書き換え候補を提案する

`trait opt::ast::CostEstimator`
ASTの実行コストを推定する
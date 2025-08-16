use crate::ast::AstNode;

// ASTの実行時のコストを評価する機能
// TODO: 動的Shape変数を正しく処理できるようにする
pub trait CostEstimator {
    fn cost(&self, ast: &AstNode) -> f32;
}

// ASTの書き換えの候補を提案する機能
pub trait Suggester {}

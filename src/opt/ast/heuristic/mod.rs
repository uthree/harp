pub mod handcode;
use crate::ast::AstNode;
pub mod beam_search;

// ASTの実行時のコストを評価する機能
// TODO: 動的Shape変数を正しく処理できるようにする
pub trait CostEstimator {
    fn estimate_cost(&self, ast: &AstNode) -> f32;
}

// ASTの書き換えの候補を提案する機能
pub trait RewriteSuggester {
    fn suggest(&self, ast: &AstNode) -> AstNode;
}

//! AST最適化器
//!
//! 各種探索アルゴリズムを用いたAST最適化器を提供します。
//!
//! - `RuleBaseOptimizer`: ルールベースの最適化器
//! - `BeamSearchOptimizer`: ビームサーチ最適化器
//! - `PrunedBfsOptimizer`: 枝刈り付き幅優先探索最適化器
//! - `PrunedDfsOptimizer`: 枝刈り付き深さ優先探索最適化器

mod beam_search;
mod pruned_bfs;
mod pruned_dfs;
mod rule_base;

pub use beam_search::BeamSearchOptimizer;
pub use pruned_bfs::PrunedBfsOptimizer;
pub use pruned_dfs::PrunedDfsOptimizer;
pub use rule_base::RuleBaseOptimizer;

use crate::ast::AstNode;

/// 最適化パス：(suggester_name, description)のリスト
pub(crate) type OptimizationPath = Vec<(String, String)>;

/// 候補情報: (suggester_name, description, path)
pub(crate) type CandidateInfo = (String, String, OptimizationPath);

/// ビームエントリ：ASTと、そのASTに至るまでの変換パス
#[derive(Clone, Debug)]
pub(crate) struct BeamEntry {
    pub ast: AstNode,
    /// このASTに至るまでの変換パス（各ステップでの(suggester_name, description)）
    pub path: OptimizationPath,
}

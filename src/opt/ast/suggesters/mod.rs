pub mod composite;
pub mod cse;
pub mod function_inlining;
pub mod group_parallelization;
pub mod loop_fusion;
pub mod loop_transforms;
pub mod parallelization_common;
pub mod rule_based;
pub mod thread_parallelization;
pub mod variable_expansion;

use crate::ast::AstNode;
use std::collections::HashSet;

// Re-export commonly used types
pub use composite::CompositeSuggester;
pub use cse::CseSuggester;
pub use function_inlining::FunctionInliningSuggester;
pub use group_parallelization::GroupParallelizationSuggester;
pub use loop_fusion::LoopFusionSuggester;
pub use loop_transforms::{LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester};
pub use rule_based::RuleBaseSuggester;
pub use thread_parallelization::ThreadParallelizationSuggester;
pub use variable_expansion::VariableExpansionSuggester;

/// 重複を排除しながら候補リストを作成するヘルパー関数
///
/// 各候補のDebug表現をキーとして重複を検出します。
pub fn deduplicate_candidates(candidates: Vec<AstNode>) -> Vec<AstNode> {
    let mut suggestions = Vec::new();
    let mut seen = HashSet::new();

    for candidate in candidates {
        let candidate_str = format!("{:?}", candidate);
        if !seen.contains(&candidate_str) {
            seen.insert(candidate_str);
            suggestions.push(candidate);
        }
    }

    suggestions
}

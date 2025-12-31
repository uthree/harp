pub mod composite;
pub mod cse;
pub mod function_inlining;
pub mod function_merge;
pub mod loop_fusion;
pub mod loop_transforms;
pub mod parallelization;
pub mod parallelization_common;
pub mod reduction_tiling;
pub mod rule_based;
pub mod variable_expansion;
pub mod vectorization;

use crate::ast::AstNode;
use std::collections::HashSet;

// Re-export commonly used types
pub use composite::CompositeSuggester;
pub use cse::CseSuggester;
pub use function_inlining::FunctionInliningSuggester;
pub use function_merge::FunctionMergeSuggester;
pub use loop_fusion::LoopFusionSuggester;
pub use loop_transforms::{LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester};
pub use parallelization::{GroupParallelizationSuggester, LocalParallelizationSuggester};
pub use reduction_tiling::ReductionTilingSuggester;
pub use rule_based::RuleBaseSuggester;
pub use variable_expansion::VariableExpansionSuggester;
pub use vectorization::VectorizationSuggester;

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

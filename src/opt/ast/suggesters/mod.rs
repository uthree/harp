pub mod composite;
pub mod function_inlining;
pub mod loop_fusion;
pub mod loop_transforms;
pub mod rule_based;

// Re-export commonly used types
pub use composite::CompositeSuggester;
pub use function_inlining::FunctionInliningSuggester;
pub use loop_fusion::LoopFusionSuggester;
pub use loop_transforms::{LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester};
pub use rule_based::RuleBaseSuggester;

pub mod composite;
pub mod loop_transforms;
pub mod rule_based;

// Re-export commonly used types
pub use composite::CompositeSuggester;
pub use loop_transforms::{LoopInliningSuggester, LoopInterchangeSuggester, LoopTilingSuggester};
pub use rule_based::RuleBaseSuggester;

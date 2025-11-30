pub mod ast_optimization;
pub mod composite;
pub mod contiguous;
pub mod fusion;
pub mod kernel_merge;
pub mod lowering;
pub mod tiling;
pub mod view;
pub mod view_merge;

// Re-export commonly used types
pub use ast_optimization::AstOptimizationSuggester;
pub use composite::CompositeSuggester;
pub use contiguous::ContiguousInsertionSuggester;
pub use fusion::FusionSuggester;
pub use kernel_merge::KernelMergeSuggester;
pub use lowering::LoweringSuggester;
pub use tiling::TilingSuggester;
pub use view::ViewInsertionSuggester;
pub use view_merge::ViewMergeSuggester;

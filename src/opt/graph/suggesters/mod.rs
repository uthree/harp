pub mod composite;
pub mod const_propagation;
pub mod contiguous;
pub mod custom_fusion;
pub mod fusion;
pub mod parallel;
pub mod simd;
pub mod tiling;
pub mod view;
pub mod view_merge;

// Re-export commonly used types
pub use composite::CompositeSuggester;
pub use const_propagation::ConstPropagationSuggester;
pub use contiguous::ContiguousInsertionSuggester;
pub use custom_fusion::CustomFusionSuggester;
pub use fusion::FusionSuggester;
pub use parallel::ParallelStrategyChanger;
pub use simd::SimdSuggester;
pub use tiling::TilingSuggester;
pub use view::ViewInsertionSuggester;
pub use view_merge::ViewMergeSuggester;

pub mod buffer_absorption;
pub mod composite;
pub mod contiguous;
pub mod fusion;
pub mod kernel_merge;
pub mod kernel_partition;
pub mod lowering;
pub mod padding_slice;
pub mod subgraph_inlining;
pub mod view;
pub mod view_merge;

// Re-export commonly used types
pub use buffer_absorption::BufferAbsorptionSuggester;
pub use composite::CompositeSuggester;
pub use contiguous::ContiguousInsertionSuggester;
pub use fusion::FusionSuggester;
pub use kernel_merge::KernelMergeSuggester;
pub use kernel_partition::KernelPartitionSuggester;
pub use lowering::LoweringSuggester;
pub use padding_slice::PaddingSliceSuggester;
pub use subgraph_inlining::SubgraphInliningSuggester;
pub use view::ViewInsertionSuggester;
pub use view_merge::ViewMergeSuggester;

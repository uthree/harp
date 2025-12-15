pub mod buffer_absorption;
pub mod composite;
pub mod contiguous;
pub mod fusion;
pub mod kernel_merge;
pub mod kernel_partition;
pub mod lowering;
pub mod program_root_absorption;
pub mod program_root_buffer_absorption;
pub mod subgraph_inlining;
pub mod tiling;
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
pub use program_root_absorption::ProgramRootAbsorptionSuggester;
pub use program_root_buffer_absorption::ProgramRootBufferAbsorptionSuggester;
pub use subgraph_inlining::SubgraphInliningSuggester;
pub use tiling::TilingSuggester;
pub use view::ViewInsertionSuggester;
pub use view_merge::ViewMergeSuggester;

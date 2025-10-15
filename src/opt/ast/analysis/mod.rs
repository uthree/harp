mod index_analysis;
mod memory_access;
mod parallelizable;
mod variable_usage;

pub use index_analysis::{
    analyze_index_pattern, are_patterns_disjoint, are_patterns_disjoint_for_parallelization,
    IndexPattern,
};
pub use memory_access::{
    collect_memory_accesses, group_accesses_by_variable, has_read_access, has_write_access,
    AccessType, MemoryAccess,
};
pub use parallelizable::{is_loop_parallelizable, ConflictReason, ParallelizabilityResult};
pub use variable_usage::collect_used_variables;

use super::index_analysis::{analyze_index_pattern, are_patterns_disjoint_for_parallelization};
use super::memory_access::{collect_memory_accesses, group_accesses_by_variable, AccessType};
use crate::ast::AstNode;

/// Result of parallelizability analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParallelizabilityResult {
    /// Loop is safe to parallelize - no memory conflicts
    Safe,
    /// Loop cannot be parallelized due to conflicts
    Unsafe(ConflictReason),
}

/// Reason why a loop cannot be parallelized
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConflictReason {
    /// Write-Write conflict: multiple iterations might write to the same location
    WriteWriteConflict { variable: String },
    /// Read-Write conflict: an iteration might read what another iteration writes
    ReadWriteConflict { variable: String },
    /// Complex index pattern that cannot be analyzed
    ComplexIndexPattern { variable: String },
}

/// Check if a loop can be safely parallelized
///
/// This function analyzes the memory access patterns in a loop body to determine
/// if parallel execution would be safe. It checks for:
/// 1. Write-Write conflicts (two iterations writing to the same location)
/// 2. Read-Write conflicts (one iteration reading what another writes)
///
/// # Arguments
/// * `loop_body` - The body of the loop to analyze
/// * `loop_var` - The loop variable name (e.g., "i")
///
/// # Returns
/// `ParallelizabilityResult::Safe` if the loop can be parallelized,
/// otherwise `ParallelizabilityResult::Unsafe` with the reason
///
/// # Examples
/// ```
/// // Safe: output[i] = input[i] * 2
/// // Each iteration writes to a different location
///
/// // Unsafe: sum += input[i]
/// // All iterations write to the same location (sum)
///
/// // Unsafe: output[i] = output[i-1] + input[i]
/// // Iteration i reads what iteration i-1 writes
/// ```
pub fn is_loop_parallelizable(loop_body: &AstNode, loop_var: &str) -> ParallelizabilityResult {
    // Collect all memory accesses in the loop body
    let accesses = collect_memory_accesses(loop_body);

    // Group accesses by variable
    let grouped = group_accesses_by_variable(&accesses);

    // Check each variable for conflicts
    for (var_name, var_accesses) in &grouped {
        // Check for write accesses
        let writes: Vec<_> = var_accesses
            .iter()
            .filter(|acc| acc.access_type == AccessType::Write)
            .collect();

        let reads: Vec<_> = var_accesses
            .iter()
            .filter(|acc| acc.access_type == AccessType::Read)
            .collect();

        // Check Write-Write conflicts
        if writes.len() > 1 {
            // Multiple writes to the same variable - need to check if they access different locations
            for i in 0..writes.len() {
                for j in (i + 1)..writes.len() {
                    let pattern_i = analyze_index_pattern(&writes[i].index, loop_var);
                    let pattern_j = analyze_index_pattern(&writes[j].index, loop_var);

                    // If patterns are complex, we can't prove safety
                    if pattern_i == super::index_analysis::IndexPattern::Complex
                        || pattern_j == super::index_analysis::IndexPattern::Complex
                    {
                        return ParallelizabilityResult::Unsafe(
                            ConflictReason::ComplexIndexPattern {
                                variable: var_name.clone(),
                            },
                        );
                    }

                    // Check if patterns might conflict across iterations
                    if !are_patterns_disjoint_for_parallelization(&pattern_i, &pattern_j) {
                        return ParallelizabilityResult::Unsafe(
                            ConflictReason::WriteWriteConflict {
                                variable: var_name.clone(),
                            },
                        );
                    }
                }
            }
        }

        // Check Read-Write conflicts
        // For parallelization, we need to ensure that:
        // 1. No iteration reads a location that another iteration might write
        // 2. If the patterns are identical (e.g., output[i] write and output[i] read),
        //    that's OK because each iteration accesses its own location
        if !writes.is_empty() && !reads.is_empty() {
            for write in &writes {
                for read in &reads {
                    let write_pattern = analyze_index_pattern(&write.index, loop_var);
                    let read_pattern = analyze_index_pattern(&read.index, loop_var);

                    // Complex patterns - can't prove safety
                    if write_pattern == super::index_analysis::IndexPattern::Complex
                        || read_pattern == super::index_analysis::IndexPattern::Complex
                    {
                        return ParallelizabilityResult::Unsafe(
                            ConflictReason::ComplexIndexPattern {
                                variable: var_name.clone(),
                            },
                        );
                    }

                    // Check if patterns are safe for parallelization
                    // This checks if patterns can conflict across different iterations
                    if !are_patterns_disjoint_for_parallelization(&write_pattern, &read_pattern) {
                        // Patterns might conflict across iterations - unsafe
                        // For example: write to i, read from i-1
                        return ParallelizabilityResult::Unsafe(
                            ConflictReason::ReadWriteConflict {
                                variable: var_name.clone(),
                            },
                        );
                    }
                }
            }
        }

        // Also check if a write pattern is constant (all iterations write to same location)
        if writes.len() == 1 {
            let pattern = analyze_index_pattern(&writes[0].index, loop_var);
            if matches!(pattern, super::index_analysis::IndexPattern::Constant(_)) {
                // All iterations write to the same constant location - unsafe
                return ParallelizabilityResult::Unsafe(ConflictReason::WriteWriteConflict {
                    variable: var_name.clone(),
                });
            }
        }
    }

    // No conflicts found - safe to parallelize
    ParallelizabilityResult::Safe
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::ConstLiteral;

    #[test]
    fn test_simple_parallel_safe() {
        // output[i] = input[i] * 2
        let body = store(
            var("output"),
            var("i"),
            mul(
                load(var("input"), var("i")),
                const_val(ConstLiteral::F32(2.0)),
            ),
        );

        let result = is_loop_parallelizable(&body, "i");
        assert_eq!(result, ParallelizabilityResult::Safe);
    }

    #[test]
    fn test_constant_write_unsafe() {
        // sum = sum + input[i] (sum is at constant index 0)
        let body = store(
            var("sum"),
            const_val(ConstLiteral::Usize(0)),
            add(
                load(var("sum"), const_val(ConstLiteral::Usize(0))),
                load(var("input"), var("i")),
            ),
        );

        let result = is_loop_parallelizable(&body, "i");
        assert!(matches!(
            result,
            ParallelizabilityResult::Unsafe(ConflictReason::WriteWriteConflict { .. })
        ));
    }

    #[test]
    fn test_dependent_iteration_unsafe() {
        // output[i] = output[i-1] + input[i]
        // Iteration i reads what iteration i-1 writes
        let body = store(
            var("output"),
            var("i"),
            add(
                load(
                    var("output"),
                    add(var("i"), const_val(ConstLiteral::Isize(-1))),
                ),
                load(var("input"), var("i")),
            ),
        );

        let result = is_loop_parallelizable(&body, "i");
        eprintln!("test_dependent_iteration_unsafe result: {:?}", result);
        // This should be unsafe because output[i] and output[i-1] might conflict
        // (one iteration writes to output[i], another reads from output[i])
        assert!(matches!(
            result,
            ParallelizabilityResult::Unsafe(ConflictReason::ReadWriteConflict { .. })
        ));
    }

    #[test]
    fn test_disjoint_accesses_safe() {
        // a[i*2] = b[i*2 + 1]
        // Even indices of a, odd indices of b
        let body = store(
            var("a"),
            mul(var("i"), const_val(ConstLiteral::Usize(2))),
            load(
                var("b"),
                add(
                    mul(var("i"), const_val(ConstLiteral::Usize(2))),
                    const_val(ConstLiteral::Usize(1)),
                ),
            ),
        );

        let result = is_loop_parallelizable(&body, "i");
        assert_eq!(result, ParallelizabilityResult::Safe);
    }

    #[test]
    fn test_complex_index_unsafe() {
        // output[i + j] = output[i]
        // Write index depends on both i and j - too complex to analyze
        let body = store(
            var("output"),
            add(var("i"), var("j")),
            load(var("output"), var("i")),
        );

        let result = is_loop_parallelizable(&body, "i");
        eprintln!("test_complex_index_unsafe result: {:?}", result);
        assert!(matches!(
            result,
            ParallelizabilityResult::Unsafe(ConflictReason::ComplexIndexPattern { .. })
        ));
    }

    #[test]
    fn test_multiple_independent_arrays_safe() {
        // output[i] = a[i] + b[i] + c[i]
        // Multiple reads, one write, all with pattern i
        let body = store(
            var("output"),
            var("i"),
            add(
                add(load(var("a"), var("i")), load(var("b"), var("i"))),
                load(var("c"), var("i")),
            ),
        );

        let result = is_loop_parallelizable(&body, "i");
        assert_eq!(result, ParallelizabilityResult::Safe);
    }

    #[test]
    fn test_self_read_write_different_indices_safe() {
        // output[i*2] = output[i*2 + 1]
        // Reading and writing to same array but different indices
        let body = store(
            var("output"),
            mul(var("i"), const_val(ConstLiteral::Usize(2))),
            load(
                var("output"),
                add(
                    mul(var("i"), const_val(ConstLiteral::Usize(2))),
                    const_val(ConstLiteral::Usize(1)),
                ),
            ),
        );

        let result = is_loop_parallelizable(&body, "i");
        // This should be safe because i*2 and i*2+1 are always different
        assert_eq!(result, ParallelizabilityResult::Safe);
    }
}

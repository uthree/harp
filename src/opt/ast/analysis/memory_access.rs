use crate::ast::AstNode;
use std::collections::HashMap;

/// Type of memory access
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    Read,
    Write,
}

/// Information about a single memory access
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryAccess {
    /// Variable being accessed (e.g., "array", "buffer")
    pub variable: String,
    /// Index expression used to access the memory
    pub index: AstNode,
    /// Type of access (read or write)
    pub access_type: AccessType,
    /// Vector width (1 for scalar, >1 for vector operations)
    pub vector_width: usize,
}

/// Collect all memory accesses (Load and Store operations) from an AST subtree
///
/// This function recursively traverses the AST and collects information about
/// all memory load and store operations. This is useful for analyzing data
/// dependencies and memory access patterns for parallelization.
///
/// # Arguments
/// * `node` - The AST node to analyze
///
/// # Returns
/// A vector of `MemoryAccess` structures describing all memory operations
pub fn collect_memory_accesses(node: &AstNode) -> Vec<MemoryAccess> {
    let mut accesses = Vec::new();
    collect_memory_accesses_recursive(node, &mut accesses);
    accesses
}

fn collect_memory_accesses_recursive(node: &AstNode, accesses: &mut Vec<MemoryAccess>) {
    match node {
        // Load operation - this is a read access
        AstNode::Load {
            target,
            index,
            vector_width,
        } => {
            // Extract variable name from target (should be a Var node)
            if let AstNode::Var(var_name) = target.as_ref() {
                accesses.push(MemoryAccess {
                    variable: var_name.clone(),
                    index: *index.clone(),
                    access_type: AccessType::Read,
                    vector_width: *vector_width,
                });
            }
            // Also check the index expression for any nested accesses
            collect_memory_accesses_recursive(index, accesses);
        }

        // Store operation - this is a write access
        AstNode::Store {
            target,
            index,
            value,
            vector_width,
        } => {
            // Extract variable name from target
            if let AstNode::Var(var_name) = target.as_ref() {
                accesses.push(MemoryAccess {
                    variable: var_name.clone(),
                    index: *index.clone(),
                    access_type: AccessType::Write,
                    vector_width: *vector_width,
                });
            }
            // Check index and value for nested accesses
            collect_memory_accesses_recursive(index, accesses);
            collect_memory_accesses_recursive(value, accesses);
        }

        // For all other nodes, recursively process children
        _ => {
            for child in node.children() {
                collect_memory_accesses_recursive(child, accesses);
            }
        }
    }
}

/// Group memory accesses by variable name
///
/// This is useful for analyzing access patterns to specific arrays or buffers.
///
/// # Arguments
/// * `accesses` - Vector of memory accesses to group
///
/// # Returns
/// A map from variable name to all accesses to that variable
pub fn group_accesses_by_variable(accesses: &[MemoryAccess]) -> HashMap<String, Vec<MemoryAccess>> {
    let mut grouped: HashMap<String, Vec<MemoryAccess>> = HashMap::new();

    for access in accesses {
        grouped
            .entry(access.variable.clone())
            .or_default()
            .push(access.clone());
    }

    grouped
}

/// Check if there are any write accesses to a variable
pub fn has_write_access(accesses: &[MemoryAccess]) -> bool {
    accesses
        .iter()
        .any(|acc| acc.access_type == AccessType::Write)
}

/// Check if there are any read accesses to a variable
pub fn has_read_access(accesses: &[MemoryAccess]) -> bool {
    accesses
        .iter()
        .any(|acc| acc.access_type == AccessType::Read)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::ConstLiteral;

    #[test]
    fn test_simple_load() {
        // input[i]
        let expr = load(var("input"), var("i"));
        let accesses = collect_memory_accesses(&expr);

        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].variable, "input");
        assert_eq!(accesses[0].access_type, AccessType::Read);
        assert_eq!(accesses[0].vector_width, 1);
        assert_eq!(accesses[0].index, var("i"));
    }

    #[test]
    fn test_simple_store() {
        // output[i] = 1.0
        let stmt = store(var("output"), var("i"), const_val(ConstLiteral::F32(1.0)));
        let accesses = collect_memory_accesses(&stmt);

        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].variable, "output");
        assert_eq!(accesses[0].access_type, AccessType::Write);
        assert_eq!(accesses[0].vector_width, 1);
        assert_eq!(accesses[0].index, var("i"));
    }

    #[test]
    fn test_load_and_store() {
        // output[i] = input[i] * 2.0
        let stmt = store(
            var("output"),
            var("i"),
            mul(
                load(var("input"), var("i")),
                const_val(ConstLiteral::F32(2.0)),
            ),
        );
        let accesses = collect_memory_accesses(&stmt);

        assert_eq!(accesses.len(), 2);

        // First access should be the write to output
        assert_eq!(accesses[0].variable, "output");
        assert_eq!(accesses[0].access_type, AccessType::Write);
        assert_eq!(accesses[0].index, var("i"));

        // Second access should be the read from input
        assert_eq!(accesses[1].variable, "input");
        assert_eq!(accesses[1].access_type, AccessType::Read);
        assert_eq!(accesses[1].index, var("i"));
    }

    #[test]
    fn test_multiple_arrays() {
        // output[i] = a[i] + b[i]
        let stmt = store(
            var("output"),
            var("i"),
            add(load(var("a"), var("i")), load(var("b"), var("i"))),
        );
        let accesses = collect_memory_accesses(&stmt);

        assert_eq!(accesses.len(), 3);

        // Write to output
        assert_eq!(accesses[0].variable, "output");
        assert_eq!(accesses[0].access_type, AccessType::Write);

        // Read from a
        assert_eq!(accesses[1].variable, "a");
        assert_eq!(accesses[1].access_type, AccessType::Read);

        // Read from b
        assert_eq!(accesses[2].variable, "b");
        assert_eq!(accesses[2].access_type, AccessType::Read);
    }

    #[test]
    fn test_loop_with_accesses() {
        // for (i = 0; i < 10; i++) { output[i] = input[i]; }
        let loop_stmt = range(
            "i",
            const_val(ConstLiteral::Usize(10)),
            store(var("output"), var("i"), load(var("input"), var("i"))),
        );
        let accesses = collect_memory_accesses(&loop_stmt);

        assert_eq!(accesses.len(), 2);
        assert_eq!(accesses[0].variable, "output");
        assert_eq!(accesses[0].access_type, AccessType::Write);
        assert_eq!(accesses[1].variable, "input");
        assert_eq!(accesses[1].access_type, AccessType::Read);
    }

    #[test]
    fn test_group_by_variable() {
        // output[i] = a[i] + a[j]
        let stmt = store(
            var("output"),
            var("i"),
            add(load(var("a"), var("i")), load(var("a"), var("j"))),
        );
        let accesses = collect_memory_accesses(&stmt);
        let grouped = group_accesses_by_variable(&accesses);

        assert_eq!(grouped.len(), 2); // output and a
        assert_eq!(grouped["output"].len(), 1);
        assert_eq!(grouped["a"].len(), 2); // Two reads from a
    }

    #[test]
    fn test_has_write_access() {
        let write = MemoryAccess {
            variable: "output".to_string(),
            index: var("i"),
            access_type: AccessType::Write,
            vector_width: 1,
        };
        let read = MemoryAccess {
            variable: "input".to_string(),
            index: var("i"),
            access_type: AccessType::Read,
            vector_width: 1,
        };

        assert!(has_write_access(std::slice::from_ref(&write)));
        assert!(!has_write_access(std::slice::from_ref(&read)));
        assert!(has_write_access(&[read, write]));
    }

    #[test]
    fn test_has_read_access() {
        let write = MemoryAccess {
            variable: "output".to_string(),
            index: var("i"),
            access_type: AccessType::Write,
            vector_width: 1,
        };
        let read = MemoryAccess {
            variable: "input".to_string(),
            index: var("i"),
            access_type: AccessType::Read,
            vector_width: 1,
        };

        assert!(has_read_access(std::slice::from_ref(&read)));
        assert!(!has_read_access(std::slice::from_ref(&write)));
        assert!(has_read_access(&[read, write]));
    }

    #[test]
    fn test_complex_index_expression() {
        // output[i*2 + 1] = input[i]
        let stmt = store(
            var("output"),
            add(
                mul(var("i"), const_val(ConstLiteral::Usize(2))),
                const_val(ConstLiteral::Usize(1)),
            ),
            load(var("input"), var("i")),
        );
        let accesses = collect_memory_accesses(&stmt);

        assert_eq!(accesses.len(), 2);
        assert_eq!(accesses[0].variable, "output");
        // Complex index expression should be preserved
        assert_eq!(
            accesses[0].index,
            add(
                mul(var("i"), const_val(ConstLiteral::Usize(2))),
                const_val(ConstLiteral::Usize(1))
            )
        );
    }
}

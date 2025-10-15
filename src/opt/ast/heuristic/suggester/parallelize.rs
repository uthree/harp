use crate::ast::helper::*;
use crate::ast::{AstNode, ConstLiteral, ThreadIdDecl, ThreadIdType};
use crate::opt::ast::analysis::{is_loop_parallelizable, ParallelizabilityResult};
use crate::opt::ast::heuristic::RewriteSuggester;

/// Suggester that parallelizes loops in kernels using GPU thread IDs
///
/// This transformation replaces Range loops with GPU thread-indexed computation.
/// It only parallelizes loops that are proven safe by the parallelizability analysis.
///
/// # Example transformation
///
/// Before:
/// ```c
/// kernel void my_kernel(float* input, float* output, size_t n) {
///     for (size_t i = 0; i < n; i++) {
///         output[i] = input[i] * 2.0;
///     }
/// }
/// ```
///
/// After:
/// ```c
/// kernel void my_kernel(float* input, float* output, size_t n) {
///     size_t global_id[3] = get_global_id();
///     size_t i = global_id[0];
///     if (i < n) {
///         output[i] = input[i] * 2.0;
///     }
/// }
/// ```
pub struct ParallelizeSuggester {
    /// Name for global thread ID variable
    global_id_name: String,
}

impl ParallelizeSuggester {
    pub fn new() -> Self {
        Self {
            global_id_name: "global_id".to_string(),
        }
    }

    pub fn with_global_id_name(name: String) -> Self {
        Self {
            global_id_name: name,
        }
    }

    /// Check if a kernel has parallelizable loops
    fn has_parallelizable_loop(&self, kernel: &AstNode) -> bool {
        if let AstNode::Kernel { statements, .. } = kernel {
            statements
                .iter()
                .any(|stmt| self.is_parallelizable_loop(stmt))
        } else {
            false
        }
    }

    /// Check if a statement is a parallelizable loop
    #[allow(clippy::only_used_in_recursion)]
    fn is_parallelizable_loop(&self, node: &AstNode) -> bool {
        match node {
            AstNode::Range {
                counter_name, body, ..
            } => {
                // Check if the loop body is safe to parallelize
                matches!(
                    is_loop_parallelizable(body, counter_name),
                    ParallelizabilityResult::Safe
                )
            }
            AstNode::Block { statements, .. } => statements
                .iter()
                .any(|stmt| self.is_parallelizable_loop(stmt)),
            _ => false,
        }
    }

    /// Parallelize a kernel by replacing loops with thread-indexed computation
    fn parallelize_kernel(&self, kernel: &AstNode) -> Option<AstNode> {
        if let AstNode::Kernel {
            name,
            scope,
            statements,
            arguments,
            return_type,
            global_size: _,
            local_size: _,
        } = kernel
        {
            // Find the first parallelizable loop
            let mut new_statements = Vec::new();
            let mut parallelized = false;
            let mut loop_counter = None;
            let mut loop_max = None;
            let mut loop_body = None;

            for stmt in statements {
                if !parallelized {
                    if let AstNode::Range {
                        counter_name,
                        start,
                        max,
                        step,
                        body,
                        ..
                    } = stmt
                    {
                        // Check if this loop is parallelizable
                        if matches!(
                            is_loop_parallelizable(body, counter_name),
                            ParallelizabilityResult::Safe
                        ) {
                            // Check if loop starts at 0 and has step 1
                            let starts_at_zero = matches!(
                                **start,
                                AstNode::Const(ConstLiteral::Usize(0))
                                    | AstNode::Const(ConstLiteral::Isize(0))
                            );
                            let step_is_one = matches!(
                                **step,
                                AstNode::Const(ConstLiteral::Usize(1))
                                    | AstNode::Const(ConstLiteral::Isize(1))
                            );

                            if starts_at_zero && step_is_one {
                                // Save loop information for parallelization
                                loop_counter = Some(counter_name.clone());
                                loop_max = Some(max.clone());
                                loop_body = Some(body.clone());
                                parallelized = true;
                                continue;
                            }
                        }
                    }
                }
                new_statements.push(stmt.clone());
            }

            // If we found a parallelizable loop, transform it
            if let (Some(counter), Some(_max), Some(body)) = (loop_counter, loop_max, loop_body) {
                // Add thread ID declaration to scope
                let mut new_scope = scope.clone();

                // Check if we already have a global_id thread ID
                let tid_exists = new_scope
                    .thread_ids
                    .iter()
                    .any(|tid| tid.name == self.global_id_name);

                let dimension = if tid_exists {
                    // Find how many dimensions are already used for this thread ID
                    new_scope.declarations.iter().filter(|d| {
                        d.name.starts_with("gid") || d.name.starts_with("lid")
                    }).count()
                } else {
                    // First dimension
                    new_scope.thread_ids.push(ThreadIdDecl {
                        name: self.global_id_name.clone(),
                        id_type: ThreadIdType::GlobalId,
                    });
                    0
                };

                // Generate index variable name: gid0, gid1, gid2, ...
                let index_var_name = format!("gid{}", dimension);

                // Add variable declaration for the index
                new_scope.declarations.push(crate::ast::VariableDecl {
                    name: index_var_name.clone(),
                    dtype: crate::ast::DType::Usize,
                    constant: false,
                    size_expr: None,
                });

                // Create statements:
                // 1. size_t gid0 = global_id[0];
                let index_assign = assign(
                    &index_var_name,
                    AstNode::Load {
                        target: Box::new(var(&self.global_id_name)),
                        index: Box::new(const_val(ConstLiteral::Usize(dimension))),
                        vector_width: 1,
                    },
                );

                // 2. Replace loop counter with the new index variable in the body
                let updated_body = body.replace_node(&var(&counter), var(&index_var_name));

                // 3. if (gid0 < max) { body }
                let bounds_check = if_then(
                    AstNode::LessThan(Box::new(var(&index_var_name)), _max.clone()),
                    updated_body,
                );

                // Prepend index assignment and bounds check, then remaining statements
                let mut parallelized_statements = vec![index_assign, bounds_check];
                parallelized_statements.extend(new_statements);

                // Update global size to match loop bound
                // For now, use a default size - this should be determined from the loop bound
                let global_size = [
                    Box::new(const_val(ConstLiteral::Usize(1024))), // Will be updated based on actual loop bound
                    Box::new(const_val(ConstLiteral::Usize(1))),
                    Box::new(const_val(ConstLiteral::Usize(1))),
                ];

                let local_size = [
                    Box::new(const_val(ConstLiteral::Usize(256))),
                    Box::new(const_val(ConstLiteral::Usize(1))),
                    Box::new(const_val(ConstLiteral::Usize(1))),
                ];

                return Some(AstNode::Kernel {
                    name: name.clone(),
                    scope: new_scope,
                    statements: parallelized_statements,
                    arguments: arguments.clone(),
                    return_type: return_type.clone(),
                    global_size,
                    local_size,
                });
            }
        }

        None
    }
}

impl Default for ParallelizeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteSuggester for ParallelizeSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Only process Program nodes
        if let AstNode::Program {
            functions,
            entry_point,
        } = ast
        {
            let mut has_parallelization = false;
            let mut new_functions = Vec::new();

            for func in functions {
                // Only process Kernel nodes
                if let AstNode::Kernel { .. } = func {
                    if self.has_parallelizable_loop(func) {
                        if let Some(parallelized) = self.parallelize_kernel(func) {
                            new_functions.push(parallelized);
                            has_parallelization = true;
                        } else {
                            new_functions.push(func.clone());
                        }
                    } else {
                        new_functions.push(func.clone());
                    }
                } else {
                    new_functions.push(func.clone());
                }
            }

            if has_parallelization {
                suggestions.push(AstNode::program(new_functions, entry_point.clone()));
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, KernelScope, Scope};

    #[test]
    fn test_simple_loop_parallelization() {
        // Create a kernel with a simple parallelizable loop:
        // kernel void my_kernel(float* output, size_t n) {
        //     for (size_t i = 0; i < n; i++) {
        //         output[i] = 1.0;
        //     }
        // }
        let loop_stmt = range_builder(
            "i",
            var("n"),
            store(var("output"), var("i"), const_val(ConstLiteral::F32(1.0))),
        )
        .start(const_val(ConstLiteral::Usize(0)))
        .step(const_val(ConstLiteral::Usize(1)))
        .build();

        let kernel = AstNode::kernel(
            "my_kernel",
            vec![
                ("output".to_string(), DType::Ptr(Box::new(DType::F32))),
                ("n".to_string(), DType::Usize),
            ],
            DType::Void,
            KernelScope {
                declarations: vec![],
                thread_ids: vec![],
            },
            vec![loop_stmt],
            [
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
            ],
            [
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
            ],
        );

        let program = AstNode::program(vec![kernel], "main");

        let suggester = ParallelizeSuggester::new();
        let suggestions = suggester.suggest(&program);

        // Should have one suggestion
        assert_eq!(suggestions.len(), 1);

        // Check the parallelized kernel
        if let AstNode::Program { functions, .. } = &suggestions[0] {
            assert_eq!(functions.len(), 1);

            if let AstNode::Kernel {
                scope, statements, ..
            } = &functions[0]
            {
                // Should have thread ID in scope
                assert_eq!(scope.thread_ids.len(), 1);
                assert_eq!(scope.thread_ids[0].name, "global_id");
                assert_eq!(scope.thread_ids[0].id_type, ThreadIdType::GlobalId);

                // Should have 2 statements: assignment and if
                assert_eq!(statements.len(), 2);

                // First statement should be: gid0 = global_id[0]
                if let AstNode::Assign(name, _) = &statements[0] {
                    assert_eq!(name, "gid0");
                } else {
                    panic!("Expected Assign node, got {:?}", statements[0]);
                }

                // Should have gid0 variable declaration in scope
                assert!(scope
                    .declarations
                    .iter()
                    .any(|d| d.name == "gid0" && d.dtype == crate::ast::DType::Usize));

                // Second statement should be: if (i < n) { body }
                if let AstNode::If {
                    condition,
                    then_branch,
                    else_branch,
                } = &statements[1]
                {
                    // Check it's a LessThan comparison
                    assert!(matches!(**condition, AstNode::LessThan(_, _)));
                    // Check there's a then branch with the body
                    assert!(matches!(**then_branch, AstNode::Store { .. }));
                    // Check no else branch
                    assert!(else_branch.is_none());
                } else {
                    panic!("Expected If node, got {:?}", statements[1]);
                }
            } else {
                panic!("Expected Kernel node");
            }
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_non_parallelizable_loop_not_transformed() {
        // Create a kernel with a non-parallelizable loop (has dependencies):
        // kernel void my_kernel(float* output, size_t n) {
        //     for (size_t i = 0; i < n; i++) {
        //         output[i] = output[i-1] + 1.0; // Depends on previous iteration
        //     }
        // }
        let loop_stmt = range_builder(
            "i",
            var("n"),
            store(
                var("output"),
                var("i"),
                add(
                    load(
                        var("output"),
                        add(var("i"), const_val(ConstLiteral::Isize(-1))),
                    ),
                    const_val(ConstLiteral::F32(1.0)),
                ),
            ),
        )
        .start(const_val(ConstLiteral::Usize(0)))
        .step(const_val(ConstLiteral::Usize(1)))
        .build();

        let kernel = AstNode::kernel(
            "my_kernel",
            vec![
                ("output".to_string(), DType::Ptr(Box::new(DType::F32))),
                ("n".to_string(), DType::Usize),
            ],
            DType::Void,
            KernelScope {
                declarations: vec![],
                thread_ids: vec![],
            },
            vec![loop_stmt],
            [
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
            ],
            [
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
            ],
        );

        let program = AstNode::program(vec![kernel], "main");

        let suggester = ParallelizeSuggester::new();
        let suggestions = suggester.suggest(&program);

        // Should have no suggestions (loop is not parallelizable)
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_regular_function_not_transformed() {
        // Regular functions should not be transformed
        let func = function(
            "my_func",
            vec![("x".to_string(), DType::F32)],
            DType::Void,
            Scope {
                declarations: vec![],
            },
            vec![range_builder(
                "i",
                const_val(ConstLiteral::Usize(10)),
                assign("x", const_val(ConstLiteral::F32(1.0))),
            )
            .build()],
        );

        let program = AstNode::program(vec![func], "main");

        let suggester = ParallelizeSuggester::new();
        let suggestions = suggester.suggest(&program);

        // Should have no suggestions (not a kernel)
        assert_eq!(suggestions.len(), 0);
    }
}

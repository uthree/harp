use crate::ast::helper::const_val;
use crate::ast::{AstNode, ConstLiteral, DType, KernelScope};
use crate::opt::ast::heuristic::RewriteSuggester;

/// Suggester that converts functions into GPU kernels
///
/// This transformation converts regular functions that have been extracted
/// from loops into GPU kernel functions. The kernel functions can then be
/// executed on GPU devices with appropriate thread indexing.
///
/// # Example transformation
///
/// Before:
/// ```c
/// void loop_func_0(float* input, float* output, size_t loop_bound) {
///     for (size_t i = 0; i < loop_bound; i++) {
///         output[i] = input[i] * 2.0;
///     }
/// }
/// ```
///
/// After:
/// ```c
/// kernel void loop_func_0_kernel(float* input, float* output, size_t loop_bound) {
///     // The loop will be replaced with thread indexing by ParallelizeSuggester
///     for (size_t i = 0; i < loop_bound; i++) {
///         output[i] = input[i] * 2.0;
///     }
/// }
/// ```
pub struct KernelizeSuggester {
    /// Suffix to add to kernel names
    kernel_suffix: String,
}

impl KernelizeSuggester {
    pub fn new() -> Self {
        Self {
            kernel_suffix: "_kernel".to_string(),
        }
    }

    pub fn with_suffix(suffix: String) -> Self {
        Self {
            kernel_suffix: suffix,
        }
    }

    /// Check if a function is a candidate for kernelization
    ///
    /// A function is a candidate if:
    /// 1. It has a loop in its body
    /// 2. The loop is safe to parallelize (will be checked by the caller)
    fn is_kernelizable_function(&self, func: &AstNode) -> bool {
        if let AstNode::Function { statements, .. } = func {
            // Check if the function contains at least one Range node
            statements.iter().any(|stmt| self.contains_loop(stmt))
        } else {
            false
        }
    }

    /// Recursively check if a node contains a loop
    #[allow(clippy::only_used_in_recursion)]
    fn contains_loop(&self, node: &AstNode) -> bool {
        match node {
            AstNode::Range { .. } => true,
            AstNode::Block { statements, .. } => {
                statements.iter().any(|stmt| self.contains_loop(stmt))
            }
            _ => node
                .children()
                .iter()
                .any(|child| self.contains_loop(child)),
        }
    }

    /// Convert a function to a kernel
    fn kernelize_function(&self, func: &AstNode) -> Option<AstNode> {
        if let AstNode::Function {
            name,
            scope,
            statements,
            arguments,
            return_type,
        } = func
        {
            // Kernels must return void
            if *return_type != DType::Void {
                return None;
            }

            // Create kernel name
            let kernel_name = format!("{}{}", name, self.kernel_suffix);

            // Convert Scope to KernelScope (no thread IDs yet - will be added by ParallelizeSuggester)
            let kernel_scope = KernelScope {
                declarations: scope.declarations.clone(),
                thread_ids: vec![],
            };

            // Create default work sizes (1,1,1) - will be filled in by ParallelizeSuggester
            let default_size = [
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
                Box::new(const_val(ConstLiteral::Usize(1))),
            ];

            // Create kernel node
            Some(AstNode::Kernel {
                name: kernel_name,
                scope: kernel_scope,
                statements: statements.clone(),
                arguments: arguments.clone(),
                return_type: return_type.clone(),
                global_size: default_size.clone(),
                local_size: default_size,
            })
        } else {
            None
        }
    }
}

impl Default for KernelizeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteSuggester for KernelizeSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Only process Program nodes
        if let AstNode::Program {
            functions,
            entry_point,
        } = ast
        {
            // Try to kernelize each function
            let mut has_kernelization = false;
            let mut new_functions = Vec::new();

            for func in functions {
                if self.is_kernelizable_function(func) {
                    if let Some(kernel) = self.kernelize_function(func) {
                        // Add both the kernel and the original function
                        new_functions.push(kernel);
                        new_functions.push(func.clone());
                        has_kernelization = true;
                    } else {
                        new_functions.push(func.clone());
                    }
                } else {
                    new_functions.push(func.clone());
                }
            }

            // If we kernelized any functions, create a new program
            if has_kernelization {
                suggestions.push(AstNode::program(new_functions, entry_point.clone()));
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::Scope;

    #[test]
    fn test_simple_function_kernelization() {
        // Create a simple function with a loop:
        // void loop_func_0(float* output, size_t loop_bound) {
        //     for (size_t i = 0; i < loop_bound; i++) {
        //         output[i] = 1.0;
        //     }
        // }
        let loop_stmt = range_builder(
            "i",
            const_val(ConstLiteral::Usize(10)),
            store(var("output"), var("i"), const_val(ConstLiteral::F32(1.0))),
        )
        .start(const_val(ConstLiteral::Usize(0)))
        .step(const_val(ConstLiteral::Usize(1)))
        .build();

        let func = function(
            "loop_func_0",
            vec![
                ("output".to_string(), DType::Ptr(Box::new(DType::F32))),
                ("loop_bound".to_string(), DType::Usize),
            ],
            DType::Void,
            Scope {
                declarations: vec![],
            },
            vec![loop_stmt],
        );

        let program = AstNode::program(vec![func], "main");

        // Apply suggester
        let suggester = KernelizeSuggester::new();
        let suggestions = suggester.suggest(&program);

        // Should have one suggestion
        assert_eq!(suggestions.len(), 1);

        // Check the suggested program
        if let AstNode::Program { functions, .. } = &suggestions[0] {
            // Should have 2 functions: the kernel and the original function
            assert_eq!(functions.len(), 2);

            // First should be the kernel
            if let AstNode::Kernel {
                name, statements, ..
            } = &functions[0]
            {
                assert_eq!(name, "loop_func_0_kernel");
                assert_eq!(statements.len(), 1);
                assert!(matches!(statements[0], AstNode::Range { .. }));
            } else {
                panic!("Expected Kernel node");
            }

            // Second should be the original function
            if let AstNode::Function { name, .. } = &functions[1] {
                assert_eq!(name, "loop_func_0");
            } else {
                panic!("Expected Function node");
            }
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_non_void_function_not_kernelized() {
        // Function that returns a value should not be kernelized
        let func = function(
            "compute",
            vec![("x".to_string(), DType::F32)],
            DType::F32,
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

        let suggester = KernelizeSuggester::new();
        let suggestions = suggester.suggest(&program);

        // Should have no suggestions (non-void functions can't be kernels)
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_function_without_loop_not_kernelized() {
        // Function without loops should not be kernelized
        let func = function(
            "simple",
            vec![("x".to_string(), DType::F32)],
            DType::Void,
            Scope {
                declarations: vec![],
            },
            vec![assign("x", const_val(ConstLiteral::F32(1.0)))],
        );

        let program = AstNode::program(vec![func], "main");

        let suggester = KernelizeSuggester::new();
        let suggestions = suggester.suggest(&program);

        // Should have no suggestions (no loops)
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_custom_kernel_suffix() {
        let loop_stmt = range_builder(
            "i",
            const_val(ConstLiteral::Usize(10)),
            store(var("output"), var("i"), const_val(ConstLiteral::F32(1.0))),
        )
        .build();

        let func = function(
            "my_func",
            vec![("output".to_string(), DType::Ptr(Box::new(DType::F32)))],
            DType::Void,
            Scope {
                declarations: vec![],
            },
            vec![loop_stmt],
        );

        let program = AstNode::program(vec![func], "main");

        let suggester = KernelizeSuggester::with_suffix("_gpu".to_string());
        let suggestions = suggester.suggest(&program);

        assert_eq!(suggestions.len(), 1);

        if let AstNode::Program { functions, .. } = &suggestions[0] {
            if let AstNode::Kernel { name, .. } = &functions[0] {
                assert_eq!(name, "my_func_gpu");
            } else {
                panic!("Expected Kernel node");
            }
        }
    }
}

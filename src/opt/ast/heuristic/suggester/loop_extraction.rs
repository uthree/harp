use crate::ast::helper::*;
use crate::ast::{AstNode, DType, Scope};
use crate::opt::ast::analysis::collect_used_variables;
use crate::opt::ast::heuristic::RewriteSuggester;
use std::cell::Cell;

/// Suggester that extracts top-level loops from functions into separate functions
///
/// This transformation makes loops more amenable to parallelization by isolating
/// them in their own functions. It also helps with code organization and optimization.
///
/// # Example transformation
///
/// Before:
/// ```c
/// void kernel_impl(float* input, float* output) {
///     float temp = 1.0;
///     for (size_t i = 0; i < 10; i++) {
///         output[i] = input[i] * temp;
///     }
/// }
/// ```
///
/// After:
/// ```c
/// void loop_func_0(float* input, float* output, float temp, size_t loop_bound) {
///     for (size_t i = 0; i < loop_bound; i++) {
///         output[i] = input[i] * temp;
///     }
/// }
///
/// void kernel_impl(float* input, float* output) {
///     float temp = 1.0;
///     loop_func_0(input, output, temp, 10);
/// }
/// ```
pub struct LoopExtractionSuggester {
    /// Counter for generating unique function names (using Cell for interior mutability)
    next_func_id: Cell<usize>,
}

impl LoopExtractionSuggester {
    pub fn new() -> Self {
        Self {
            next_func_id: Cell::new(0),
        }
    }

    /// Extract a single top-level loop from a function
    ///
    /// Returns a tuple of (extracted_function, call_statement) or None if extraction fails
    fn extract_loop(
        &self,
        loop_node: &AstNode,
        function_params: &[(String, DType)],
        function_scope: &Scope,
    ) -> Option<(AstNode, AstNode)> {
        // Extract loop components
        let (counter_name, start, max, step, body, unroll) = match loop_node {
            AstNode::Range {
                counter_name,
                start,
                max,
                step,
                body,
                unroll,
            } => (counter_name, start, max, step, body, unroll),
            _ => return None,
        };

        // Collect variables used in the loop (excluding the counter)
        let used_vars = collect_used_variables(loop_node);

        // Determine which variables need to be passed as arguments
        // We need to pass:
        // 1. Function parameters that are used
        // 2. Local variables that are used (from function scope)
        let mut loop_args = Vec::new();
        let mut call_args = Vec::new();

        // Add function parameters that are used
        for (param_name, param_type) in function_params {
            if used_vars.contains(param_name) {
                loop_args.push((param_name.clone(), param_type.clone()));
                call_args.push(var(param_name));
            }
        }

        // Add local variables that are used
        for decl in &function_scope.declarations {
            if used_vars.contains(&decl.name) {
                loop_args.push((decl.name.clone(), decl.dtype.clone()));
                call_args.push(var(&decl.name));
            }
        }

        // Add loop bound as a parameter (if it's not a constant, we pass the expression)
        // For simplicity, we'll add a size_t parameter for the loop bound
        let func_id = self.next_func_id.get();
        let loop_bound_param = format!("loop_bound_{}", func_id);
        loop_args.push((loop_bound_param.clone(), DType::Usize));
        call_args.push(*max.clone());

        // Generate function name
        let func_name = format!("loop_func_{}", func_id);
        self.next_func_id.set(func_id + 1);

        // Create the new loop with the bound parameter
        let new_loop = AstNode::Range {
            counter_name: counter_name.clone(),
            start: Box::new(*start.clone()),
            max: Box::new(var(&loop_bound_param)),
            step: Box::new(*step.clone()),
            body: Box::new(*body.clone()),
            unroll: *unroll,
        };

        // Create the extracted function
        let extracted_func = function(
            func_name.clone(),
            loop_args,
            DType::Void,
            Scope {
                declarations: Vec::new(),
            },
            vec![new_loop],
        );

        // Create the call statement
        let call_stmt = AstNode::CallFunction {
            name: func_name,
            args: call_args,
        };

        Some((extracted_func, call_stmt))
    }
}

impl Default for LoopExtractionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteSuggester for LoopExtractionSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Only process Program nodes
        if let AstNode::Program {
            functions,
            entry_point,
        } = ast
        {
            // For each function in the program
            for func in functions {
                if let AstNode::Function {
                    name,
                    scope,
                    statements,
                    arguments,
                    return_type,
                } = func
                {
                    // Find top-level loops (Range nodes at the statement level)
                    let mut new_statements = Vec::new();
                    let mut extracted_functions = Vec::new();
                    let mut has_extraction = false;

                    for stmt in statements {
                        if let AstNode::Range { .. } = stmt {
                            // Try to extract this loop
                            if let Some((extracted_func, call_stmt)) =
                                self.extract_loop(stmt, arguments, scope)
                            {
                                // Replace loop with call
                                new_statements.push(call_stmt);
                                extracted_functions.push(extracted_func);
                                has_extraction = true;
                            } else {
                                // Keep original if extraction fails
                                new_statements.push(stmt.clone());
                            }
                        } else {
                            new_statements.push(stmt.clone());
                        }
                    }

                    // If we extracted any loops, create a new program with the modified function
                    if has_extraction {
                        let modified_func = function(
                            name.clone(),
                            arguments.clone(),
                            return_type.clone(),
                            scope.clone(),
                            new_statements,
                        );

                        // Build new function list: extracted functions + modified original + other functions
                        let mut new_functions = extracted_functions;
                        new_functions.push(modified_func);

                        // Add other functions (not the one we just modified)
                        for f in functions {
                            if let AstNode::Function {
                                name: fname,
                                scope: _,
                                statements: _,
                                arguments: _,
                                return_type: _,
                            } = f
                            {
                                if fname != name {
                                    new_functions.push(f.clone());
                                }
                            } else {
                                // Keep non-function nodes (e.g., kernels)
                                new_functions.push(f.clone());
                            }
                        }

                        suggestions.push(AstNode::program(new_functions, entry_point.clone()));
                    }
                }
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{ConstLiteral, VariableDecl};

    #[test]
    fn test_simple_loop_extraction() {
        // Create a simple function with a loop:
        // void foo(float* output) {
        //     for (size_t i = 0; i < 10; i++) {
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
            "foo",
            vec![("output".to_string(), DType::Ptr(Box::new(DType::F32)))],
            DType::Void,
            Scope {
                declarations: Vec::new(),
            },
            vec![loop_stmt],
        );

        let program = AstNode::program(vec![func], "foo");

        // Apply suggester
        let suggester = LoopExtractionSuggester::new();
        let suggestions = suggester.suggest(&program);

        // Should have one suggestion
        assert_eq!(suggestions.len(), 1);

        // Check the suggested program
        if let AstNode::Program {
            functions,
            entry_point,
        } = &suggestions[0]
        {
            assert_eq!(entry_point, "foo");
            // Should have 2 functions: loop_func_0 and foo
            assert_eq!(functions.len(), 2);

            // First function should be the extracted loop
            if let AstNode::Function {
                name,
                statements,
                arguments,
                ..
            } = &functions[0]
            {
                assert_eq!(name, "loop_func_0");
                // Should have output pointer and loop_bound parameter
                assert_eq!(arguments.len(), 2);
                assert_eq!(arguments[0].0, "output");
                assert_eq!(arguments[1].0, "loop_bound_0");
                // Should contain the loop
                assert_eq!(statements.len(), 1);
                assert!(matches!(statements[0], AstNode::Range { .. }));
            } else {
                panic!("Expected Function node");
            }

            // Second function should be the modified original
            if let AstNode::Function {
                name,
                statements,
                arguments,
                ..
            } = &functions[1]
            {
                assert_eq!(name, "foo");
                assert_eq!(arguments.len(), 1);
                // Should contain a function call instead of the loop
                assert_eq!(statements.len(), 1);
                if let AstNode::CallFunction { name, args } = &statements[0] {
                    assert_eq!(name, "loop_func_0");
                    assert_eq!(args.len(), 2); // output + loop_bound
                } else {
                    panic!("Expected CallFunction node");
                }
            } else {
                panic!("Expected Function node");
            }
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_loop_with_local_variable() {
        // void foo(float* output) {
        //     float multiplier = 2.0;
        //     for (size_t i = 0; i < 10; i++) {
        //         output[i] = multiplier;
        //     }
        // }
        let loop_stmt = range_builder(
            "i",
            const_val(ConstLiteral::Usize(10)),
            store(var("output"), var("i"), var("multiplier")),
        )
        .start(const_val(ConstLiteral::Usize(0)))
        .step(const_val(ConstLiteral::Usize(1)))
        .build();

        let func = function(
            "foo",
            vec![("output".to_string(), DType::Ptr(Box::new(DType::F32)))],
            DType::Void,
            Scope {
                declarations: vec![VariableDecl {
                    name: "multiplier".to_string(),
                    dtype: DType::F32,
                    constant: false,
                    size_expr: None,
                }],
            },
            vec![
                assign("multiplier", const_val(ConstLiteral::F32(2.0))),
                loop_stmt,
            ],
        );

        let program = AstNode::program(vec![func], "foo");

        let suggester = LoopExtractionSuggester::new();
        let suggestions = suggester.suggest(&program);

        assert_eq!(suggestions.len(), 1);

        if let AstNode::Program { functions, .. } = &suggestions[0] {
            // Extracted function should have 3 parameters: output, multiplier, loop_bound
            if let AstNode::Function { arguments, .. } = &functions[0] {
                assert_eq!(arguments.len(), 3);
                assert_eq!(arguments[0].0, "output");
                assert_eq!(arguments[1].0, "multiplier");
                assert_eq!(arguments[2].0, "loop_bound_0");
            } else {
                panic!("Expected Function node");
            }
        }
    }

    #[test]
    fn test_no_extraction_without_loops() {
        // Function without loops should not be modified
        let func = function(
            "foo",
            vec![("x".to_string(), DType::F32)],
            DType::Void,
            Scope {
                declarations: Vec::new(),
            },
            vec![assign("x", const_val(ConstLiteral::F32(1.0)))],
        );

        let program = AstNode::program(vec![func], "foo");

        let suggester = LoopExtractionSuggester::new();
        let suggestions = suggester.suggest(&program);

        // No suggestions should be made
        assert_eq!(suggestions.len(), 0);
    }
}

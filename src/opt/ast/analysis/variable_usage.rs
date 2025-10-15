use crate::ast::AstNode;
use std::collections::HashSet;

/// Collect all variable names used (read) in an AST subtree
///
/// This function recursively traverses the AST and collects all variable names
/// that are referenced (not just assigned to). This is useful for determining
/// which variables need to be passed as arguments when extracting code into
/// a separate function.
///
/// # Arguments
/// * `node` - The AST node to analyze
///
/// # Returns
/// A set of variable names that are used in the given AST subtree
pub fn collect_used_variables(node: &AstNode) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_used_variables_recursive(node, &mut vars);
    vars
}

fn collect_used_variables_recursive(node: &AstNode, vars: &mut HashSet<String>) {
    match node {
        // Variable reference - this is what we're looking for
        AstNode::Var(name) => {
            vars.insert(name.clone());
        }

        // Assignment - collect from RHS only (LHS is a definition, not a use)
        AstNode::Assign(_lhs, rhs) => {
            collect_used_variables_recursive(rhs, vars);
        }

        // Load - collect from target and index
        AstNode::Load {
            target,
            index,
            vector_width: _,
        } => {
            collect_used_variables_recursive(target, vars);
            collect_used_variables_recursive(index, vars);
        }

        // Store - collect from target, index, and value
        AstNode::Store {
            target,
            index,
            value,
            vector_width: _,
        } => {
            collect_used_variables_recursive(target, vars);
            collect_used_variables_recursive(index, vars);
            collect_used_variables_recursive(value, vars);
        }

        // Range - collect from bounds and body, but exclude the counter variable
        // from the results since it's defined by the loop itself
        AstNode::Range {
            counter_name,
            start,
            max,
            step,
            body,
            unroll: _,
        } => {
            collect_used_variables_recursive(start, vars);
            collect_used_variables_recursive(max, vars);
            collect_used_variables_recursive(step, vars);
            collect_used_variables_recursive(body, vars);
            // Remove the counter variable since it's defined by the loop
            vars.remove(counter_name);
        }

        // If - collect from condition and both branches
        AstNode::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_used_variables_recursive(condition, vars);
            collect_used_variables_recursive(then_branch, vars);
            if let Some(else_br) = else_branch {
                collect_used_variables_recursive(else_br, vars);
            }
        }

        // Block - collect from statements, but exclude variables declared in the scope
        AstNode::Block { scope, statements } => {
            for stmt in statements {
                collect_used_variables_recursive(stmt, vars);
            }
            // Remove variables that are declared in this scope
            for decl in &scope.declarations {
                vars.remove(&decl.name);
            }
        }

        // Binary operations
        AstNode::Add(a, b)
        | AstNode::Mul(a, b)
        | AstNode::Max(a, b)
        | AstNode::Rem(a, b)
        | AstNode::LessThan(a, b)
        | AstNode::Eq(a, b)
        | AstNode::BitAnd(a, b)
        | AstNode::BitOr(a, b)
        | AstNode::BitXor(a, b)
        | AstNode::Shl(a, b)
        | AstNode::Shr(a, b) => {
            collect_used_variables_recursive(a, vars);
            collect_used_variables_recursive(b, vars);
        }

        // Unary operations
        AstNode::Neg(a)
        | AstNode::Recip(a)
        | AstNode::Sin(a)
        | AstNode::Sqrt(a)
        | AstNode::Log2(a)
        | AstNode::Exp2(a)
        | AstNode::BitNot(a) => {
            collect_used_variables_recursive(a, vars);
        }

        // Select (ternary)
        AstNode::Select {
            cond,
            true_val,
            false_val,
        } => {
            collect_used_variables_recursive(cond, vars);
            collect_used_variables_recursive(true_val, vars);
            collect_used_variables_recursive(false_val, vars);
        }

        // Cast
        AstNode::Cast { dtype: _, expr } => {
            collect_used_variables_recursive(expr, vars);
        }

        // Function call
        AstNode::CallFunction { name: _, args } => {
            for arg in args {
                collect_used_variables_recursive(arg, vars);
            }
        }

        // Kernel call
        AstNode::CallKernel {
            name: _,
            args,
            global_size,
            local_size,
        } => {
            for arg in args {
                collect_used_variables_recursive(arg, vars);
            }
            for size in global_size {
                collect_used_variables_recursive(size, vars);
            }
            for size in local_size {
                collect_used_variables_recursive(size, vars);
            }
        }

        // Function/Kernel definition - collect from body
        AstNode::Function {
            name: _,
            scope,
            statements,
            arguments,
            return_type: _,
        } => {
            for stmt in statements {
                collect_used_variables_recursive(stmt, vars);
            }
            // Remove parameters from the set
            for (param_name, _) in arguments {
                vars.remove(param_name);
            }
            // Remove local variables
            for decl in &scope.declarations {
                vars.remove(&decl.name);
            }
        }

        AstNode::Kernel {
            name: _,
            scope,
            statements,
            arguments,
            return_type: _,
            global_size,
            local_size,
        } => {
            for stmt in statements {
                collect_used_variables_recursive(stmt, vars);
            }
            for size in global_size {
                collect_used_variables_recursive(size, vars);
            }
            for size in local_size {
                collect_used_variables_recursive(size, vars);
            }
            // Remove parameters
            for (param_name, _) in arguments {
                vars.remove(param_name);
            }
            // Remove thread IDs
            for thread_id in &scope.thread_ids {
                vars.remove(&thread_id.name);
            }
            // Remove local variables
            for decl in &scope.declarations {
                vars.remove(&decl.name);
            }
        }

        // Program - collect from all functions
        AstNode::Program {
            functions,
            entry_point: _,
        } => {
            for func in functions {
                collect_used_variables_recursive(func, vars);
            }
        }

        // Terminals - no variables to collect
        AstNode::Const(_) | AstNode::Rand | AstNode::Barrier | AstNode::Drop(_) => {}

        // Pattern matching placeholder
        AstNode::Capture(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;
    use crate::ast::ConstLiteral;

    #[test]
    fn test_simple_variable_usage() {
        // x + y
        let expr = add(var("x"), var("y"));
        let vars = collect_used_variables(&expr);
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn test_assignment_only_rhs() {
        // z = x + y
        let stmt = assign("z", add(var("x"), var("y")));
        let vars = collect_used_variables(&stmt);
        // z is not in the set because it's being assigned to, not used
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("z"));
    }

    #[test]
    fn test_loop_excludes_counter() {
        // for (i = 0; i < n; i++) { sum = sum + i; }
        let loop_stmt = range_builder("i", var("n"), assign("sum", add(var("sum"), var("i"))))
            .start(const_val(ConstLiteral::Usize(0)))
            .step(const_val(ConstLiteral::Usize(1)))
            .build();
        let vars = collect_used_variables(&loop_stmt);
        // i should not be in the set (it's the loop counter)
        // n and sum should be in the set
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("n"));
        assert!(vars.contains("sum"));
        assert!(!vars.contains("i"));
    }

    #[test]
    fn test_load_and_store() {
        // output[i] = input[i] * 2
        let stmt = store(
            var("output"),
            var("i"),
            mul(
                load(var("input"), var("i")),
                const_val(ConstLiteral::F32(2.0)),
            ),
        );
        let vars = collect_used_variables(&stmt);
        assert_eq!(vars.len(), 3);
        assert!(vars.contains("output"));
        assert!(vars.contains("input"));
        assert!(vars.contains("i"));
    }

    #[test]
    fn test_block_excludes_locals() {
        use crate::ast::{DType, Scope, VariableDecl};

        // { float temp; temp = x + y; z = temp * 2; }
        let scope = Scope {
            declarations: vec![VariableDecl {
                name: "temp".to_string(),
                dtype: DType::F32,
                constant: false,
                size_expr: None,
            }],
        };
        let block = AstNode::Block {
            scope,
            statements: vec![
                assign("temp", add(var("x"), var("y"))),
                assign("z", mul(var("temp"), const_val(ConstLiteral::F32(2.0)))),
            ],
        };
        let vars = collect_used_variables(&block);
        // temp should not be in the set (it's declared in the block)
        assert_eq!(vars.len(), 2);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
        assert!(!vars.contains("temp"));
    }

    #[test]
    fn test_function_excludes_params_and_locals() {
        use crate::ast::{DType, Scope, VariableDecl};

        // void foo(float* a, float b) { float temp = b * 2; a[0] = temp + c; }
        let func = function(
            "foo",
            vec![
                ("a".to_string(), DType::Ptr(Box::new(DType::F32))),
                ("b".to_string(), DType::F32),
            ],
            DType::Void,
            Scope {
                declarations: vec![VariableDecl {
                    name: "temp".to_string(),
                    dtype: DType::F32,
                    constant: false,
                    size_expr: None,
                }],
            },
            vec![
                assign("temp", mul(var("b"), const_val(ConstLiteral::F32(2.0)))),
                store(
                    var("a"),
                    const_val(ConstLiteral::Usize(0)),
                    add(var("temp"), var("c")),
                ),
            ],
        );
        let vars = collect_used_variables(&func);
        // Only c should be in the set (a, b, temp are all defined in the function)
        assert_eq!(vars.len(), 1);
        assert!(vars.contains("c"));
        assert!(!vars.contains("a"));
        assert!(!vars.contains("b"));
        assert!(!vars.contains("temp"));
    }
}

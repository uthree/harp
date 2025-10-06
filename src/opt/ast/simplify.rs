use crate::ast::{pattern::AstRewriteRule, pattern::AstRewriter, AstNode, ConstLiteral};
use crate::{ast_pattern, ast_rewriter};
use std::rc::Rc;

/// Removes consecutive Barrier nodes in a Block, leaving only one.
/// This is useful for optimizing generated code that may have redundant synchronization points.
pub fn coalesce_barriers(ast: &mut AstNode) {
    match ast {
        AstNode::Block { ref mut statements, .. } => {
            let mut new_statements = Vec::new();
            let mut last_was_barrier = false;

            for stmt in statements.iter() {
                match stmt {
                    AstNode::Barrier => {
                        if !last_was_barrier {
                            new_statements.push(stmt.clone());
                            last_was_barrier = true;
                        }
                        // Skip consecutive barriers
                    }
                    _ => {
                        new_statements.push(stmt.clone());
                        last_was_barrier = false;
                    }
                }
            }

            *statements = new_statements;

            // Recursively apply to nested blocks
            for stmt in statements.iter_mut() {
                coalesce_barriers(stmt);
            }
        }
        _ => {
            // Recursively apply to children
            let children: Vec<AstNode> = ast.children().into_iter().cloned().collect();
            for (i, mut child) in children.into_iter().enumerate() {
                coalesce_barriers(&mut child);
                let mut new_children: Vec<AstNode> = ast.children().into_iter().cloned().collect();
                new_children[i] = child;
                *ast = ast.clone().replace_children(new_children);
            }
        }
    }
}

/// Creates an AstRewriter that removes meaningless operations.
///
/// Examples of simplifications:
/// - `a + 0` -> `a`
/// - `a * 1` -> `a`
/// - `a * 0` -> `0`
/// - `a / 1` -> `a`
/// - `-(-a)` -> `a`
/// - `recip(recip(a))` -> `a`
pub fn simplify_rewriter() -> AstRewriter {
    // Helper functions for creating constants
    fn i(val: isize) -> AstNode {
        AstNode::from(val)
    }
    fn f(val: f32) -> AstNode {
        AstNode::from(val)
    }
    fn u(val: usize) -> AstNode {
        AstNode::from(val)
    }

    // Identity helper: checks if a node is an identity element for an operation
    #[allow(dead_code)]
    fn is_zero(node: &AstNode) -> bool {
        match node {
            AstNode::Const(ConstLiteral::Isize(0)) => true,
            AstNode::Const(ConstLiteral::Usize(0)) => true,
            AstNode::Const(ConstLiteral::F32(f)) if *f == 0.0 => true,
            _ => false,
        }
    }

    #[allow(dead_code)]
    fn is_one(node: &AstNode) -> bool {
        match node {
            AstNode::Const(ConstLiteral::Isize(1)) => true,
            AstNode::Const(ConstLiteral::Usize(1)) => true,
            AstNode::Const(ConstLiteral::F32(f)) if *f == 1.0 => true,
            _ => false,
        }
    }

    ast_rewriter!(
        "simplify",
        // Addition with zero
        ast_pattern!(|a| a + i(0) => a.clone()),
        ast_pattern!(|a| a + f(0.0) => a.clone()),
        ast_pattern!(|a| a + u(0) => a.clone()),
        // Multiplication with one
        ast_pattern!(|a| a * i(1) => a.clone()),
        ast_pattern!(|a| a * f(1.0) => a.clone()),
        ast_pattern!(|a| a * u(1) => a.clone()),
        // Multiplication with zero
        ast_pattern!(|_a| _a * i(0) => i(0)),
        ast_pattern!(|_a| _a * f(0.0) => f(0.0)),
        ast_pattern!(|_a| _a * u(0) => u(0)),
        // Division by one
        ast_pattern!(|a| a / i(1) => a.clone()),
        ast_pattern!(|a| a / f(1.0) => a.clone()),
        ast_pattern!(|a| a / u(1) => a.clone()),
        // Double negation: -(-a) -> a
        Rc::clone(&{
            let a = AstNode::Capture(0);
            let pattern = AstNode::Neg(Box::new(AstNode::Neg(Box::new(a))));
            AstRewriteRule::new(
                pattern,
                |captured_nodes: &[AstNode]| captured_nodes[0].clone(),
                |_| true,
            )
        }),
        // Double reciprocal: recip(recip(a)) -> a
        Rc::clone(&{
            let a = AstNode::Capture(0);
            let pattern = AstNode::Recip(Box::new(AstNode::Recip(Box::new(a))));
            AstRewriteRule::new(
                pattern,
                |captured_nodes: &[AstNode]| captured_nodes[0].clone(),
                |_| true,
            )
        }),
        // Remainder with 1 is always 0
        ast_pattern!(|_a| _a % i(1) => i(0)),
        ast_pattern!(|_a| _a % f(1.0) => f(0.0)),
        ast_pattern!(|_a| _a % u(1) => u(0))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i(val: isize) -> AstNode {
        AstNode::from(val)
    }

    fn f(val: f32) -> AstNode {
        AstNode::from(val)
    }

    #[test]
    fn test_add_zero() {
        let rewriter = simplify_rewriter();
        let mut ast = i(5) + i(0);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(5));
    }

    #[test]
    fn test_mul_one() {
        let rewriter = simplify_rewriter();
        let mut ast = f(3.5) * f(1.0);
        rewriter.apply(&mut ast);
        assert_eq!(ast, f(3.5));
    }

    #[test]
    fn test_mul_zero() {
        let rewriter = simplify_rewriter();
        let mut ast = i(42) * i(0);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(0));
    }

    #[test]
    fn test_div_one() {
        let rewriter = simplify_rewriter();
        let mut ast = i(10) / i(1);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(10));
    }

    #[test]
    fn test_double_negation() {
        let rewriter = simplify_rewriter();
        let mut ast = -(-i(5));
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(5));
    }

    #[test]
    fn test_double_reciprocal() {
        let rewriter = simplify_rewriter();
        let mut ast = i(5).recip().recip();
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(5));
    }

    #[test]
    fn test_remainder_one() {
        let rewriter = simplify_rewriter();
        let mut ast = i(42) % i(1);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(0));
    }

    #[test]
    fn test_nested_simplification() {
        let rewriter = simplify_rewriter();
        // (5 + 0) * 1
        let mut ast = (i(5) + i(0)) * i(1);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(5));
    }

    #[test]
    fn test_coalesce_barriers() {
        use crate::ast::Scope;

        // Create a block with consecutive barriers
        let mut ast = AstNode::Block {
            scope: Scope { declarations: vec![] },
            statements: vec![
                AstNode::Assign("x".to_string(), Box::new(i(1))),
                AstNode::Barrier,
                AstNode::Barrier,
                AstNode::Barrier,
                AstNode::Assign("y".to_string(), Box::new(i(2))),
                AstNode::Barrier,
                AstNode::Assign("z".to_string(), Box::new(i(3))),
            ],
        };

        coalesce_barriers(&mut ast);

        if let AstNode::Block { statements, .. } = ast {
            assert_eq!(statements.len(), 5);
            assert!(matches!(statements[0], AstNode::Assign(_, _)));
            assert!(matches!(statements[1], AstNode::Barrier));
            assert!(matches!(statements[2], AstNode::Assign(_, _)));
            assert!(matches!(statements[3], AstNode::Barrier));
            assert!(matches!(statements[4], AstNode::Assign(_, _)));
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_coalesce_barriers_nested() {
        use crate::ast::Scope;

        // Create nested blocks with barriers
        let inner_block = AstNode::Block {
            scope: Scope { declarations: vec![] },
            statements: vec![
                AstNode::Barrier,
                AstNode::Barrier,
                AstNode::Assign("a".to_string(), Box::new(i(1))),
            ],
        };

        let mut ast = AstNode::Block {
            scope: Scope { declarations: vec![] },
            statements: vec![
                AstNode::Barrier,
                AstNode::Barrier,
                inner_block,
                AstNode::Barrier,
            ],
        };

        coalesce_barriers(&mut ast);

        if let AstNode::Block { statements, .. } = ast {
            assert_eq!(statements.len(), 3);
            assert!(matches!(statements[0], AstNode::Barrier));
            assert!(matches!(statements[1], AstNode::Block { .. }));
            assert!(matches!(statements[2], AstNode::Barrier));

            // Check inner block
            if let AstNode::Block { statements: inner_stmts, .. } = &statements[1] {
                assert_eq!(inner_stmts.len(), 2);
                assert!(matches!(inner_stmts[0], AstNode::Barrier));
                assert!(matches!(inner_stmts[1], AstNode::Assign(_, _)));
            } else {
                panic!("Expected inner Block");
            }
        } else {
            panic!("Expected Block");
        }
    }
}

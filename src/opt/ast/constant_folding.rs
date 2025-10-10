use crate::ast::{pattern::AstRewriter, AstNode, ConstLiteral};
use crate::{ast_pattern, ast_rewriter};

/// Creates an AstRewriter that folds constant expressions at compile time.
///
/// Examples:
/// - `2 + 3` -> `5`
/// - `4 * 5` -> `20`
/// - `-(2)` -> `-2`
/// - `sin(0.0)` -> `0.0`
pub fn constant_folding_rewriter() -> AstRewriter {
    ast_rewriter!(
        "constant_folding",
        // Addition
        ast_pattern!(|a, b| a + b, if is_const(a) && is_const(b) => {
            fold_binary_op(a, b, |x, y| x + y)
        }),
        // Multiplication
        ast_pattern!(|a, b| a * b, if is_const(a) && is_const(b) => {
            fold_binary_op(a, b, |x, y| x * y)
        }),
        // Division
        ast_pattern!(|a, b| a / b, if is_const(a) && is_const(b) => {
            fold_binary_op(a, b, |x, y| x / y)
        }),
        // Remainder
        ast_pattern!(|a, b| a % b, if is_const(a) && is_const(b) => {
            fold_binary_op(a, b, |x, y| x % y)
        }),
        // Negation
        ast_pattern!(|a| AstNode::Neg(Box::new(a.clone())), if is_const(a) => {
            fold_unary_op(a, |x| -x)
        }),
        // Reciprocal
        ast_pattern!(|a| AstNode::Recip(Box::new(a.clone())), if is_const(a) => {
            fold_unary_op(a, |x| x.recip())
        }),
        // Sin
        ast_pattern!(|a| AstNode::Sin(Box::new(a.clone())), if is_const(a) => {
            fold_unary_op(a, |x| x.sin())
        }),
        // Sqrt
        ast_pattern!(|a| AstNode::Sqrt(Box::new(a.clone())), if is_const(a) => {
            fold_unary_op(a, |x| x.sqrt())
        }),
        // Exp2
        ast_pattern!(|a| AstNode::Exp2(Box::new(a.clone())), if is_const(a) => {
            fold_unary_op(a, |x| x.exp2())
        }),
        // Log2
        ast_pattern!(|a| AstNode::Log2(Box::new(a.clone())), if is_const(a) => {
            fold_unary_op(a, |x| x.log2())
        })
    )
}

fn is_const(node: &AstNode) -> bool {
    matches!(node, AstNode::Const(_))
}

fn const_to_f32(lit: &ConstLiteral) -> f32 {
    match lit {
        ConstLiteral::F32(f) => *f,
        ConstLiteral::Isize(i) => *i as f32,
        ConstLiteral::Usize(u) => *u as f32,
        ConstLiteral::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn fold_binary_op<F>(a: &AstNode, b: &AstNode, op: F) -> AstNode
where
    F: Fn(f32, f32) -> f32,
{
    if let (AstNode::Const(a_lit), AstNode::Const(b_lit)) = (a, b) {
        let a_val = const_to_f32(a_lit);
        let b_val = const_to_f32(b_lit);
        let result = op(a_val, b_val);

        // Try to preserve the type of the first operand
        match a_lit {
            ConstLiteral::Isize(_) => {
                if result.fract() == 0.0 && result.is_finite() {
                    AstNode::from(result as isize)
                } else {
                    AstNode::from(result)
                }
            }
            ConstLiteral::Usize(_) => {
                if result >= 0.0 && result.fract() == 0.0 && result.is_finite() {
                    AstNode::from(result as usize)
                } else {
                    AstNode::from(result)
                }
            }
            ConstLiteral::F32(_) => AstNode::from(result),
            ConstLiteral::Bool(_) => AstNode::from(result != 0.0),
        }
    } else {
        unreachable!("fold_binary_op called with non-const nodes")
    }
}

fn fold_unary_op<F>(a: &AstNode, op: F) -> AstNode
where
    F: Fn(f32) -> f32,
{
    if let AstNode::Const(a_lit) = a {
        let a_val = const_to_f32(a_lit);
        let result = op(a_val);

        // Try to preserve the type
        match a_lit {
            ConstLiteral::Isize(_) => {
                if result.fract() == 0.0 && result.is_finite() {
                    AstNode::from(result as isize)
                } else {
                    AstNode::from(result)
                }
            }
            ConstLiteral::Usize(_) => {
                if result >= 0.0 && result.fract() == 0.0 && result.is_finite() {
                    AstNode::from(result as usize)
                } else {
                    AstNode::from(result)
                }
            }
            ConstLiteral::F32(_) => AstNode::from(result),
            ConstLiteral::Bool(_) => AstNode::from(result != 0.0),
        }
    } else {
        unreachable!("fold_unary_op called with non-const node")
    }
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
    fn test_add_constants() {
        let rewriter = constant_folding_rewriter();
        let mut ast = i(2) + i(3);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(5));
    }

    #[test]
    fn test_mul_constants() {
        let rewriter = constant_folding_rewriter();
        let mut ast = i(4) * i(5);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(20));
    }

    #[test]
    fn test_div_constants() {
        let rewriter = constant_folding_rewriter();
        let mut ast = i(20) / i(4);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(5));
    }

    #[test]
    fn test_rem_constants() {
        let rewriter = constant_folding_rewriter();
        let mut ast = i(10) % i(3);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(1));
    }

    #[test]
    fn test_neg_constant() {
        let rewriter = constant_folding_rewriter();
        let mut ast = -i(5);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(-5));
    }

    #[test]
    fn test_recip_constant() {
        let rewriter = constant_folding_rewriter();
        let mut ast = f(2.0).recip();
        rewriter.apply(&mut ast);
        assert_eq!(ast, f(0.5));
    }

    #[test]
    fn test_sin_constant() {
        let rewriter = constant_folding_rewriter();
        let mut ast = f(0.0).sin();
        rewriter.apply(&mut ast);
        assert_eq!(ast, f(0.0));
    }

    #[test]
    fn test_sqrt_constant() {
        let rewriter = constant_folding_rewriter();
        let mut ast = f(4.0).sqrt();
        rewriter.apply(&mut ast);
        assert_eq!(ast, f(2.0));
    }

    #[test]
    fn test_nested_folding() {
        let rewriter = constant_folding_rewriter();
        // (2 + 3) * 4
        let mut ast = (i(2) + i(3)) * i(4);
        rewriter.apply(&mut ast);
        assert_eq!(ast, i(20));
    }

    #[test]
    fn test_mixed_types() {
        let rewriter = constant_folding_rewriter();
        // 2.5 + 3.0
        let mut ast = f(2.5) + f(3.0);
        rewriter.apply(&mut ast);
        assert_eq!(ast, f(5.5));
    }
}

use crate::ast::pattern::{AstRewriteRule, AstRewriter};
use crate::ast::{AstNode, AstOp, Const};
use crate::astpat;
use std::rc::Rc;

fn is_const(node: &AstNode) -> bool {
    matches!(node.op, AstOp::Const(_))
}

fn get_const_val(node: &AstNode) -> Option<Const> {
    if let AstOp::Const(c) = &node.op {
        Some(c.clone())
    } else {
        None
    }
}

/// a * (b + c) => (a * b) + (a * c)
/// (a + b) * c => (a * c) + (b * c)
pub fn distributive_rules() -> AstRewriter {
    let rules: Vec<Rc<AstRewriteRule>> = vec![
        astpat!(|a, b, c| a * (b + c) => (a.clone() * b) + (a * c)),
        astpat!(|a, b, c| (a + b) * c => (a * c.clone()) + (b * c)),
    ];
    AstRewriter::with_rules("distributive rules", rules)
}

/// a + b => b + a
/// a * b => b * a
/// WARNING: These rules can cause infinite loops with the recursive `apply` method.
/// They are better suited for heuristic optimizers that can control application,
/// such as `get_possible_rewrites`.
pub fn commutative_rules() -> AstRewriter {
    let rules: Vec<Rc<AstRewriteRule>> = vec![
        astpat!(|a, b| a + b => b + a),
        astpat!(|a, b| a * b => b * a),
    ];
    AstRewriter::with_rules("commutative rules", rules)
}

/// (a + b) + c => a + (b + c)
/// (a * b) * c => a * (b * c)
pub fn associative_rules() -> AstRewriter {
    let rules: Vec<Rc<AstRewriteRule>> = vec![
        astpat!(|a, b, c| (a + b) + c => a + (b + c)),
        astpat!(|a, b, c| (a * b) * c => a * (b * c)),
    ];
    AstRewriter::with_rules("associative rules", rules)
}

pub fn algebraic_simplification() -> AstRewriter {
    let rules: Vec<Rc<AstRewriteRule>> = [
        // --- Identity Rules ---
        vec![
            astpat!(|a| a + AstNode::from(0isize) => a),
            astpat!(|a| AstNode::from(0isize) + a => a),
            astpat!(|a| a - AstNode::from(0isize) => a),
            astpat!(|a| a * AstNode::from(1isize) => a),
            astpat!(|a| AstNode::from(1isize) * a => a),
            astpat!(|a| a / AstNode::from(1isize) => a),
        ],
        // --- Annihilation Rules ---
        vec![
            astpat!(|_a| _a * AstNode::from(0isize) => AstNode::from(0isize)),
            astpat!(|_a| AstNode::from(0isize) * _a => AstNode::from(0isize)),
            astpat!(|_a| AstNode::from(0isize) / _a => AstNode::from(0isize)),
        ],
        // --- Other Rules ---
        vec![astpat!(|a| -(-a) => a)],
        // --- Constant Folding Rules ---
        vec![
            astpat!(|a, b| a + b, if is_const(&a) && is_const(&b) => {
                if let (Some(Const::Isize(va)), Some(Const::Isize(vb))) = (get_const_val(&a), get_const_val(&b)) {
                    AstNode::from(va + vb)
                } else {
                    a + b
                }
            }),
            astpat!(|a, b| a - b, if is_const(&a) && is_const(&b) => {
                if let (Some(Const::Isize(va)), Some(Const::Isize(vb))) = (get_const_val(&a), get_const_val(&b)) {
                    AstNode::from(va - vb)
                } else {
                    a - b
                }
            }),
            astpat!(|a, b| a * b, if is_const(&a) && is_const(&b) => {
                if let (Some(Const::Isize(va)), Some(Const::Isize(vb))) = (get_const_val(&a), get_const_val(&b)) {
                    AstNode::from(va * vb)
                } else {
                    a * b
                }
            }),
        ],
    ]
    .concat();
    AstRewriter::with_rules("AlgebraicSimplification", rules)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{AstNode, DType},
        opt::ast::{AstOptimizer, RulebasedAstOptimizer},
    };

    fn assert_optimization(original: AstNode, expected: AstNode) {
        let mut optimizer = RulebasedAstOptimizer::new(algebraic_simplification());
        let optimized = optimizer.optimize(&original);
        assert_eq!(optimized, expected);
    }

    #[test]
    fn test_add_zero() {
        assert_optimization(
            AstNode::var("x", DType::Isize) + AstNode::from(0isize),
            AstNode::var("x", DType::Isize),
        );
        assert_optimization(
            AstNode::from(0isize) + AstNode::var("x", DType::Isize),
            AstNode::var("x", DType::Isize),
        );
    }

    #[test]
    fn test_mul_one() {
        assert_optimization(
            AstNode::var("x", DType::Isize) * AstNode::from(1isize),
            AstNode::var("x", DType::Isize),
        );
        assert_optimization(
            AstNode::from(1isize) * AstNode::var("x", DType::Isize),
            AstNode::var("x", DType::Isize),
        );
    }

    #[test]
    fn test_mul_zero() {
        assert_optimization(
            AstNode::var("x", DType::Isize) * AstNode::from(0isize),
            AstNode::from(0isize),
        );
        assert_optimization(
            AstNode::from(0isize) * AstNode::var("x", DType::Isize),
            AstNode::from(0isize),
        );
    }

    #[test]
    fn test_double_negation() {
        assert_optimization(
            -(-AstNode::var("x", DType::Isize)),
            AstNode::var("x", DType::Isize),
        );
    }

    #[test]
    fn test_constant_folding_isize() {
        assert_optimization(
            AstNode::from(2isize) + AstNode::from(3isize),
            AstNode::from(5isize),
        );
        assert_optimization(
            AstNode::from(5isize) - AstNode::from(3isize),
            AstNode::from(2isize),
        );
        assert_optimization(
            AstNode::from(2isize) * AstNode::from(3isize),
            AstNode::from(6isize),
        );
    }

    #[test]
    fn test_recursive_optimization() {
        assert_optimization(
            (AstNode::var("x", DType::Isize) + AstNode::from(0isize)) * AstNode::from(1isize),
            AstNode::var("x", DType::Isize),
        );
        assert_optimization(
            (AstNode::from(2isize) + AstNode::from(3isize)) * AstNode::var("x", DType::Isize),
            AstNode::from(5isize) * AstNode::var("x", DType::Isize),
        );
    }
}

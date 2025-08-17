use crate::ast::pattern::{AstRewriteRule, AstRewriter};
use crate::ast::{AstNode, AstOp, Const};
use crate::astpat;
use crate::opt::ast::AstOptimizer;
use crate::opt::ast::heuristic::handcode::{associative_rules, distributive_rules};
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
        + associative_rules()
        + distributive_rules()
}

pub struct AlgebraicOptimizer {
    rewriter: AstRewriter,
}

impl Default for AlgebraicOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AlgebraicOptimizer {
    pub fn new() -> Self {
        let rewriter = algebraic_simplification();
        Self { rewriter }
    }

    pub fn optimize(&self, ast: &AstNode) -> AstNode {
        self.rewriter.apply(ast)
    }
}

impl AstOptimizer for AlgebraicOptimizer {
    fn optimize(&mut self, ast: &AstNode) -> AstNode {
        self.rewriter.apply(ast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, DType};

    fn assert_optimization(original: AstNode, expected: AstNode) {
        let optimizer = AlgebraicOptimizer::new();
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

    #[test]
    fn test_algebraic_laws() {
        let a = AstNode::var("a", DType::Isize);
        let b = AstNode::var("b", DType::Isize);
        let c = AstNode::var("c", DType::Isize);

        // Associative
        assert_optimization(
            (a.clone() + b.clone()) + c.clone(),
            a.clone() + (b.clone() + c.clone()),
        );

        // Distributive
        assert_optimization(
            a.clone() * (b.clone() + c.clone()),
            (a.clone() * b) + (a * c),
        );
    }
}

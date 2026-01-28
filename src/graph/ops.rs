//! Arithmetic and comparison operations for GraphNode

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::ast::{AstNode, DType};

use super::Expr;
use super::node::{GraphNode, GraphOp};
use super::shape::View;

// ============================================================================
// Binary Operation Helper
// ============================================================================

/// Compute broadcast shape and expand dimensions as needed.
/// Returns (expanded_lhs, expanded_rhs) where both have the same shape.
fn broadcast(lhs: &GraphNode, rhs: &GraphNode) -> (GraphNode, GraphNode) {
    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    // If shapes are already equal, no broadcasting needed
    if lhs_shape == rhs_shape {
        return (lhs.clone(), rhs.clone());
    }

    // Align shapes: prepend dimensions of 1 to the shorter one
    let lhs_ndim = lhs_shape.len();
    let rhs_ndim = rhs_shape.len();
    let max_ndim = lhs_ndim.max(rhs_ndim);

    // Pad lhs with leading 1s
    let mut lhs_node = lhs.clone();
    for _ in 0..(max_ndim - lhs_ndim) {
        lhs_node = lhs_node.unsqueeze(0);
    }

    // Pad rhs with leading 1s
    let mut rhs_node = rhs.clone();
    for _ in 0..(max_ndim - rhs_ndim) {
        rhs_node = rhs_node.unsqueeze(0);
    }

    let lhs_shape = lhs_node.shape();
    let rhs_shape = rhs_node.shape();

    // Expand dimensions of size 1
    for i in 0..max_ndim {
        let lhs_dim = &lhs_shape[i];
        let rhs_dim = &rhs_shape[i];

        if lhs_dim != rhs_dim {
            // One must be 1 for broadcasting to work
            if *lhs_dim == Expr::Const(1) {
                lhs_node = lhs_node.expand(i, rhs_dim.clone());
            } else if *rhs_dim == Expr::Const(1) {
                rhs_node = rhs_node.expand(i, lhs_dim.clone());
            } else {
                panic!(
                    "Cannot broadcast shapes: {:?} and {:?} at dimension {}",
                    lhs_shape, rhs_shape, i
                );
            }
        }
    }

    (lhs_node, rhs_node)
}

/// Common implementation for binary operations with broadcasting support.
fn binary_op<F>(lhs: &GraphNode, rhs: &GraphNode, combine: F) -> GraphNode
where
    F: FnOnce(AstNode, AstNode) -> AstNode,
{
    // Apply broadcasting
    let (lhs_bc, rhs_bc) = broadcast(lhs, rhs);

    // Inputs are referenced as Wildcard("0") and Wildcard("1")
    let lhs_ref = AstNode::Wildcard("0".to_string());
    let rhs_ref = AstNode::Wildcard("1".to_string());

    let combined = combine(lhs_ref, rhs_ref);

    // Result is a new tensor with contiguous layout
    // (the broadcast view only affects how inputs are read, not the output)
    let result_view = View::contiguous(lhs_bc.shape());

    GraphNode::new(
        vec![lhs_bc.clone(), rhs_bc.clone()],
        result_view,
        GraphOp::MapReduce {
            map: combined,
            reduce: None,
        },
        lhs_bc.dtype().clone(), // TODO: type promotion
        None,
    )
}

// ============================================================================
// Binary Operation Implementations
// ============================================================================

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $ast_variant:ident) => {
        impl $trait for GraphNode {
            type Output = GraphNode;

            fn $method(self, rhs: GraphNode) -> GraphNode {
                binary_op(&self, &rhs, |a, b| {
                    AstNode::$ast_variant(Box::new(a), Box::new(b))
                })
            }
        }

        impl $trait<&GraphNode> for GraphNode {
            type Output = GraphNode;

            fn $method(self, rhs: &GraphNode) -> GraphNode {
                binary_op(&self, rhs, |a, b| {
                    AstNode::$ast_variant(Box::new(a), Box::new(b))
                })
            }
        }

        impl $trait<GraphNode> for &GraphNode {
            type Output = GraphNode;

            fn $method(self, rhs: GraphNode) -> GraphNode {
                binary_op(self, &rhs, |a, b| {
                    AstNode::$ast_variant(Box::new(a), Box::new(b))
                })
            }
        }

        impl $trait<&GraphNode> for &GraphNode {
            type Output = GraphNode;

            fn $method(self, rhs: &GraphNode) -> GraphNode {
                binary_op(self, rhs, |a, b| {
                    AstNode::$ast_variant(Box::new(a), Box::new(b))
                })
            }
        }
    };
}

impl_binary_op!(Add, add, Add);
impl_binary_op!(Mul, mul, Mul);

// Sub is implemented as Add with negation
impl Sub for GraphNode {
    type Output = GraphNode;

    fn sub(self, rhs: GraphNode) -> GraphNode {
        binary_op(&self, &rhs, |a, b| {
            // a - b = a + (-b)
            AstNode::Add(
                Box::new(a),
                Box::new(AstNode::Mul(
                    Box::new(b),
                    Box::new(AstNode::Const((-1.0f32).into())),
                )),
            )
        })
    }
}

impl Sub<&GraphNode> for GraphNode {
    type Output = GraphNode;

    fn sub(self, rhs: &GraphNode) -> GraphNode {
        (&self).sub(rhs)
    }
}

impl Sub<GraphNode> for &GraphNode {
    type Output = GraphNode;

    fn sub(self, rhs: GraphNode) -> GraphNode {
        self.sub(&rhs)
    }
}

impl Sub<&GraphNode> for &GraphNode {
    type Output = GraphNode;

    fn sub(self, rhs: &GraphNode) -> GraphNode {
        binary_op(self, rhs, |a, b| {
            AstNode::Add(
                Box::new(a),
                Box::new(AstNode::Mul(
                    Box::new(b),
                    Box::new(AstNode::Const((-1.0f32).into())),
                )),
            )
        })
    }
}

// Div is implemented as Mul with reciprocal
impl Div for GraphNode {
    type Output = GraphNode;

    fn div(self, rhs: GraphNode) -> GraphNode {
        binary_op(&self, &rhs, |a, b| {
            // a / b = a * (1/b)
            AstNode::Mul(Box::new(a), Box::new(AstNode::Recip(Box::new(b))))
        })
    }
}

impl Div<&GraphNode> for GraphNode {
    type Output = GraphNode;

    fn div(self, rhs: &GraphNode) -> GraphNode {
        (&self).div(rhs)
    }
}

impl Div<GraphNode> for &GraphNode {
    type Output = GraphNode;

    fn div(self, rhs: GraphNode) -> GraphNode {
        self.div(&rhs)
    }
}

impl Div<&GraphNode> for &GraphNode {
    type Output = GraphNode;

    fn div(self, rhs: &GraphNode) -> GraphNode {
        binary_op(self, rhs, |a, b| {
            AstNode::Mul(Box::new(a), Box::new(AstNode::Recip(Box::new(b))))
        })
    }
}

// ============================================================================
// Unary Operations
// ============================================================================

impl Neg for GraphNode {
    type Output = GraphNode;

    fn neg(self) -> GraphNode {
        self.unary_op(|x| AstNode::Mul(Box::new(x), Box::new(AstNode::Const((-1.0f32).into()))))
    }
}

impl Neg for &GraphNode {
    type Output = GraphNode;

    fn neg(self) -> GraphNode {
        self.unary_op(|x| AstNode::Mul(Box::new(x), Box::new(AstNode::Const((-1.0f32).into()))))
    }
}

impl GraphNode {
    /// Apply a unary operation
    pub(crate) fn unary_op<F>(&self, transform: F) -> GraphNode
    where
        F: FnOnce(AstNode) -> AstNode,
    {
        let input_ref = AstNode::Wildcard("0".to_string());
        let transformed = transform(input_ref);

        GraphNode::new(
            vec![self.clone()],
            self.view().clone(),
            GraphOp::MapReduce {
                map: transformed,
                reduce: None,
            },
            self.dtype().clone(),
            None,
        )
    }

    // ========================================================================
    // Math Functions
    // ========================================================================

    /// Square root
    pub fn sqrt(&self) -> GraphNode {
        self.unary_op(|x| AstNode::Sqrt(Box::new(x)))
    }

    /// Reciprocal (1/x)
    pub fn recip(&self) -> GraphNode {
        self.unary_op(|x| AstNode::Recip(Box::new(x)))
    }

    /// Base-2 logarithm
    pub fn log2(&self) -> GraphNode {
        self.unary_op(|x| AstNode::Log2(Box::new(x)))
    }

    /// Base-2 exponential
    pub fn exp2(&self) -> GraphNode {
        self.unary_op(|x| AstNode::Exp2(Box::new(x)))
    }

    /// Sine
    pub fn sin(&self) -> GraphNode {
        self.unary_op(|x| AstNode::Sin(Box::new(x)))
    }

    /// Floor
    pub fn floor(&self) -> GraphNode {
        self.unary_op(|x| AstNode::Floor(Box::new(x)))
    }

    // ========================================================================
    // Derived Math Functions
    // ========================================================================

    /// Natural logarithm (ln)
    pub fn ln(&self) -> GraphNode {
        // ln(x) = log2(x) / log2(e)
        let log2_e = std::f32::consts::LOG2_E;
        self.unary_op(|x| {
            AstNode::Mul(
                Box::new(AstNode::Log2(Box::new(x))),
                Box::new(AstNode::Recip(Box::new(AstNode::Const(log2_e.into())))),
            )
        })
    }

    /// Natural exponential (e^x)
    pub fn exp(&self) -> GraphNode {
        // exp(x) = exp2(x * log2(e))
        let log2_e = std::f32::consts::LOG2_E;
        self.unary_op(|x| {
            AstNode::Exp2(Box::new(AstNode::Mul(
                Box::new(x),
                Box::new(AstNode::Const(log2_e.into())),
            )))
        })
    }

    /// Cosine
    pub fn cos(&self) -> GraphNode {
        // cos(x) = sin(x + pi/2)
        let half_pi = std::f32::consts::FRAC_PI_2;
        self.unary_op(|x| {
            AstNode::Sin(Box::new(AstNode::Add(
                Box::new(x),
                Box::new(AstNode::Const(half_pi.into())),
            )))
        })
    }

    /// Absolute value
    pub fn abs(&self) -> GraphNode {
        // abs(x) = x * sign(x) where sign(x) = select(x < 0, -1, 1)
        self.unary_op(|x| {
            let sign = AstNode::Select {
                cond: Box::new(AstNode::Lt(
                    Box::new(x.clone()),
                    Box::new(AstNode::Const(0.0f32.into())),
                )),
                then_val: Box::new(AstNode::Const((-1.0f32).into())),
                else_val: Box::new(AstNode::Const(1.0f32.into())),
            };
            AstNode::Mul(Box::new(x), Box::new(sign))
        })
    }

    // ========================================================================
    // Comparison Operations (return Bool dtype)
    // ========================================================================

    /// Less than comparison
    pub fn lt(&self, other: &GraphNode) -> GraphNode {
        let result = binary_op(self, other, |a, b| AstNode::Lt(Box::new(a), Box::new(b)));
        // Change dtype to Bool
        GraphNode::new(
            result.sources().to_vec(),
            result.view().clone(),
            result.op().clone(),
            DType::Bool,
            None,
        )
    }

    /// Greater than comparison
    pub fn gt(&self, other: &GraphNode) -> GraphNode {
        other.lt(self)
    }

    /// Less than or equal comparison
    pub fn le(&self, other: &GraphNode) -> GraphNode {
        // a <= b ⟺ !(b < a)
        other.lt(self).logical_not()
    }

    /// Greater than or equal comparison
    pub fn ge(&self, other: &GraphNode) -> GraphNode {
        // a >= b ⟺ !(a < b)
        self.lt(other).logical_not()
    }

    /// Equal comparison
    pub fn eq_node(&self, other: &GraphNode) -> GraphNode {
        // a == b ⟺ !(a < b) && !(b < a)
        self.le(other).logical_and(&other.le(self))
    }

    /// Not equal comparison
    pub fn ne_node(&self, other: &GraphNode) -> GraphNode {
        self.eq_node(other).logical_not()
    }

    // ========================================================================
    // Logical Operations
    // ========================================================================

    /// Logical AND
    pub fn logical_and(&self, other: &GraphNode) -> GraphNode {
        binary_op(self, other, |a, b| AstNode::And(Box::new(a), Box::new(b)))
    }

    /// Logical NOT
    pub fn logical_not(&self) -> GraphNode {
        self.unary_op(|x| AstNode::Not(Box::new(x)))
    }

    /// Logical OR
    pub fn logical_or(&self, other: &GraphNode) -> GraphNode {
        // a || b = !(!a && !b)
        self.logical_not()
            .logical_and(&other.logical_not())
            .logical_not()
    }

    // ========================================================================
    // Conditional Operations
    // ========================================================================

    /// Conditional selection: where(cond, self, other)
    ///
    /// Returns `self` where `cond` is true, `other` where false.
    pub fn where_cond(&self, cond: &GraphNode, other: &GraphNode) -> GraphNode {
        // Three-input operation
        let cond_ref = AstNode::Wildcard("0".to_string());
        let then_ref = AstNode::Wildcard("1".to_string());
        let else_ref = AstNode::Wildcard("2".to_string());

        let select = AstNode::Select {
            cond: Box::new(cond_ref),
            then_val: Box::new(then_ref),
            else_val: Box::new(else_ref),
        };

        GraphNode::new(
            vec![cond.clone(), self.clone(), other.clone()],
            self.view().clone(),
            GraphOp::MapReduce {
                map: select,
                reduce: None,
            },
            self.dtype().clone(),
            None,
        )
    }

    /// Element-wise maximum
    pub fn maximum(&self, other: &GraphNode) -> GraphNode {
        binary_op(self, other, |a, b| AstNode::Max(Box::new(a), Box::new(b)))
    }

    /// Element-wise minimum
    pub fn minimum(&self, other: &GraphNode) -> GraphNode {
        // min(a, b) = select(a < b, a, b)
        binary_op(self, other, |a, b| AstNode::Select {
            cond: Box::new(AstNode::Lt(Box::new(a.clone()), Box::new(b.clone()))),
            then_val: Box::new(a),
            else_val: Box::new(b),
        })
    }

    /// Clamp values to a range
    pub fn clamp(&self, min_val: &GraphNode, max_val: &GraphNode) -> GraphNode {
        self.maximum(min_val).minimum(max_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::input;
    use crate::graph::shape::Expr;

    fn make_test_node() -> GraphNode {
        input(vec![Expr::Const(32), Expr::Const(64)], DType::F32)
    }

    #[test]
    fn test_add() {
        let a = make_test_node();
        let b = make_test_node();
        let c = &a + &b;

        assert_eq!(c.sources().len(), 2);
        assert_eq!(c.shape(), a.shape());
    }

    #[test]
    fn test_mul() {
        let a = make_test_node();
        let b = make_test_node();
        let c = &a * &b;

        assert_eq!(c.sources().len(), 2);
    }

    #[test]
    fn test_sub() {
        let a = make_test_node();
        let b = make_test_node();
        let c = &a - &b;

        assert_eq!(c.sources().len(), 2);
    }

    #[test]
    fn test_div() {
        let a = make_test_node();
        let b = make_test_node();
        let c = &a / &b;

        assert_eq!(c.sources().len(), 2);
    }

    #[test]
    fn test_neg() {
        let a = make_test_node();
        let b = -&a;

        assert_eq!(b.sources().len(), 1);
    }

    #[test]
    fn test_sqrt() {
        let a = make_test_node();
        let b = a.sqrt();

        assert_eq!(b.sources().len(), 1);
    }

    #[test]
    fn test_chained_ops() {
        let a = make_test_node();
        let b = make_test_node();
        let c = (&a + &b).sqrt().sum(1);

        // c should have reduction
        match c.op() {
            GraphOp::MapReduce { reduce, .. } => {
                assert!(reduce.is_some());
            }
            _ => panic!("Expected MapReduce"),
        }
    }

    #[test]
    fn test_comparison() {
        let a = make_test_node();
        let b = make_test_node();
        let c = a.lt(&b);

        assert_eq!(c.dtype(), &DType::Bool);
    }

    #[test]
    fn test_where_cond() {
        let cond = make_test_node();
        let a = make_test_node();
        let b = make_test_node();
        let c = a.where_cond(&cond, &b);

        assert_eq!(c.sources().len(), 3);
    }
}

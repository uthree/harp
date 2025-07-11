//! # Simplify Module
//!
//! Provides a collection of `Rewriter`s for simplifying computation graphs.
//! These rewriters can be used individually or fused together to apply a
//! broad range of algebraic and constant-folding optimizations.
//!
//! ## Rewriters
//!
//! - `expansion_rewriter`: Expands high-level functions into simpler operations.
//! - `algebraic_rewriter`: Applies algebraic identities to simplify expressions.
//! - `constant_folding_rewriter`: Simplifies expressions involving only constants.
//! - `default_rewriter`: A fused rewriter that combines all available simplifications.

use crate::node::{Node, NodeData, capture, constant, cos, exp2, sin, tan};
use crate::op::{Const, Exp2, Log2, OpAdd, OpMul, Recip, Sin, Sqrt};
use crate::pattern::{RewriteRule, Rewriter};
use crate::rewrite_rule;
use std::sync::Arc;

/// Returns a `Rewriter` for expanding high-level functions into simpler operations.
pub fn expansion_rewriter() -> Rewriter {
    let rules = vec![
        rewrite_rule!(let x = capture("x"); crate::node::exp(x.clone()) => exp2(x * constant(std::f32::consts::LOG2_E))),
        rewrite_rule!(let x = capture("x"); cos(x.clone()) => sin(x + constant(std::f32::consts::FRAC_PI_2))),
        rewrite_rule!(let x = capture("x"); tan(x.clone()) => sin(x.clone()) * crate::node::recip(cos(x))),
    ];
    Rewriter::new("expansion", rules)
}

/// Returns a `Rewriter` for algebraic simplifications (e.g., identity, annihilator).
pub fn algebraic_rewriter() -> Rewriter {
    let rules = vec![
        rewrite_rule!(let x = capture("x"); x.clone() + constant(0.0f32) => x),
        rewrite_rule!(let x = capture("x"); constant(0.0f32) + x.clone() => x),
        rewrite_rule!(let x = capture("x"); x.clone() * constant(1.0f32) => x),
        rewrite_rule!(let x = capture("x"); constant(1.0f32) * x.clone() => x),
        rewrite_rule!(let x = capture("x"); x.clone() * constant(0.0f32) => constant(0.0f32)),
        rewrite_rule!(let x = capture("x"); constant(0.0f32) * x.clone() => constant(0.0f32)),
    ];
    Rewriter::new("algebraic", rules)
}

/// Returns a `Rewriter` for constant folding.
pub fn constant_folding_rewriter() -> Rewriter {
    let rules = vec![
        RewriteRule::new_fn(
            Node::from(Arc::new(NodeData {
                op: Box::new(OpAdd),
                src: vec![capture("a"), capture("b")],
            })),
            |_, captures| {
                let a = captures
                    .get("a")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                let b = captures
                    .get("b")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                Some(constant(a + b))
            },
        ),
        RewriteRule::new_fn(
            Node::from(Arc::new(NodeData {
                op: Box::new(OpMul),
                src: vec![capture("a"), capture("b")],
            })),
            |_, captures| {
                let a = captures
                    .get("a")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                let b = captures
                    .get("b")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                Some(constant(a * b))
            },
        ),
        RewriteRule::new_fn(
            Node::from(Arc::new(NodeData {
                op: Box::new(Exp2),
                src: vec![capture("x")],
            })),
            |_, captures| {
                let x = captures
                    .get("x")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                Some(constant(x.exp2()))
            },
        ),
        RewriteRule::new_fn(
            Node::from(Arc::new(NodeData {
                op: Box::new(Log2),
                src: vec![capture("x")],
            })),
            |_, captures| {
                let x = captures
                    .get("x")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                Some(constant(x.log2()))
            },
        ),
        RewriteRule::new_fn(
            Node::from(Arc::new(NodeData {
                op: Box::new(Sqrt),
                src: vec![capture("x")],
            })),
            |_, captures| {
                let x = captures
                    .get("x")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                Some(constant(x.sqrt()))
            },
        ),
        RewriteRule::new_fn(
            Node::from(Arc::new(NodeData {
                op: Box::new(Recip),
                src: vec![capture("x")],
            })),
            |_, captures| {
                let x = captures
                    .get("x")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                Some(constant(x.recip()))
            },
        ),
        RewriteRule::new_fn(
            Node::from(Arc::new(NodeData {
                op: Box::new(Sin),
                src: vec![capture("x")],
            })),
            |_, captures| {
                let x = captures
                    .get("x")?
                    .op()
                    .as_any()
                    .downcast_ref::<Const>()?
                    .0
                    .as_any()
                    .downcast_ref::<f32>()?;
                Some(constant(x.sin()))
            },
        ),
    ];
    Rewriter::new("constant_folding", rules)
}

/// Returns a `Rewriter` with a default set of simplification rules.
pub fn default_rewriter() -> Rewriter {
    constant_folding_rewriter() + expansion_rewriter() + algebraic_rewriter()
}

/// Simplifies a `Node` graph by repeatedly applying simplification rewriters
/// until a fixed point is reached.
pub fn simplify(node: Node) -> Node {
    let rewriter = default_rewriter();
    let mut current_node = node;
    loop {
        let last_node = current_node.clone();
        current_node = rewriter.rewrite(current_node);
        if current_node == last_node {
            break;
        }
    }
    current_node
}

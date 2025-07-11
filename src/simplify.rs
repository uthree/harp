//! # Simplify Module
//!
//! Provides a collection of `Rewriter`s for simplifying computation graphs.
//! These rewriters can be used individually or fused together to apply a
//! broad range of algebraic and constant-folding optimizations.
//!
//! ## Rewriters
//!
//! - `constant_folding_rewriter`: Simplifies expressions involving only constants
//!   (e.g., `2.0 + 3.0` becomes `5.0`).
//! - `algebraic_simplification_rewriter`: Applies algebraic identities to simplify
//!   expressions (e.g., `x + 0` becomes `x`, `x * 1` becomes `x`).
//! - `default_rewriter`: A fused rewriter that combines all available simplifications.

use crate::node::{Node, NodeData, capture, constant, exp2, sin};
use crate::op::{Const, Exp2, Log2, OpAdd, OpMul, Recip, Sin, Sqrt};
use crate::pattern::{RewriteRule, Rewriter};
use crate::rewrite_rule;
use std::sync::Arc;

/// Returns a `Rewriter` with a default set of simplification rules.
pub fn default_rewriter() -> Rewriter {
    let mut rules = Vec::new();

    // --- Function Expansion ---
    rules.push(rewrite_rule!(let x = capture("x");
        crate::node::exp(x.clone()) => exp2(x * constant(std::f32::consts::LOG2_E))
    ));
    rules.push(rewrite_rule!(let x = capture("x");
        crate::node::cos(x.clone()) => sin(x + constant(std::f32::consts::FRAC_PI_2))
    ));
    rules.push(rewrite_rule!(let x = capture("x");
        crate::node::tan(x.clone()) => sin(x.clone()) / sin(x + constant(std::f32::consts::FRAC_PI_2))
    ));

    // --- Constant Folding ---
    rules.push(RewriteRule::new_fn(
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
    ));
    rules.push(RewriteRule::new_fn(
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
    ));
    rules.push(RewriteRule::new_fn(
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
    ));
    rules.push(RewriteRule::new_fn(
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
    ));
    rules.push(RewriteRule::new_fn(
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
    ));
    rules.push(RewriteRule::new_fn(
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
    ));
    rules.push(RewriteRule::new_fn(
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
    ));

    // --- Algebraic Simplifications ---
    rules.push(rewrite_rule!(let x = capture("x"); x.clone() + constant(0.0f32) => x));
    rules.push(rewrite_rule!(let x = capture("x"); constant(0.0f32) + x.clone() => x));
    rules.push(rewrite_rule!(let x = capture("x"); x.clone() * constant(1.0f32) => x));
    rules.push(rewrite_rule!(let x = capture("x"); constant(1.0f32) * x.clone() => x));
    rules.push(
        rewrite_rule!(let x = capture("x"); x.clone() * constant(0.0f32) => constant(0.0f32)),
    );
    rules.push(
        rewrite_rule!(let x = capture("x"); constant(0.0f32) * x.clone() => constant(0.0f32)),
    );

    Rewriter::new("default", rules)
}

/// Simplifies a `Node` graph using the default `Rewriter`.
pub fn simplify(node: Node) -> Node {
    let rewriter = default_rewriter();
    rewriter.rewrite(node)
}

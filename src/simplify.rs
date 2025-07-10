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

use crate::node::{Const, constant, exp2, log2, recip, sin, sqrt};
use crate::pattern::Rewriter;
use crate::rewriter;

/// Creates a `Rewriter` for constant folding.
///
/// This rewriter simplifies expressions where all operands are constants.
/// For example, `constant(2.0) + constant(3.0)` will be rewritten to `constant(5.0)`.
pub fn constant_folding_rewriter() -> Rewriter {
    rewriter!([
        (
            let x = capture("x"), y = capture("y")
            => x + y
            => |x, y| {
                let x_op = x.op().as_any().downcast_ref::<Const>()?;
                let y_op = y.op().as_any().downcast_ref::<Const>()?;
                if let (Some(x_val), Some(y_val)) = (x_op.0.as_any().downcast_ref::<f32>(), y_op.0.as_any().downcast_ref::<f32>()) {
                    return Some(constant(x_val + y_val));
                }
                if let (Some(x_val), Some(y_val)) = (x_op.0.as_any().downcast_ref::<f64>(), y_op.0.as_any().downcast_ref::<f64>()) {
                    return Some(constant(x_val + y_val));
                }
                None
            }
        ),
        (
            let x = capture("x"), y = capture("y")
            => x * y
            => |x, y| {
                let x_op = x.op().as_any().downcast_ref::<Const>()?;
                let y_op = y.op().as_any().downcast_ref::<Const>()?;
                if let (Some(x_val), Some(y_val)) = (x_op.0.as_any().downcast_ref::<f32>(), y_op.0.as_any().downcast_ref::<f32>()) {
                    return Some(constant(x_val * y_val));
                }
                if let (Some(x_val), Some(y_val)) = (x_op.0.as_any().downcast_ref::<f64>(), y_op.0.as_any().downcast_ref::<f64>()) {
                    return Some(constant(x_val * y_val));
                }
                None
            }
        ),
        (
            let x = capture("x")
            => recip(x)
            => |x| {
                if let Some(const_op) = x.op().as_any().downcast_ref::<Const>() {
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f32>() {
                        return Some(constant(1.0 / val));
                    }
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f64>() {
                        return Some(constant(1.0 / val));
                    }
                }
                None
            }
        ),
        (
            let x = capture("x")
            => sin(x)
            => |x| {
                if let Some(const_op) = x.op().as_any().downcast_ref::<Const>() {
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f32>() {
                        return Some(constant(val.sin()));
                    }
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f64>() {
                        return Some(constant(val.sin()));
                    }
                }
                None
            }
        ),
        (
            let x = capture("x")
            => exp2(x)
            => |x| {
                if let Some(const_op) = x.op().as_any().downcast_ref::<Const>() {
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f32>() {
                        return Some(constant(val.exp2()));
                    }
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f64>() {
                        return Some(constant(val.exp2()));
                    }
                }
                None
            }
        ),
        (
            let x = capture("x")
            => log2(x)
            => |x| {
                if let Some(const_op) = x.op().as_any().downcast_ref::<Const>() {
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f32>() {
                        return Some(constant(val.log2()));
                    }
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f64>() {
                        return Some(constant(val.log2()));
                    }
                }
                None
            }
        ),
        (
            let x = capture("x")
            => sqrt(x)
            => |x| {
                if let Some(const_op) = x.op().as_any().downcast_ref::<Const>() {
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f32>() {
                        return Some(constant(val.sqrt()));
                    }
                    if let Some(val) = const_op.0.as_any().downcast_ref::<f64>() {
                        return Some(constant(val.sqrt()));
                    }
                }
                None
            }
        )
    ])
}

/// Creates a `Rewriter` for algebraic simplifications.
///
/// This rewriter applies rules like:
/// - `x + 0 -> x`
/// - `x * 1 -> x`
/// - `x * 0 -> 0`
pub fn algebraic_simplification_rewriter() -> Rewriter {
    rewriter!([
        (let x = capture("x") => x.clone() + constant(0.0f32) => |x| Some(x)),
        (let x = capture("x") => constant(0.0f32) + x.clone() => |x| Some(x)),
        (let x = capture("x") => x.clone() * constant(1.0f32) => |x| Some(x)),
        (let x = capture("x") => constant(1.0f32) * x.clone() => |x| Some(x)),
        (let x = capture("x") => x * constant(0.0f32) => |_x| Some(constant(0.0f32))),
        (let x = capture("x") => constant(0.0f32) * x => |_x| Some(constant(0.0f32)))
    ])
}

/// Creates a default `Rewriter` by fusing all available simplification rewriters.
pub fn default_rewriter() -> Rewriter {
    constant_folding_rewriter().fused(algebraic_simplification_rewriter())
}

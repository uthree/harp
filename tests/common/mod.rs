use harp::node::{constant, exp2, log2, recip, sin, sqrt};
use harp::pattern::Rewriter;
#[allow(unused_imports)]
use harp::{capture, rewriter};

#[allow(unused_imports)]
use harp::node::Node;

pub fn eval_rules() -> Rewriter {
    rewriter!([
        (
            let x = capture("x"), y = capture("y")
            => x + y
            => |x, y| {
                let x_op = x.op().as_any().downcast_ref::<harp::node::Const>()?;
                let y_op = y.op().as_any().downcast_ref::<harp::node::Const>()?;
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
                let x_op = x.op().as_any().downcast_ref::<harp::node::Const>()?;
                let y_op = y.op().as_any().downcast_ref::<harp::node::Const>()?;
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
                if let Some(const_op) = x.op().as_any().downcast_ref::<harp::node::Const>() {
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
                if let Some(const_op) = x.op().as_any().downcast_ref::<harp::node::Const>() {
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
                if let Some(const_op) = x.op().as_any().downcast_ref::<harp::node::Const>() {
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
                if let Some(const_op) = x.op().as_any().downcast_ref::<harp::node::Const>() {
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
                if let Some(const_op) = x.op().as_any().downcast_ref::<harp::node::Const>() {
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

#[allow(unused_imports)]
use harp::capture;
use harp::node::{constant, exp2, log2, sin, sqrt};
use harp::pattern::Rewriter;
use harp::rewriter;

fn eval_rules() -> Rewriter {
    rewriter!([
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

#[test]
fn test_eval_sin() {
    let rewriter = eval_rules();
    let graph = sin(constant(std::f32::consts::PI / 2.0));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::node::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - 1.0).abs() < 1e-6);
}

#[test]
fn test_eval_exp2() {
    let rewriter = eval_rules();
    let graph = exp2(constant(2.0f32));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::node::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - 4.0).abs() < 1e-6);
}

#[test]
fn test_eval_log2() {
    let rewriter = eval_rules();
    let graph = log2(constant(8.0f32));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::node::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - 3.0).abs() < 1e-6);
}

#[test]
fn test_eval_sqrt() {
    let rewriter = eval_rules();
    let graph = sqrt(constant(16.0f32));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::node::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<f32>()
        .unwrap();
    assert!((result - 4.0).abs() < 1e-6);
}

use harp::node::{cast, constant};
use harp::op::Const;
use harp::pattern::Rewriter;
use harp::{capture, rewriter};

fn cast_rules() -> Rewriter {
    rewriter!("cast_rules", [
        (
            let x = capture("x")
            => cast::<i32>(x)
            => |_node, x| {
                let const_op = x.op().as_any().downcast_ref::<Const>()?;
                if let Some(val) = const_op.0.as_any().downcast_ref::<f32>() {
                    return Some(constant(*val as i32));
                }
                None
            }
        )
    ])
}

#[test]
fn test_cast_f32_to_i32() {
    let rewriter = cast_rules();
    let graph = cast::<i32>(constant(12.34f32));
    let rewritten_graph = rewriter.rewrite(graph);
    let result = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<harp::op::Const>()
        .unwrap()
        .0
        .as_any()
        .downcast_ref::<i32>()
        .unwrap();
    assert_eq!(*result, 12);
}

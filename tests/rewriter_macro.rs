use harp::node::{self, Node};
use harp::op::Const;
use harp::pattern::Rewriter;
use harp::{capture, rewriter};

#[test]
#[ignore]
fn test_rewriter_macro() {
    let rewriter = rewriter!([
        (
            let x = capture("x")
            => x * Node::from(1.0f32)
            => |x| Some(x)
        )
    ]);

    let a = self::node::constant(10.0f32);
    let b = self::node::constant(5.0f32);
    let graph = (a + b) * 1.0;

    let rewritten_graph = rewriter.rewrite(graph);
    let const_op = rewritten_graph
        .op()
        .as_any()
        .downcast_ref::<Const>()
        .unwrap();
    let val = const_op.0.as_any().downcast_ref::<f32>().unwrap();
    assert_eq!(*val, 15.0);
}


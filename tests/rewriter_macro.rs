use harp::node::{self, Const, Node};
use harp::rewriter;

#[test]
#[ignore]
fn test_rewriter_macro() {
    // Define a rewriter with multiple rules using the macro
    let symbolic_rewriter = rewriter!([
        // Rule 1: x + 0 -> x
        (
            let x = capture("x")
            => x + Node::from(0.0f32)
            => |x| Some(x)
        ),
        // Rule 2: x * 1 -> x
        (
            let x = capture("x")
            => x * Node::from(1.0f32)
            => |x| Some(x)
        ),
        // Rule 3: Constant Folding for addition
        (
            let x = capture("x"), y = capture("y")
            => x + y
            => |x, y| {
                if let (Some(const_x), Some(const_y)) = (
                    x.op().as_any().downcast_ref::<Const>(),
                    y.op().as_any().downcast_ref::<Const>(),
                ) {
                    if let (Some(val_x), Some(val_y)) = (
                        const_x.0.as_any().downcast_ref::<f32>(),
                        const_y.0.as_any().downcast_ref::<f32>(),
                    ) {
                        return Some(node::constant(val_x + val_y));
                    }
                }
                None
            }
        )
    ]);

    // Test case 1: 2.0 + 0.0 => 2.0
    let graph1 = node::constant(2.0f32) + Node::from(0.0f32);
    let rewritten1 = symbolic_rewriter.rewrite(graph1);
    assert_eq!(rewritten1, node::constant(2.0f32));

    // Test case 2: (2.0 * 1.0) + 3.0 => 2.0 + 3.0 => 5.0
    let graph2 = (node::constant(2.0f32) * Node::from(1.0f32)) + node::constant(3.0f32);
    let rewritten2 = symbolic_rewriter.rewrite(graph2);
    assert_eq!(rewritten2, node::constant(5.0f32));

    // Test case 3: a + 5.0 (no change)
    let a = node::recip(node::constant(10.0f32)); // A non-constant node
    let graph3 = a.clone() + node::constant(5.0f32);
    let rewritten3 = symbolic_rewriter.rewrite(graph3.clone());
    assert_eq!(rewritten3, graph3);
}

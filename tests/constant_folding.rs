use harp::node::{self, Const};
use harp::pattern::{RewriteRule, Rewriter};

#[test]
fn test_constant_folding_add() {
    // Define the rewrite rule: x + y -> Const(x_val + y_val)
    let rule = RewriteRule::new_fn(
        // Search for "x + y"
        {
            let x = node::capture("x");
            let y = node::capture("y");
            x + y
        },
        // Rewrite logic
        |_node, captures| {
            let x_node = captures.get("x")?;
            let y_node = captures.get("y")?;

            let x_const = x_node.op().as_any().downcast_ref::<node::Const>()?;
            let y_const = y_node.op().as_any().downcast_ref::<node::Const>()?;

            let x_val = x_const.0.as_any().downcast_ref::<f32>()?;
            let y_val = y_const.0.as_any().downcast_ref::<f32>()?;

            Some(node::constant(x_val + y_val))
        },
    );

    // 1. Test with two constants: 2.0 + 3.0 => 5.0
    let graph1 = node::constant(2.0f32) + node::constant(3.0f32);
    let rewriter = Rewriter::new(vec![rule]);
    let rewritten_graph1 = rewriter.rewrite(graph1);
    assert_eq!(rewritten_graph1, node::constant(5.0f32));

    // 2. Test with one constant and one variable: a + 3.0 => should not change
    let a = node::capture("a"); // Using capture as a placeholder for a variable
    let graph2 = a.clone() + node::constant(3.0f32);
    // Re-define the rule for the second test
    let rule2 = RewriteRule::new_fn(
        {
            let x = node::capture("x");
            let y = node::capture("y");
            x + y
        },
        |_node, captures| {
            let x_node = captures.get("x")?;
            let y_node = captures.get("y")?;
            if let (Some(const_x), Some(const_y)) = (
                x_node.op().as_any().downcast_ref::<Const>(),
                y_node.op().as_any().downcast_ref::<Const>(),
            ) {
                if let (Some(val_x), Some(val_y)) = (
                    const_x.0.as_any().downcast_ref::<f32>(),
                    const_y.0.as_any().downcast_ref::<f32>(),
                ) {
                    return Some(node::constant(val_x + val_y));
                }
            }
            None
        },
    );
    let rewriter2 = Rewriter::new(vec![rule2]);
    let rewritten_graph2 = rewriter2.rewrite(graph2.clone());
    assert_eq!(rewritten_graph2, graph2);
}

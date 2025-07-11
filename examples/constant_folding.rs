use harp::node::{self, constant};
use harp::op::Const;
use harp::{capture, rewriter};

fn main() {
    // Initialize the logger
    env_logger::init();

    // Define a rewriter for constant folding.
    let constant_folder = rewriter!("constant_folder", [
        (
            let x = capture("x"), y = capture("y")
            => x + y
            => |_node, x, y| {
                let x_const = x.op().as_any().downcast_ref::<Const>()?;
                let y_const = y.op().as_any().downcast_ref::<Const>()?;
                let x_val = x_const.0.as_any().downcast_ref::<f32>()?;
                let y_val = y_const.0.as_any().downcast_ref::<f32>()?;
                Some(constant(x_val + y_val))
            }
        )
    ]);

    // Create a graph: 2.0 + 3.0
    let graph = constant(2.0f32) + constant(3.0f32);

    // Rewrite the graph
    let rewritten_graph = constant_folder.rewrite(graph);

    // The result should be a constant node with value 5.0
    println!("Rewritten graph: {:?}", rewritten_graph);
}
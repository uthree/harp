use harp::node::{self, Node};
use harp::op::{FusedOp, OpDiv, OpSub, Operator};
use rstest::rstest;
use std::any::Any;

// Define a new fused operator for testing
#[derive(Debug, Clone)]
struct FusedMulAdd;

impl Operator for FusedMulAdd {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &'static str {
        "FusedMulAdd"
    }
}

impl FusedOp for FusedMulAdd {
    fn fallback(&self, operands: &[Node]) -> Node {
        assert_eq!(operands.len(), 3, "FusedMulAdd expects 3 operands");
        // Decomposes into (a * b) + c
        (operands[0].clone() * operands[1].clone()) + operands[2].clone()
    }
}

#[test]
fn test_fused_op_fallback() {
    // Create operands
    let a = node::variable("a");
    let b = node::variable("b");
    let c = node::variable("c");

    // Create the fused operator node
    let fused_op = FusedMulAdd;
    let fused_node = Node::new(fused_op, vec![a.clone(), b.clone(), c.clone()]);

    // Get the FusedOp trait object
    let fused_op_trait = fused_node
        .op()
        .as_any()
        .downcast_ref::<FusedMulAdd>()
        .unwrap();

    // Call fallback
    let decomposed_node = fused_op_trait.fallback(fused_node.src());

    // Define the expected decomposed graph
    let expected_node = (a * b) + c;

    // Check if the decomposed graph is correct
    assert_eq!(decomposed_node, expected_node);
}

#[rstest]
#[case(OpSub, |a, b| a - b)]
#[case(OpDiv, |a, b| a / b)]
fn test_builtin_fused_ops<O, F>(#[case] op: O, #[case] expected_decomposed: F)
where
    O: Operator + FusedOp + 'static,
    F: Fn(Node, Node) -> Node,
{
    // Create operands
    let a = node::variable("a");
    let b = node::variable("b");

    // Create the fused operator node
    let fused_node = Node::new(op, vec![a.clone(), b.clone()]);

    // Get the FusedOp trait object
    let fused_op_trait = fused_node.op().as_any().downcast_ref::<O>().unwrap();

    // Call fallback
    let decomposed_node = fused_op_trait.fallback(fused_node.src());

    // Define the expected decomposed graph
    let expected_node = expected_decomposed(a, b);

    // Check if the decomposed graph is correct
    assert_eq!(decomposed_node, expected_node);
}

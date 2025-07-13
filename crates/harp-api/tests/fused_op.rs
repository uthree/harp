use harp_api::prelude::*;
use harp_ir::node::{self, Node};
use harp_ir::op::{FusedOp, OpDiv, OpRandn, OpSub, Operator};
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
    let fused_node = Node::new(Box::new(fused_op), vec![a.clone(), b.clone(), c.clone()]);

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
    let fused_node = Node::new(Box::new(op), vec![a.clone(), b.clone()]);

    // Get the FusedOp trait object
    let fused_op_trait = fused_node.op().as_any().downcast_ref::<O>().unwrap();

    // Call fallback
    let decomposed_node = fused_op_trait.fallback(fused_node.src());

    // Define the expected decomposed graph
    let expected_node = expected_decomposed(a, b);

    // Check if the decomposed graph is correct
    assert_eq!(decomposed_node, expected_node);
}

#[test]
fn test_oprandn_fallback_structure() {
    let op = OpRandn;
    let fallback_graph = op.fallback(&[]);

    // Check the structure of the Box-Muller transform graph
    // Z1 = sqrt(-2 * ln(U1)) * cos(2 * pi * U2)
    assert_eq!(fallback_graph.op().name(), "OpMul");
    assert_eq!(fallback_graph.src().len(), 2);

    // Check term1: sqrt(-2 * ln(U1))
    let term1 = &fallback_graph.src()[0];
    assert_eq!(term1.op().name(), "Sqrt");
    assert_eq!(term1.src().len(), 1);

    let sqrt_inner = &term1.src()[0];
    assert_eq!(sqrt_inner.op().name(), "OpMul");
    assert_eq!(sqrt_inner.src().len(), 2);
    assert_eq!(sqrt_inner.src()[0].op().name(), "Const"); // -2.0
    assert_eq!(sqrt_inner.src()[1].op().name(), "OpMul"); // ln(U1) = log2(U1) * LN_2

    let ln_u1 = &sqrt_inner.src()[1];
    assert_eq!(ln_u1.src()[0].op().name(), "Log2");
    assert_eq!(ln_u1.src()[0].src()[0].op().name(), "OpUniform"); // U1
    assert_eq!(ln_u1.src()[1].op().name(), "Const"); // LN_2

    // Check term2: cos(2 * pi * U2)
    let term2 = &fallback_graph.src()[1];
    assert_eq!(term2.op().name(), "Sin"); // cos(x) is implemented as sin(x + PI/2)
    assert_eq!(term2.src().len(), 1);

    let cos_inner = &term2.src()[0]; // x + PI/2
    assert_eq!(cos_inner.op().name(), "OpAdd");
    assert_eq!(cos_inner.src()[1].op().name(), "Const"); // PI/2

    let cos_arg = &cos_inner.src()[0]; // x = 2 * pi * U2
    assert_eq!(cos_arg.op().name(), "OpMul");
    assert_eq!(cos_arg.src()[0].op().name(), "Const"); // 2 * pi
    assert_eq!(cos_arg.src()[1].op().name(), "OpUniform"); // U2
}

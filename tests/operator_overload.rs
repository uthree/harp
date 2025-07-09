use harp::node::{self, Node, NodeRef, OpAdd, OpMul, Recip};
use std::sync::Arc;

#[test]
fn test_operator_overloads() {
    // Build the expression `a - b / c` using overloaded operators.
    let a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let c = node::constant(3.0f32);

    // a - b / c  => a + (-(b * recip(c)))
    let result_node = a.clone() - b.clone() / c.clone();

    // Manually construct the expected graph structure.
    // 1. recip(c)
    let recip_c = NodeRef::from(Arc::new(Node {
        op: Box::new(Recip),
        src: vec![c],
    }));
    // 2. b * recip(c)
    let b_div_c = NodeRef::from(Arc::new(Node {
        op: Box::new(OpMul),
        src: vec![b, recip_c],
    }));
    // 3. -(b * recip(c)) which is (b * recip(c)) * -1.0
    let neg_b_div_c = NodeRef::from(Arc::new(Node {
        op: Box::new(OpMul),
        src: vec![b_div_c, node::constant(-1.0f32)],
    }));
    // 4. a + (-(b * recip(c)))
    let expected_node = NodeRef::from(Arc::new(Node {
        op: Box::new(OpAdd),
        src: vec![a, neg_b_div_c],
    }));

    // Compare the generated node with the expected node.
    assert_eq!(*result_node, *expected_node);
}

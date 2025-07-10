use harp::node::{self};

#[test]
fn test_operator_overloads() {
    // Build the expression `a - b / c` using overloaded operators.
    let a = node::constant(1.0f32);
    let b = node::constant(2.0f32);
    let c = node::constant(3.0f32);

    // a - b / c  => a + (-(b * recip(c)))
    let result_node = a.clone() - b.clone() / c.clone();

    // Manually construct the expected graph structure.
    let b_div_c = b.clone() / c.clone();
    let neg_b_div_c = -b_div_c;
    let expected_node = a.clone() + neg_b_div_c;

    // Compare the generated node with the expected node.
    assert_eq!(result_node, expected_node);
}

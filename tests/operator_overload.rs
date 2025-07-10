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
    let recip_c = node::recip(c);
    let b_div_c = node::mul(b, recip_c);
    let neg_b_div_c = node::mul(b_div_c, node::constant(-1.0f32));
    let expected_node = node::add(a, neg_b_div_c);

    // Compare the generated node with the expected node.
    assert_eq!(result_node, expected_node);
}


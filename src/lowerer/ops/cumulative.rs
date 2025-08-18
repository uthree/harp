use crate::{
    ast::{AstNode, AstOp, DType},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_cumulative(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
    axis: &usize,
) -> AstNode {
    // Create an accumulator variable, initialized to the identity of the op.
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());

    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported cumulative operation: {:?}", op),
    };
    let declare_acc = AstNode::declare(&acc_name, node.dtype.clone(), Some(init_val));

    // Create an inner loop to accumulate values from the source.
    // The loop runs from 0 up to the current index along the cumulative axis.
    let cum_limit = indices[*axis].clone() + AstNode::from(1isize);
    let cidx_name = format!("cidx{}", lowerer.ridx_counter);
    lowerer.ridx_counter += 1;
    let cidx_var = AstNode::var(&cidx_name, DType::Isize);

    let mut inner_indices = indices.to_vec();
    inner_indices[*axis] = cidx_var;

    let value_to_accumulate = lowerer.lower_node_rec(&node.src[0], &mut inner_indices, inputs);

    let update_op = AstNode::_new(
        op.clone(),
        vec![acc_var.clone(), value_to_accumulate],
        node.dtype.clone(),
    );
    let assign_op = AstNode::assign(acc_var.clone(), update_op);

    let loop_node = AstNode::_new(
        AstOp::Range {
            counter: cidx_name,
            step: 1,
        },
        vec![cum_limit, assign_op],
        DType::Void,
    );

    // The final result is a block that declares the accumulator, runs the loop,
    // and then returns the final accumulator value.
    AstNode::_new(
        AstOp::Block,
        vec![declare_acc, loop_node, acc_var],
        node.dtype.clone(),
    )
}

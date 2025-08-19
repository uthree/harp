use crate::{
    ast::{AstNode, AstOp, DType},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_reduce(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
    axis: &usize,
) -> AstNode {
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());

    // Initialize accumulator.
    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported reduction operation: {:?}", op),
    };
    let declare_acc = AstNode::declare(&acc_name, node.dtype.clone(), Some(init_val));

    // Create reduction loop.
    let reduce_dim = node.src[0].shape()[*axis].clone();
    let ridx_name = format!("ridx{}", lowerer.ridx_counter);
    lowerer.ridx_counter += 1;
    let ridx_var = AstNode::var(&ridx_name, DType::Isize);

    let mut inner_indices = indices.to_vec();
    // Insert the reduction loop variable at the reduction axis.
    inner_indices.insert(*axis, ridx_var);

    let value_to_reduce = lowerer.lower_node_rec(&node.src[0], &mut inner_indices, inputs);

    let update_op = AstNode::_new(
        op.clone(),
        vec![acc_var.clone(), value_to_reduce],
        node.dtype.clone(),
    );
    let assign_op = AstNode::assign(acc_var.clone(), update_op);

    let loop_node = AstNode::_new(
        AstOp::Range {
            counter: ridx_name,
            step: 1,
        },
        vec![reduce_dim.into(), AstNode::block(vec![assign_op])],
        DType::Void,
    );

    // Return a block that declares, computes, and returns the accumulator.
    AstNode::_new(
        AstOp::Block,
        vec![declare_acc, loop_node, acc_var],
        node.dtype.clone(),
    )
}

use crate::{
    ast::{AstNode, AstOp},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_fused_elementwise(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    fused_ast: &AstNode,
) -> AstNode {
    replace_captures_rec(lowerer, node, indices, inputs, fused_ast)
}

pub fn lower_fused_elementwise_reduce(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    fused_ast: &AstNode,
    op: &AstOp,
    axes: &[usize],
) -> AstNode {
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());

    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported reduction operation: {:?}", op),
    };
    let declare_acc = AstNode::declare(&acc_name, node.dtype.clone(), Some(init_val));

    let mut inner_indices = indices.to_vec();
    let mut body = {
        let value_to_reduce =
            replace_captures_rec(lowerer, node, &mut inner_indices, inputs, fused_ast);
        let update_op = AstNode::_new(
            op.clone(),
            vec![acc_var.clone(), value_to_reduce],
            node.dtype.clone(),
        );
        AstNode::assign(acc_var.clone(), update_op)
    };

    for (i, axis) in axes.iter().enumerate().rev() {
        let reduce_dim = node.src[0].shape()[*axis].clone();
        let ridx_name = format!("ridx{}_{}", lowerer.ridx_counter, i);
        let ridx_var = AstNode::var(&ridx_name, node.dtype.clone());
        inner_indices.insert(*axis, ridx_var);

        body = AstNode::_new(
            AstOp::Range {
                counter: ridx_name,
                step: 1,
            },
            vec![reduce_dim.into(), AstNode::block(vec![body])],
            node.dtype.clone(),
        );
    }
    lowerer.ridx_counter += 1;

    let mut block_content = vec![declare_acc];
    block_content.push(body);
    block_content.push(acc_var);

    AstNode::_new(AstOp::Block, block_content, node.dtype.clone())
}

pub fn lower_fused_reduce(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
    axes: &[usize],
) -> AstNode {
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());

    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported reduction operation: {:?}", op),
    };
    let declare_acc = AstNode::declare(&acc_name, node.dtype.clone(), Some(init_val));

    let mut inner_indices = indices.to_vec();
    let mut loops = vec![];

    let mut body = {
        let value_to_reduce = lowerer.lower_node_rec(&node.src[0], &mut inner_indices, inputs);
        let update_op = AstNode::_new(
            op.clone(),
            vec![acc_var.clone(), value_to_reduce],
            node.dtype.clone(),
        );
        AstNode::assign(acc_var.clone(), update_op)
    };

    for (i, axis) in axes.iter().enumerate().rev() {
        let reduce_dim = node.src[0].shape()[*axis].clone();
        let ridx_name = format!("ridx{}_{}", lowerer.ridx_counter, i);
        let ridx_var = AstNode::var(&ridx_name, node.dtype.clone());
        inner_indices.insert(*axis, ridx_var);

        body = AstNode::_new(
            AstOp::Range {
                counter: ridx_name,
                step: 1,
            },
            vec![reduce_dim.into(), AstNode::block(vec![body])],
            node.dtype.clone(),
        );
    }
    loops.push(body);
    lowerer.ridx_counter += 1;

    let mut block_content = vec![declare_acc];
    block_content.extend(loops);
    block_content.push(acc_var);

    AstNode::_new(AstOp::Block, block_content, node.dtype.clone())
}

fn replace_captures_rec(
    lowerer: &mut Lowerer,
    graph_node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    fused_ast: &AstNode,
) -> AstNode {
    if let AstOp::Capture(n) = fused_ast.op {
        return lowerer.lower_node_rec(&graph_node.src[n], indices, inputs);
    }

    let new_srcs = fused_ast
        .src
        .iter()
        .map(|src| replace_captures_rec(lowerer, graph_node, indices, inputs, src))
        .collect();

    AstNode::_new(fused_ast.op.clone(), new_srcs, fused_ast.dtype.clone())
}

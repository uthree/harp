use crate::{
    ast::{AstNode, AstOp},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_elementwise(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
) -> AstNode {
    let srcs = node
        .src
        .iter()
        .map(|n| lowerer.lower_node_rec(n, indices, inputs))
        .collect();
    AstNode::_new(op.clone(), srcs, node.dtype.clone())
}

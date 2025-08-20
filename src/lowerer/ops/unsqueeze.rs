use crate::{
    ast::AstNode,
    graph::{GraphNode, GraphOp},
    lowerer::Lowerer,
};

pub fn lower_unsqueeze(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
) -> Vec<AstNode> {
    let src_node = &node.src[0];
    let axis = if let GraphOp::Unsqueeze(axis) = node.op {
        axis
    } else {
        unreachable!()
    };

    let mut src_indices = indices.to_vec();
    if axis < src_indices.len() {
        src_indices.remove(axis);
    }

    lowerer.lower_node_rec(src_node, &mut src_indices, inputs)
}

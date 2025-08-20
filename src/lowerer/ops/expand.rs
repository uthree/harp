use crate::{
    ast::AstNode,
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_expand(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
) -> Vec<AstNode> {
    let src_node = &node.src[0];
    let src_shape = src_node.shape();
    let new_shape = node.shape();

    let mut src_indices = indices.to_vec();

    for i in 0..src_shape.len() {
        if src_shape[i] != new_shape[i] {
            // The shape of the axis to be expanded must be 1.
            // This is checked when building the graph.
            // The index for the expanded axis is always 0.
            src_indices[i] = AstNode::from(0isize);
        }
    }

    lowerer.lower_node_rec(src_node, &mut src_indices, inputs, None)
}

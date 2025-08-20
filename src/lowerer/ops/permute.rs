use crate::{ast::AstNode, graph::GraphNode, lowerer::Lowerer};

pub fn lower_permute(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    axes: &[usize],
) -> Vec<AstNode> {
    // Reorder indices according to the permutation axes before recursing.
    let mut unpermuted_indices = vec![AstNode::from(0isize); indices.len()];
    for (i, &axis) in axes.iter().enumerate() {
        unpermuted_indices[axis] = indices[i].clone();
    }
    lowerer.lower_node_rec(&node.src[0], &mut unpermuted_indices, inputs)
}

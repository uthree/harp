use crate::{ast::AstNode, graph::GraphNode, lowerer::Lowerer};

pub fn lower_contiguous(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
) -> Vec<AstNode> {
    // A contiguous op is a hint for memory layout. In a fused kernel,
    // we "look through" it and directly compute from its source,
    // relying on the view of the source node to handle memory access correctly.
    lowerer.lower_node_rec(&node.src[0], indices, inputs)
}

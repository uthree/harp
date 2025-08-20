use crate::{
    ast::{AstNode, DType},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_input(
    _lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
) -> Vec<AstNode> {
    let input_idx = inputs
        .iter()
        .position(|n| n == node)
        .expect("Input node not found in graph inputs");
    let input_var = AstNode::var(
        &format!("input{}", input_idx),
        DType::Ptr(Box::new(node.dtype.clone())),
    );

    // Use the node's view to calculate the physical memory index from logical indices.
    let physical_index = node.view.to_physical_index_ast(indices);
    let ptr = AstNode::index(input_var, physical_index);
    vec![AstNode::load(ptr)]
}

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

fn replace_captures_rec(
    lowerer: &mut Lowerer,
    graph_node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    fused_ast: &AstNode,
) -> AstNode {
    if let AstOp::Capture(n) = fused_ast.op {
        // The captured node corresponds to the n-th source of the FusedElementwise graph node.
        // We need to lower that source node recursively.
        return lowerer.lower_node_rec(&graph_node.src[n], indices, inputs);
    }

    // Recursively process the source nodes of the fused AST.
    let new_srcs = fused_ast
        .src
        .iter()
        .map(|src| replace_captures_rec(lowerer, graph_node, indices, inputs, src))
        .collect();

    AstNode::_new(fused_ast.op.clone(), new_srcs, fused_ast.dtype.clone())
}

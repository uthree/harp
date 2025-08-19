use crate::ast::{AstNode, AstOp};
use crate::graph::{GraphNode, GraphNodeData, GraphOp};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

/// Fuse elementwise operations in the graph.
pub fn fuse_elementwise(nodes: &[GraphNode]) -> Vec<GraphNode> {
    let sorted_nodes = topological_sort(nodes);
    let mut memo = HashMap::new();
    let mut usage_counts = HashMap::new();

    for node in &sorted_nodes {
        for src in &node.src {
            *usage_counts.entry(src.clone()).or_insert(0) += 1;
        }
    }

    for node in &sorted_nodes {
        let new_srcs = node
            .src
            .iter()
            .map(|src| memo.get(src).cloned().unwrap_or_else(|| src.clone()))
            .collect::<Vec<_>>();

        let mut current_node = if new_srcs.iter().zip(&node.src).any(|(a, b)| a != b) {
            GraphNode(Rc::new(GraphNodeData {
                op: node.op.clone(),
                src: new_srcs,
                dtype: node.dtype.clone(),
                view: node.view.clone(),
            }))
        } else {
            node.clone()
        };

        if let GraphOp::Elementwise(op) = &current_node.op {
            let mut should_fuse = false;
            for src in &current_node.src {
                if (matches!(src.op, GraphOp::Elementwise(_) | GraphOp::FusedElementwise(_)))
                    && usage_counts.get(src).copied().unwrap_or(0) <= 1
                {
                    should_fuse = true;
                    break;
                }
            }

            if should_fuse {
                let mut ast_memo = HashMap::new();
                let mut final_inputs = Vec::new();
                collect_inputs(&current_node, &mut final_inputs, &usage_counts);
                final_inputs.sort_by_key(|n| Rc::as_ptr(&n.0));

                let fused_ast =
                    build_fused_ast_rec(&current_node, &mut ast_memo, &final_inputs, &usage_counts);

                current_node = GraphNode(Rc::new(GraphNodeData {
                    op: GraphOp::FusedElementwise(fused_ast),
                    src: final_inputs,
                    dtype: current_node.dtype.clone(),
                    view: current_node.view.clone(),
                }));
            }
        }
        memo.insert(node.clone(), current_node);
    }

    nodes
        .iter()
        .map(|n| memo.get(n).cloned().unwrap_or_else(|| n.clone()))
        .collect()
}

fn can_be_fused(node: &GraphNode) -> bool {
    matches!(node.op, GraphOp::Elementwise(_) | GraphOp::FusedElementwise(_))
}

fn collect_inputs(
    node: &GraphNode,
    inputs: &mut Vec<GraphNode>,
    usage_counts: &HashMap<GraphNode, usize>,
) {
    if can_be_fused(node) && usage_counts.get(node).copied().unwrap_or(0) <= 1 {
        for src in &node.src {
            collect_inputs(src, inputs, usage_counts);
        }
    } else if !inputs.contains(node) {
        inputs.push(node.clone());
    }
}

fn build_fused_ast_rec(
    node: &GraphNode,
    memo: &mut HashMap<GraphNode, AstNode>,
    inputs: &[GraphNode],
    usage_counts: &HashMap<GraphNode, usize>,
) -> AstNode {
    if let Some(ast) = memo.get(node) {
        return ast.clone();
    }

    if !can_be_fused(node) || usage_counts.get(node).copied().unwrap_or(0) > 1 {
        let idx = inputs
            .iter()
            .position(|i| i == node)
            .unwrap_or_else(|| panic!("Input node not found in inputs list: {:?}", node));
        return AstNode::capture(idx);
    }

    let ast_srcs = node
        .src
        .iter()
        .map(|src| build_fused_ast_rec(src, memo, inputs, usage_counts))
        .collect();

    let op = match &node.op {
        GraphOp::Elementwise(op) => op.clone(),
        GraphOp::FusedElementwise(ast) => {
            // This is a pre-fused node that we are fusing into a larger one.
            // We need to remap its captures to the new set of inputs.
            return remap_captures(ast, node, inputs);
        }
        _ => unreachable!(),
    };

    let ast = AstNode::_new(op, ast_srcs, node.dtype.clone());
    memo.insert(node.clone(), ast.clone());
    ast
}

fn remap_captures(ast: &AstNode, original_node: &GraphNode, new_inputs: &[GraphNode]) -> AstNode {
    match &ast.op {
        AstOp::Capture(idx) => {
            let original_input = &original_node.src[*idx];
            let new_idx = new_inputs.iter().position(|s| s == original_input).unwrap();
            AstNode::capture(new_idx)
        }
        _ => {
            let new_srcs = ast.src.iter().map(|s| remap_captures(s, original_node, new_inputs)).collect();
            AstNode::_new(ast.op.clone(), new_srcs, ast.dtype.clone())
        }
    }
}

fn topological_sort(nodes: &[GraphNode]) -> Vec<GraphNode> {
    let mut sorted = Vec::new();
    let mut visited = HashSet::new();
    for node in nodes {
        topological_sort_rec(node, &mut sorted, &mut visited);
    }
    sorted
}

fn topological_sort_rec(
    node: &GraphNode,
    sorted: &mut Vec<GraphNode>,
    visited: &mut HashSet<GraphNode>,
) {
    if !visited.insert(node.clone()) {
        return;
    }
    for src in &node.src {
        topological_sort_rec(src, sorted, visited);
    }
    sorted.push(node.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Const, DType};
    use crate::graph::shape::expr::Expr as ShapeExpr;
    use crate::graph::Graph;

    #[test]
    fn test_simple_fusion() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b; // elementwise
        let d = -c; // elementwise
        graph.outputs.push(d);

        graph.optimize();

        let final_node = &graph.outputs[0];
        assert!(matches!(final_node.op, GraphOp::FusedElementwise(_)));
        assert_eq!(final_node.src.len(), 2);
    }

    #[test]
    fn test_fusion_with_shared_node() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let shared = a + b; // shared node
        let out1 = -shared.clone();
        let out2 = shared.clone() * GraphNode::full(Const::F32(2.0f32.to_bits()), shape.clone());
        graph.outputs.push(out1);
        graph.outputs.push(out2);

        graph.optimize();

        // shared is used twice, so it should not be fused into out1.
        let final_node1 = &graph.outputs[0];
        assert!(matches!(final_node1.op, GraphOp::Elementwise(AstOp::Neg)));
        assert_eq!(final_node1.src.len(), 1);
        assert!(matches!(
            final_node1.src[0].op,
            GraphOp::Elementwise(AstOp::Add)
        ));

        // shared should not be fused into out2 either.
        let final_node2 = &graph.outputs[1];
        assert!(matches!(
            final_node2.op,
            GraphOp::Elementwise(AstOp::Mul)
        ));
        assert_eq!(final_node2.src.len(), 2);
        assert!(matches!(
            final_node2.src[0].op,
            GraphOp::Elementwise(AstOp::Add)
        ));
    }

    #[test]
    fn test_no_fusion_single_op() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b;
        graph.outputs.push(c);

        graph.optimize();

        let final_node = &graph.outputs[0];
        // No fusion should happen as there's only one elementwise op.
        assert!(matches!(final_node.op, GraphOp::Elementwise(AstOp::Add)));
    }

    #[test]
    fn test_complex_fusion_chain() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = graph.add_input(shape.clone(), &dtype);

        let x = a + b;
        let y = x * c;
        let z = -y;
        graph.outputs.push(z);

        graph.optimize();

        let final_node = &graph.outputs[0];
        assert!(matches!(final_node.op, GraphOp::FusedElementwise(_)));
        assert_eq!(final_node.src.len(), 3); // a, b, c
    }
}
use crate::ast::AstNode;
use crate::graph::{Graph, GraphOp, NodeData, NodeId};
use rustc_hash::{FxHashMap, FxHashSet};

pub struct ElementwiseFusion;

impl Default for ElementwiseFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementwiseFusion {
    pub fn new() -> Self {
        Self
    }
}

impl ElementwiseFusion {
    pub fn optimize(&self, graph: &Graph) -> Graph {
        let mut new_graph = Graph::new();
        // Step 1: Identify fusion groups without mutating the graph.
        let (fusion_groups, visited) = {
            let nodes = graph.nodes.borrow();
            let mut users = FxHashMap::default();
            for (id, node) in nodes.iter().enumerate() {
                for &src_id in &node.src {
                    users
                        .entry(src_id)
                        .or_insert_with(Vec::new)
                        .push(NodeId(id));
                }
            }

            let mut fusion_groups: FxHashMap<NodeId, Vec<NodeId>> = FxHashMap::default();
            let mut visited = FxHashSet::default();

            for i in (0..nodes.len()).rev() {
                let id = NodeId(i);
                if visited.contains(&id) {
                    continue;
                }

                let mut chain = vec![];
                let mut current_id = id;

                // Build a chain backwards from the current node.
                while let Some(node_data) = nodes.get(current_id.0) {
                    if !node_data.op.is_elementwise() || visited.contains(&current_id) {
                        break;
                    }
                    let user_count = users.get(&current_id).map_or(0, |u| u.len());
                    if user_count > 1 {
                        break;
                    }
                    // For now, only fuse single-input elementwise operations.
                    // This simplifies building the fused AST.
                    if node_data.src.len() != 1 {
                        break;
                    }

                    chain.push(current_id);
                    // We don't mark as visited here yet, only after a valid chain is found.
                    current_id = node_data.src[0];
                }

                if chain.len() > 1 {
                    // The chain is built backwards, so we reverse it to get the correct execution order.
                    chain.reverse();
                    let root_of_fusion = chain[0];
                    for &node_id in &chain {
                        visited.insert(node_id);
                    }
                    fusion_groups.insert(root_of_fusion, chain);
                }
            }
            (fusion_groups, visited)
        };

        if fusion_groups.is_empty() {
            return graph.clone(); // No fusion opportunities found.
        }

        // Step 2: Rebuild the graph with fused nodes.
        let original_nodes = graph.nodes.borrow().clone();
        let mut new_nodes: Vec<NodeData> = Vec::new();
        let mut old_to_new_id: FxHashMap<NodeId, NodeId> = FxHashMap::default();

        for (i, node) in original_nodes.iter().enumerate() {
            let old_id = NodeId(i);
            if visited.contains(&old_id) && !fusion_groups.contains_key(&old_id) {
                // This node is part of a fusion chain, but not the root.
                // It will be handled when the root is processed.
                continue;
            }

            let new_id = NodeId(new_nodes.len());

            if let Some(chain) = fusion_groups.get(&old_id) {
                let fused_node_data: Vec<_> = chain
                    .iter()
                    .map(|&id| original_nodes[id.0].clone())
                    .collect();

                let first_node = &fused_node_data[0];
                let last_node = fused_node_data.last().unwrap();

                // Build the fused AST. Start with a capture of the input.
                let mut current_ast = AstNode::capture(0, first_node.dtype.clone());
                // Apply each operation in the chain to the AST.
                for node_data in &fused_node_data {
                    if let GraphOp::Elementwise(op) = &node_data.op {
                        current_ast =
                            AstNode::new(op.clone(), vec![current_ast], node_data.dtype.clone());
                    }
                }

                let new_node = NodeData {
                    op: GraphOp::FusedElementwise(current_ast),
                    src: first_node.src.clone(), // The src of the first node in the chain is the src of the fused node.
                    dtype: last_node.dtype.clone(),
                    shape: last_node.shape.clone(),
                };
                new_nodes.push(new_node);

                for &id_in_chain in chain {
                    old_to_new_id.insert(id_in_chain, new_id);
                }
            } else {
                // This node is not part of any fusion.
                new_nodes.push(node.clone());
                old_to_new_id.insert(old_id, new_id);
            }
        }

        // Step 3: Remap src IDs in the new nodes.
        for node in &mut new_nodes {
            node.src = node
                .src
                .iter()
                .map(|old_id| *old_to_new_id.get(old_id).unwrap_or(old_id))
                .collect();
        }

        // Step 4: Update graph outputs.
        let mut outputs = graph.outputs.borrow().clone();
        for output_id in outputs.iter_mut() {
            if let Some(new_id) = old_to_new_id.get(output_id) {
                *output_id = *new_id;
            }
        }

        new_graph.nodes = std::cell::RefCell::new(new_nodes);
        new_graph.outputs = std::cell::RefCell::new(outputs);
        new_graph.inputs = graph.inputs.clone();
        new_graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstOp, DType};
    use crate::graph::GraphOp;

    #[test]
    fn test_elementwise_fusion() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = -a;
        let c = b.sin();
        let _d = c.exp2().as_output();

        let optimizer = ElementwiseFusion::new();
        let new_graph = optimizer.optimize(&graph);

        let nodes = new_graph.nodes.borrow();
        assert_eq!(
            nodes.len(),
            2,
            "Graph should have an input node and a fused node"
        );

        let fused_node = &nodes[1];
        if let GraphOp::FusedElementwise(ast) = &fused_node.op {
            // Expected AST: exp2(sin(-capture(0)))
            assert_eq!(ast.op, AstOp::Exp2);
            let sin_node = &ast.src[0];
            assert_eq!(sin_node.op, AstOp::Sin);
            let neg_node = &sin_node.src[0];
            assert_eq!(neg_node.op, AstOp::Neg);
            let capture_node = &neg_node.src[0];
            assert!(matches!(capture_node.op, AstOp::Capture(0, _)));
        } else {
            panic!(
                "Expected a FusedElementwise node, but found {:?}",
                fused_node.op
            );
        }

        // Check that the source of the fused node is the original input
        assert_eq!(fused_node.src, vec![a.id]);
    }
}

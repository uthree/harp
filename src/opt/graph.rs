use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::{Graph, NodeData, NodeId, TensorOp};

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

    pub fn run(&self, graph: &mut Graph) {
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
                    if node_data.src.len() != 1 {
                        // Only fuse single-input ops for now.
                        break;
                    }

                    chain.push(current_id);
                    visited.insert(current_id);
                    current_id = node_data.src[0];
                }

                if chain.len() > 1 {
                    let root_of_fusion = *chain.last().unwrap();
                    // The chain is built backwards, so we reverse it to get the correct execution order.
                    chain.reverse();
                    fusion_groups.insert(root_of_fusion, chain);
                }
            }
            (fusion_groups, visited)
        };

        if fusion_groups.is_empty() {
            return; // No fusion opportunities found.
        }

        // Step 2: Rebuild the graph with fused nodes.
        let new_nodes: Vec<NodeData> = Vec::new();
        let old_to_new_id: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        let original_nodes = graph.nodes.borrow().clone();

        {
            // Rebuild the graph
            let mut new_nodes: Vec<NodeData> = Vec::new();
            let mut old_to_new_id: FxHashMap<NodeId, NodeId> = FxHashMap::default();

            for (i, node) in original_nodes.iter().enumerate() {
                let old_id = NodeId(i);
                if visited.contains(&old_id) && !fusion_groups.contains_key(&old_id) {
                    continue;
                }

                let new_id = NodeId(new_nodes.len());

                if let Some(chain) = fusion_groups.get(&old_id) {
                    let fused_node_data: Vec<NodeData> = chain
                        .iter()
                        .map(|&id| original_nodes[id.0].clone())
                        .collect();
                    let first_node = fused_node_data.first().unwrap();
                    let last_node = fused_node_data.last().unwrap();

                    new_nodes.push(NodeData {
                        op: TensorOp::Fused(fused_node_data.clone()),
                        src: first_node.src.clone(),
                        dtype: last_node.dtype.clone(),
                        shape: last_node.shape.clone(),
                    });
                    for &id_in_chain in chain {
                        old_to_new_id.insert(id_in_chain, new_id);
                    }
                } else {
                    new_nodes.push(node.clone());
                    old_to_new_id.insert(old_id, new_id);
                }
            }

            for node in &mut new_nodes {
                node.src = node
                    .src
                    .iter()
                    .map(|old_id| *old_to_new_id.get(old_id).unwrap())
                    .collect();
            }

            let mut outputs = graph.outputs.borrow_mut();
            for output_id in outputs.iter_mut() {
                if let Some(new_id) = old_to_new_id.get(output_id) {
                    *output_id = *new_id;
                }
            }
            *graph.nodes.borrow_mut() = new_nodes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_elementwise_fusion() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = -a;
        let c = b.sin();
        let _d = c.exp2().as_output();

        let optimizer = ElementwiseFusion::new();
        optimizer.run(&mut graph);

        let nodes = graph.nodes.borrow();
        assert_eq!(nodes.len(), 2); // Input + Fused node

        let fused_node = &nodes[1];
        if let TensorOp::Fused(fused_ops) = &fused_node.op {
            assert_eq!(fused_ops.len(), 3); // Neg, Sin, Exp2
            assert!(matches!(fused_ops[0].op, TensorOp::Elementwise(_))); // Neg
            assert!(matches!(fused_ops[1].op, TensorOp::Elementwise(_))); // Sin
            assert!(matches!(fused_ops[2].op, TensorOp::Elementwise(_))); // Exp2
        } else {
            panic!("Expected a fused node, but found {:?}", fused_node.op);
        }
    }
}

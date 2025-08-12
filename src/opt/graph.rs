use crate::ast::AstNode;
use crate::graph::{Graph, GraphOp, NodeData, NodeId};
use crate::opt::DeterministicGraphOptimizer;
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

impl DeterministicGraphOptimizer for ElementwiseFusion {
    fn optimize(&self, graph: &Graph) -> Graph {
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
                    if user_count > 1 && !graph.outputs.borrow().contains(&current_id) {
                        break;
                    }

                    chain.push(current_id);

                    // Find the next node to check. If multiple sources, prefer the one that is not a constant.
                    let non_const_srcs: Vec<_> = node_data
                        .src
                        .iter()
                        .filter(|&&id| !nodes[id.0].op.is_full())
                        .collect();

                    if non_const_srcs.len() == 1 {
                        current_id = *non_const_srcs[0];
                    } else {
                        // If there are 0 or >1 non-const sources, we can't continue the chain linearly.
                        break;
                    }
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
                let last_node_in_chain = chain.last().unwrap();
                let last_node_data = &original_nodes[last_node_in_chain.0];

                let mut fused_srcs = Vec::new();
                let mut node_to_ast: FxHashMap<NodeId, AstNode> = FxHashMap::default();

                // Find all unique non-constant sources for the entire fused operation
                for &node_id in chain.iter() {
                    let node_data = &original_nodes[node_id.0];
                    for &src_id in &node_data.src {
                        let src_node_data = &original_nodes[src_id.0];
                        // If the source is not part of the chain and not a constant, it's an external input.
                        if !chain.contains(&src_id)
                            && !src_node_data.op.is_full()
                            && !fused_srcs.contains(&src_id)
                        {
                            fused_srcs.push(src_id);
                        }
                    }
                }

                // Build the fused AST
                for &node_id in chain.iter() {
                    let node_data = &original_nodes[node_id.0];
                    let mut ast_srcs = Vec::new();
                    for &src_id in &node_data.src {
                        if let Some(existing_ast) = node_to_ast.get(&src_id) {
                            ast_srcs.push(existing_ast.clone());
                        } else {
                            let src_node_data = &original_nodes[src_id.0];
                            if let GraphOp::Full(const_val) = src_node_data.op {
                                ast_srcs.push(const_val.into());
                            } else {
                                // It's an external input, create a capture node.
                                let capture_idx =
                                    fused_srcs.iter().position(|&id| id == src_id).unwrap();
                                ast_srcs.push(AstNode::capture(
                                    capture_idx,
                                    src_node_data.dtype.clone(),
                                ));
                            }
                        }
                    }
                    if let GraphOp::Elementwise(op) = &node_data.op {
                        let new_ast =
                            AstNode::new(op.clone(), ast_srcs, node_data.dtype.clone());
                        node_to_ast.insert(node_id, new_ast);
                    }
                }

                let final_fused_ast = node_to_ast.get(last_node_in_chain).unwrap().clone();

                let new_node = NodeData {
                    op: GraphOp::FusedElementwise(final_fused_ast),
                    src: fused_srcs,
                    dtype: last_node_data.dtype.clone(),
                    shape: last_node_data.shape.clone(),
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

    #[test]
    fn test_multi_input_elementwise_fusion() {
        crate::init_logger();
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let const_2 = graph.full(2.0f32, vec![]);
        let const_1 = graph.full(1.0f32, vec![]);

        let b = a * const_2;
        let c = b + const_1; // This should all be fused
        let _d = c.as_output();

        let optimizer = ElementwiseFusion::new();
        let new_graph = optimizer.optimize(&graph);

        let nodes = new_graph.nodes.borrow();
        assert_eq!(
            nodes.len(),
            4,
            "Graph should have input, 2 consts, and a fused node"
        );

        let fused_node = nodes
            .iter()
            .find(|n| matches!(n.op, GraphOp::FusedElementwise(_)))
            .unwrap();

        assert_eq!(fused_node.src.len(), 1, "Fused node should have one source");
        assert!(fused_node.src.contains(&a.id));

        if let GraphOp::FusedElementwise(ast) = &fused_node.op {
            // Expected AST: (capture(a) * 2.0) + 1.0
            assert_eq!(ast.op, AstOp::Add);
            assert_eq!(ast.src.len(), 2);

            let (mul_node, const_1_node) = (&ast.src[0], &ast.src[1]);
            assert!(matches!(const_1_node.op, AstOp::Const(_)));

            assert_eq!(mul_node.op, AstOp::Mul);
            assert_eq!(mul_node.src.len(), 2);
            assert!(matches!(mul_node.src[0].op, AstOp::Capture(_, _)));
            assert!(matches!(mul_node.src[1].op, AstOp::Const(_)));
        } else {
            panic!("Expected FusedElementwise node");
        }
    }
}

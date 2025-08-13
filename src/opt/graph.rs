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
                        let new_ast = AstNode::new(op.clone(), ast_srcs, node_data.dtype.clone());
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

pub struct FuseElementwiseReduce;

impl FuseElementwiseReduce {
    pub fn new() -> Self {
        Self
    }
}

impl Default for FuseElementwiseReduce {
    fn default() -> Self {
        Self::new()
    }
}

impl DeterministicGraphOptimizer for FuseElementwiseReduce {
    fn optimize(&self, graph: &Graph) -> Graph {
        let mut new_graph = graph.clone();
        let mut replacements: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        let mut changed = true;

        while changed {
            changed = false;
            let nodes = new_graph.nodes.borrow().clone();
            let mut users = FxHashMap::default();
            for (id, node) in nodes.iter().enumerate() {
                for &src_id in &node.src {
                    users
                        .entry(src_id)
                        .or_insert_with(Vec::new)
                        .push(NodeId(id));
                }
            }

            for (reduce_node_id_val, reduce_node) in nodes.iter().enumerate().rev() {
                let reduce_node_id = NodeId(reduce_node_id_val);
                if replacements.contains_key(&reduce_node_id) {
                    continue;
                }

                if let GraphOp::Reduce(reduce_op, axis) = &reduce_node.op {
                    let elementwise_node_id = reduce_node.src[0];
                    let elementwise_node = &nodes[elementwise_node_id.0];

                    let user_count = users.get(&elementwise_node_id).map_or(0, |u| u.len());

                    if let GraphOp::Elementwise(elementwise_op) = &elementwise_node.op
                        && user_count == 1
                    {
                        // This is a candidate for fusion.
                        let mut ast_srcs = Vec::new();
                        for &src_id in &elementwise_node.src {
                            let src_node_data = &nodes[src_id.0];
                            if let GraphOp::Full(const_val) = src_node_data.op {
                                ast_srcs.push(const_val.into());
                            } else {
                                let capture_idx = elementwise_node
                                    .src
                                    .iter()
                                    .position(|&id| id == src_id)
                                    .unwrap();
                                ast_srcs.push(AstNode::capture(
                                    capture_idx,
                                    src_node_data.dtype.clone(),
                                ));
                            }
                        }

                        let fused_ast = AstNode::new(
                            elementwise_op.clone(),
                            ast_srcs,
                            elementwise_node.dtype.clone(),
                        );

                        let new_op = GraphOp::FusedElementwiseReduce(
                            fused_ast,
                            reduce_op.clone(),
                            vec![*axis],
                        );

                        let new_node_id = new_graph.add_node(
                            new_op,
                            elementwise_node.src.clone(),
                            reduce_node.dtype.clone(),
                            reduce_node.shape.clone(),
                        );

                        replacements.insert(reduce_node_id, new_node_id);
                        changed = true;
                        break; // Restart the process with the modified graph
                    }
                }
            }

            if changed {
                let mut final_graph = new_graph.clone();
                let mut final_nodes = final_graph.nodes.borrow().clone();
                for node in &mut final_nodes {
                    for src_id in &mut node.src {
                        if let Some(new_id) = replacements.get(src_id) {
                            *src_id = *new_id;
                        }
                    }
                }
                let mut final_outputs = final_graph.outputs.borrow().clone();
                for output_id in &mut final_outputs {
                    if let Some(new_id) = replacements.get(output_id) {
                        *output_id = *new_id;
                    }
                }

                final_graph.nodes = std::cell::RefCell::new(final_nodes);
                final_graph.outputs = std::cell::RefCell::new(final_outputs);
                new_graph = final_graph;
            }
        }

        new_graph
    }
}

pub struct FuseReductions;

impl FuseReductions {
    pub fn new() -> Self {
        Self
    }
}

pub struct CompositeGraphOptimizer {
    optimizers: Vec<Box<dyn DeterministicGraphOptimizer>>,
}

impl CompositeGraphOptimizer {
    pub fn new(optimizers: Vec<Box<dyn DeterministicGraphOptimizer>>) -> Self {
        CompositeGraphOptimizer { optimizers }
    }
}

impl DeterministicGraphOptimizer for CompositeGraphOptimizer {
    fn optimize(&self, graph: &Graph) -> Graph {
        let mut graph = graph.clone();
        for opt in self.optimizers.iter() {
            graph = opt.optimize(&graph);
        }
        graph
    }
}

impl Default for FuseReductions {
    fn default() -> Self {
        Self::new()
    }
}

impl DeterministicGraphOptimizer for FuseReductions {
    fn optimize(&self, graph: &Graph) -> Graph {
        let mut new_graph = graph.clone();
        let mut replacements: FxHashMap<NodeId, NodeId> = FxHashMap::default();
        let mut changed = true;

        while changed {
            changed = false;
            let nodes = new_graph.nodes.borrow().clone();
            let mut users = FxHashMap::default();
            for (id, node) in nodes.iter().enumerate() {
                for &src_id in &node.src {
                    users
                        .entry(src_id)
                        .or_insert_with(Vec::new)
                        .push(NodeId(id));
                }
            }

            for (outer_reduce_id_val, outer_reduce_node) in nodes.iter().enumerate().rev() {
                let outer_reduce_id = NodeId(outer_reduce_id_val);
                if replacements.contains_key(&outer_reduce_id) {
                    continue;
                }

                if let GraphOp::Reduce(outer_op, outer_axis) = &outer_reduce_node.op {
                    let inner_reduce_id = outer_reduce_node.src[0];
                    let inner_reduce_node = &nodes[inner_reduce_id.0];
                    let user_count = users.get(&inner_reduce_id).map_or(0, |u| u.len());

                    if user_count == 1 {
                        if let GraphOp::Reduce(inner_op, inner_axis) = &inner_reduce_node.op {
                            if inner_op == outer_op {
                                // Fusion condition met.
                                let new_op = GraphOp::FusedReduce(
                                    outer_op.clone(),
                                    vec![*inner_axis, *outer_axis],
                                );
                                let new_node_id = new_graph.add_node(
                                    new_op,
                                    inner_reduce_node.src.clone(),
                                    outer_reduce_node.dtype.clone(),
                                    outer_reduce_node.shape.clone(),
                                );
                                replacements.insert(outer_reduce_id, new_node_id);
                                changed = true;
                                break; // Restart scan
                            }
                        } else if let GraphOp::FusedReduce(inner_op, inner_axes) =
                            &inner_reduce_node.op
                            && inner_op == outer_op
                        {
                            // Also fuse a Reduce into an existing FusedReduce
                            let mut new_axes = inner_axes.clone();
                            new_axes.push(*outer_axis);
                            let new_op = GraphOp::FusedReduce(outer_op.clone(), new_axes);
                            let new_node_id = new_graph.add_node(
                                new_op,
                                inner_reduce_node.src.clone(),
                                outer_reduce_node.dtype.clone(),
                                outer_reduce_node.shape.clone(),
                            );
                            replacements.insert(outer_reduce_id, new_node_id);
                            changed = true;
                            break; // Restart scan
                        }
                    }
                }
            }

            if changed {
                let mut final_graph = new_graph.clone();
                let mut final_nodes = final_graph.nodes.borrow().clone();
                for node in &mut final_nodes {
                    for src_id in &mut node.src {
                        if let Some(new_id) = replacements.get(src_id) {
                            *src_id = *new_id;
                        }
                    }
                }
                let mut final_outputs = final_graph.outputs.borrow().clone();
                for output_id in &mut final_outputs {
                    if let Some(new_id) = replacements.get(output_id) {
                        *output_id = *new_id;
                    }
                }

                final_graph.nodes = std::cell::RefCell::new(final_nodes);
                final_graph.outputs = std::cell::RefCell::new(final_outputs);
                new_graph = final_graph;
            }
        }
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

    #[test]
    fn test_fuse_elementwise_reduce() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into()]);
        let b = graph.input(DType::F32, vec![10.into()]);
        let c = (a + b).sum(0).as_output();

        let optimizer = FuseElementwiseReduce::new();
        let new_graph = optimizer.optimize(&graph);

        let new_nodes = new_graph.nodes.borrow();
        let final_op = &new_graph
            .get_view(*new_graph.outputs.borrow().first().unwrap())
            .op();

        assert!(matches!(final_op, GraphOp::FusedElementwiseReduce(..)));
    }

    #[test]
    fn test_fuse_reductions() {
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![10.into(), 20.into(), 30.into()]);
        let b = a.sum(0);
        let _c = b.sum(1).as_output(); // sum(0) and sum(1) should be fused

        let optimizer = FuseReductions::new();
        let new_graph = optimizer.optimize(&graph);

        let final_op = &new_graph
            .get_view(*new_graph.outputs.borrow().first().unwrap())
            .op();

        if let GraphOp::FusedReduce(op, axes) = final_op {
            assert_eq!(*op, AstOp::Add);
            assert_eq!(axes.len(), 2);
            assert!(axes.contains(&0));
            // The second reduction axis is on the *new* shape, so it's axis 1, not 2.
            assert!(axes.contains(&1));
        } else {
            panic!("Expected FusedReduce op, but found {:?}", final_op);
        }
    }
}

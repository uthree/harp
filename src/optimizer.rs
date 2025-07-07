use crate::{graph::Graph, interpreter::Interpreter, node::Node, operator, tensor::TensorData};
use petgraph::{Direction, algo::toposort, graph::NodeIndex, visit::EdgeRef};
use std::collections::HashMap;

pub trait GraphOptimizer {
    fn optimize(&mut self, graph: &mut Graph);
}

pub struct EliminateUnusedNodes {}

impl GraphOptimizer for EliminateUnusedNodes {
    fn optimize(&mut self, graph: &mut Graph) {
        let mut nodes_to_remove = Vec::new();
        let topo_order = toposort(&graph.graph, None).unwrap();

        for node_idx in topo_order.iter().rev() {
            // If a node has no outgoing edges (i.e., no consumers) and is not an output node (which we don't have yet),
            // it's unused. For now, we assume any node with no outgoing edges is unused.
            if graph
                .graph
                .edges_directed(*node_idx, Direction::Outgoing)
                .count()
                == 0
                && !graph.outputs.contains(node_idx)
                && !graph.inputs.contains(node_idx)
            {
                nodes_to_remove.push(*node_idx);
            }
        }

        for node_idx in nodes_to_remove {
            graph.graph.remove_node(node_idx);
        }
    }
}

pub struct ConstantFolding {}

impl GraphOptimizer for ConstantFolding {
    fn optimize(&mut self, graph: &mut Graph) {
        let topo_order = toposort(&graph.graph, None).unwrap();
        let mut interpreter = Interpreter::new();
        let mut nodes_to_replace = HashMap::new();

        for node_idx in topo_order {
            let node = graph.node_weight(node_idx).unwrap();
            let op = node.op();

            // Check if all parents are already constants or inputs
            let mut all_parents_are_const_or_input = true;
            let mut parent_data = HashMap::<NodeIndex, TensorData>::new();

            for edge in graph.graph.edges_directed(node_idx, Direction::Incoming) {
                let parent_idx = edge.source();
                let parent_node = graph.node_weight(parent_idx).unwrap();
                if let Some(const_op) = parent_node.op().as_any().downcast_ref::<operator::Const>()
                {
                    parent_data.insert(edge.source(), const_op.data.clone());
                } else if parent_node.op().as_any().downcast_ref::<operator::Input>().is_some() {
                    all_parents_are_const_or_input = false;
                    break;
                } else {
                    all_parents_are_const_or_input = false;
                    break;
                }
            }

            if all_parents_are_const_or_input {
                // If all parents are constants, evaluate this node
                if let Ok(result_data) = interpreter.evaluate(node_idx, &graph.graph, &parent_data)
                {
                    nodes_to_replace.insert(node_idx, result_data);
                }
            }
        }

        // Replace evaluated nodes with new Const nodes
        for (node_idx, data) in nodes_to_replace {
            let new_op = operator::Const { data };
            let new_node = Node::new(new_op, graph.node_weight(node_idx).unwrap().shape.clone());
            graph.graph.add_node(new_node);
            // TODO: Replace edges properly. This is complex and requires re-wiring.
            // For now, just adding the new node. Actual graph transformation will be more involved.
        }
    }
}

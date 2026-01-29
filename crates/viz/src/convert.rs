//! Conversion utilities from GraphNode to egui-snarl Snarl

use std::collections::HashMap;

use egui_snarl::{InPinId, NodeId, OutPinId, Snarl};

use eclat::graph::{GraphNode, GraphOp};

/// Visualization node for egui-snarl
#[derive(Clone)]
pub struct VizNode {
    /// Label to display
    pub label: String,
    /// Operation type name
    pub op_type: String,
    /// Shape as string
    pub shape_str: String,
    /// DType as string
    pub dtype_str: String,
    /// Optional name
    pub name: Option<String>,
    /// Number of input connections
    pub input_count: usize,
    /// Whether this node was changed in the current step
    pub changed: bool,
}

impl VizNode {
    /// Create a VizNode from a GraphNode
    pub fn from_graph_node(node: &GraphNode) -> Self {
        use eclat::graph::Expr;

        let inner = &node.0;

        // Short op type names
        let op_type = match &inner.op {
            GraphOp::View(_) => "View".to_string(),
            GraphOp::MapReduce { reduce, .. } => {
                if let Some((op, _)) = reduce {
                    format!("{:?}", op)  // Just "Sum", "Max", etc.
                } else {
                    "Map".to_string()
                }
            }
            GraphOp::Unfold { .. } => "Unfold".to_string(),
            GraphOp::Scatter { .. } => "Scatter".to_string(),
            GraphOp::Scan { scan_op, .. } => format!("Scan{:?}", scan_op),
            GraphOp::MatMul { .. } => "MatMul".to_string(),
        };

        let label = inner.name.clone().unwrap_or_else(|| op_type.clone());

        // Compact shape: [64,128] instead of [Const(64), Const(128)]
        let shape_str = inner
            .view
            .shape()
            .iter()
            .map(|e| match e {
                Expr::Const(v) => v.to_string(),
                Expr::Sym(name) => name.clone(),
                _ => "?".to_string(),
            })
            .collect::<Vec<_>>()
            .join("×");

        let dtype_str = format!("{:?}", inner.dtype);

        Self {
            label,
            op_type,
            shape_str,
            dtype_str,
            name: inner.name.clone(),
            input_count: inner.src.len(),
            changed: false,
        }
    }
}

/// Convert a list of root GraphNodes to a Snarl graph
pub fn graph_to_snarl(roots: &[GraphNode]) -> Snarl<VizNode> {
    let mut snarl = Snarl::new();
    let mut node_map: HashMap<usize, NodeId> = HashMap::new();
    let mut depth_count: HashMap<usize, usize> = HashMap::new();

    // First pass: calculate max depth
    fn calc_depth(node: &GraphNode, cache: &mut HashMap<usize, usize>) -> usize {
        let ptr = std::rc::Rc::as_ptr(&node.0) as usize;
        if let Some(&d) = cache.get(&ptr) {
            return d;
        }
        let depth = if node.0.src.is_empty() {
            0
        } else {
            node.0.src.iter().map(|s| calc_depth(s, cache)).max().unwrap_or(0) + 1
        };
        cache.insert(ptr, depth);
        depth
    }

    let mut depth_cache: HashMap<usize, usize> = HashMap::new();
    for root in roots {
        calc_depth(root, &mut depth_cache);
    }
    let max_depth = depth_cache.values().copied().max().unwrap_or(0);

    // Helper function to get or create a node
    fn get_or_create_node(
        graph_node: &GraphNode,
        snarl: &mut Snarl<VizNode>,
        node_map: &mut HashMap<usize, NodeId>,
        depth_count: &mut HashMap<usize, usize>,
        depth_cache: &HashMap<usize, usize>,
        max_depth: usize,
    ) -> NodeId {
        let ptr = std::rc::Rc::as_ptr(&graph_node.0) as usize;

        if let Some(&node_id) = node_map.get(&ptr) {
            return node_id;
        }

        // Create VizNode
        let viz_node = VizNode::from_graph_node(graph_node);

        // Get depth and calculate position
        // Inputs (depth=0) on left, outputs (roots) on right
        let depth = depth_cache.get(&ptr).copied().unwrap_or(0);
        let x_pos = depth as f32 * 180.0;
        let y_idx = *depth_count.get(&depth).unwrap_or(&0);
        depth_count.insert(depth, y_idx + 1);
        let y_pos = y_idx as f32 * 80.0;

        let pos = egui::pos2(x_pos, y_pos);
        let node_id = snarl.insert_node(pos, viz_node);
        node_map.insert(ptr, node_id);

        // Process sources (inputs)
        for (input_idx, src) in graph_node.0.src.iter().enumerate() {
            let src_node_id = get_or_create_node(
                src, snarl, node_map, depth_count, depth_cache, max_depth
            );

            let out_pin = OutPinId {
                node: src_node_id,
                output: 0,
            };
            let in_pin = InPinId {
                node: node_id,
                input: input_idx,
            };
            snarl.connect(out_pin, in_pin);
        }

        node_id
    }

    // Process all roots
    for root in roots {
        get_or_create_node(
            root, &mut snarl, &mut node_map, &mut depth_count, &depth_cache, max_depth
        );
    }

    snarl
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let roots: Vec<GraphNode> = vec![];
        let snarl = graph_to_snarl(&roots);
        assert!(snarl.node_ids().next().is_none());
    }
}

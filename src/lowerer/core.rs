use crate::ast::AstNode;
use crate::graph::{Graph, GraphNode};
use std::collections::HashMap;

pub struct Lowerer {
    pub(super) next_temp_id: usize,
    pub(super) node_to_var: HashMap<GraphNode, String>,
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            next_temp_id: 0,
            node_to_var: HashMap::new(),
        }
    }

    pub fn lower(&mut self, graph: &Graph) -> AstNode {
        let kernel_function = self.create_kernel_function(graph);
        let entry_function = self.create_entry_function(graph, &kernel_function);

        AstNode::program(vec![kernel_function, entry_function], "kernel_main")
    }

    pub(super) fn get_or_create_var_name(&mut self, node: &GraphNode) -> String {
        if let Some(name) = self.node_to_var.get(node) {
            name.clone()
        } else {
            let name = format!("temp{}", self.next_temp_id);
            self.next_temp_id += 1;
            self.node_to_var.insert(node.clone(), name.clone());
            name
        }
    }
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

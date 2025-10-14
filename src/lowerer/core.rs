use crate::ast::AstNode;
use crate::graph::{Graph, GraphNode};
use std::collections::HashMap;

/// Internal context for lowering process
/// Holds temporary state during Graph -> AST conversion
pub(in crate::lowerer) struct LowerContext {
    pub(in crate::lowerer) next_temp_id: usize,
    pub(in crate::lowerer) node_to_var: HashMap<GraphNode, String>,
}

impl LowerContext {
    fn new() -> Self {
        Self {
            next_temp_id: 0,
            node_to_var: HashMap::new(),
        }
    }

    pub(in crate::lowerer) fn get_or_create_var_name(&mut self, node: &GraphNode) -> String {
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

/// Public lowerer API for backward compatibility
pub struct Lowerer {
    ctx: LowerContext,
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            ctx: LowerContext::new(),
        }
    }

    pub fn lower(&mut self, graph: &Graph) -> AstNode {
        let kernel_function = self.ctx.create_kernel_function(graph);
        let entry_function = self.ctx.create_entry_function(graph, &kernel_function);

        AstNode::program(vec![kernel_function, entry_function], "kernel_main")
    }
}

/// Lower a Graph to an AST Program
///
/// This is the main entry point for the lowering process.
/// Converts a computation graph into an executable AST program.
pub fn lower(graph: &Graph) -> AstNode {
    let mut lowerer = Lowerer::new();
    lowerer.lower(graph)
}

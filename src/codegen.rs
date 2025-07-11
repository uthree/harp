use crate::node::{Node, NodeData};
use crate::renderer::Renderer;
use std::collections::HashMap;

/// A code generation engine that traverses a computation graph and renders it to a string.
pub struct CodeGenerator<'a> {
    /// The renderer backend (e.g., `CRenderer`).
    renderer: &'a dyn Renderer,
    /// A cache to store the rendered string for each node to avoid re-computation.
    node_cache: HashMap<*const NodeData, String>,
}

impl<'a> CodeGenerator<'a> {
    /// Creates a new `CodeGenerator` with the given renderer.
    pub fn new(renderer: &'a dyn Renderer) -> Self {
        Self {
            renderer,
            node_cache: HashMap::new(),
        }
    }

    /// Renders a single node in the graph, using a cache to avoid re-computation.
    fn render_node(&mut self, node: &Node) -> String {
        // Check cache first
        if let Some(cached) = self.node_cache.get(&node.ptr()) {
            return cached.clone();
        }

        // 1. Recursively render the source (operand) nodes.
        let rendered_operands: Vec<String> = node
            .src()
            .iter()
            .map(|child_node| self.render_node(child_node))
            .collect();

        // 2. Try to render the current node's operator directly.
        let result = if let Some(rendered_op) = self.renderer.render_op(node.op().as_ref(), &rendered_operands) {
            rendered_op
        // 3. If direct rendering is not supported, check for a FusedOp fallback.
        } else if let Some(fused_op) = node.op().as_fused_op() {
            let fallback_node = fused_op.fallback(node.src());
            self.render_node(&fallback_node)
        // 4. If no rendering path is found, panic.
        } else {
            panic!("Rendering not implemented for operator: {}", node.op().name());
        };

        // Cache the result and return
        self.node_cache.insert(node.ptr(), result.clone());
        result
    }

    /// Generates the final code for the entire graph starting from the root node.
    pub fn generate(&mut self, root: &Node) -> String {
        self.render_node(root)
    }
}

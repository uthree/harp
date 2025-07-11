use crate::node::{Node, NodeData};
use crate::op::FusedOp;
use crate::backend::renderer::Renderer;
use std::collections::{HashMap, HashSet};

/// A code generation engine that traverses a computation graph and renders it to a string.
pub struct CodeGenerator<'a> {
    renderer: &'a dyn Renderer,
    node_to_var: HashMap<*const NodeData, String>,
    statements: Vec<String>,
    var_counter: usize,
}

impl<'a> CodeGenerator<'a> {
    /// Creates a new `CodeGenerator` with the given renderer.
    pub fn new(renderer: &'a dyn Renderer) -> Self {
        Self {
            renderer,
            node_to_var: HashMap::new(),
            statements: Vec::new(),
            var_counter: 0,
        }
    }

    fn new_var(&mut self) -> String {
        let name = format!("v{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    /// Generates the final code for the entire graph starting from the root node.
    pub fn generate(&mut self, root: &Node) -> String {
        let sorted_nodes = self.topological_sort(root);

        for node in &sorted_nodes {
            self.render_node(node);
        }

        let function_body = self.statements.join("\n    ");

        // If the root node is a Store, it doesn't produce a return value.
        if root.op().as_any().is::<crate::op::Store>() {
            format!("void compute() {{\n    {}\n}}", function_body)
        } else {
            let result_var = self.node_to_var.get(&root.ptr()).unwrap();
            format!(
                "float compute() {{\n    {}\n    return {};\n}}",
                function_body, result_var
            )
        }
    }

    /// Renders a single node and stores the result (variable name and statement).
    fn render_node(&mut self, node: &Node) {
        if self.node_to_var.contains_key(&node.ptr()) {
            return;
        }

        let rendered_operands: Vec<String> = node
            .src()
            .iter()
            .map(|child| self.node_to_var.get(&child.ptr()).unwrap().clone())
            .collect();

        if let Some(expr) = self.renderer.render_op(node.op().as_ref(), &rendered_operands) {
            if node.op().as_any().is::<crate::op::Const>()
                || node.op().as_any().is::<crate::op::Variable>()
                || node.op().as_any().is::<crate::op::LoopVariable>()
            {
                self.node_to_var.insert(node.ptr(), expr);
            } else {
                let var_name = self.new_var();
                self.statements.push(format!("float {var_name} = {expr};"));
                self.node_to_var.insert(node.ptr(), var_name);
            }
        // Handle Store operator specifically
        } else if let Some(store_op) = node.op().as_any().downcast_ref::<crate::op::Store>() {
            // src[0] is the index, src[1] is the value to store.
            let index_var = self.node_to_var.get(&node.src()[0].ptr()).unwrap();
            let value_var = self.node_to_var.get(&node.src()[1].ptr()).unwrap();
            let statement = format!("{}[{}] = {};", store_op.0, index_var, value_var);
            self.statements.push(statement);
            // Store does not produce a value, so we don't add it to node_to_var.
            // Or we could add a special marker if needed. For now, we do nothing.
            
        // Handle Loop operator specifically
        } else if let Some(loop_op) = node.op().as_any().downcast_ref::<crate::op::Loop>() {
            let count_var = self.node_to_var.get(&loop_op.count.ptr()).unwrap();
            
            // Generate the body of the loop using a separate generator
            let mut body_codegen = CodeGenerator::new(self.renderer);
            let _body_result_var = body_codegen.generate_loop_body(&loop_op.body);

            let loop_body_code = body_codegen.statements.join("\n        ");
            let loop_statement = format!(
                "for (int i = 0; i < {}; ++i) {{\n        {}\n    }}",
                count_var, loop_body_code
            );
            self.statements.push(loop_statement);
            // Note: The result of the loop is not captured for now.
            // This would require a more complex mechanism to handle loop-carried dependencies.
            self.node_to_var.insert(node.ptr(), "loop_result".to_string());

        } else if let Some(fused_op) = node.op().as_fused_op() {
            let fallback_node = fused_op.fallback(node.src());
            // Recursively render the fallback graph. This is not the most efficient
            // way, but it works for now. A better way would be to integrate the
            // fallback graph directly into the topological sort.
            let var_name = self.render_fallback_node(&fallback_node);
            self.node_to_var.insert(node.ptr(), var_name);
        } else {
            panic!("Rendering not implemented for operator: {}", node.op().name());
        }
    }

    /// Generates the body of a loop, returning the final variable name.
    fn generate_loop_body(&mut self, root: &Node) -> String {
        let sorted_nodes = self.topological_sort(root);
        for node in &sorted_nodes {
            self.render_node(node);
        }
        self.node_to_var.get(&root.ptr()).unwrap().clone()
    }
    
    fn render_fallback_node(&mut self, node: &Node) -> String {
        if let Some(var) = self.node_to_var.get(&node.ptr()) {
            return var.clone();
        }
        let rendered_operands: Vec<String> =
            node.src().iter().map(|n| self.render_fallback_node(n)).collect();
        let expr = self.renderer.render_op(node.op().as_ref(), &rendered_operands).unwrap();
        if node.op().as_any().is::<crate::op::Const>()
            || node.op().as_any().is::<crate::op::Variable>()
            || node.op().as_any().is::<crate::op::LoopVariable>()
        {
            expr
        } else {
            let var_name = self.new_var();
            self.statements.push(format!("float {var_name} = {expr};"));
            self.node_to_var.insert(node.ptr(), var_name.clone());
            var_name
        }
    }

    /// Performs a topological sort of the graph starting from the root node.
    fn topological_sort(&self, root: &Node) -> Vec<Node> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        self.visit(root, &mut visited, &mut sorted);
        sorted
    }

    fn visit(&self, node: &Node, visited: &mut HashSet<*const NodeData>, sorted: &mut Vec<Node>) {
        if !visited.insert(node.ptr()) {
            return;
        }
        for child in node.src() {
            self.visit(child, visited, sorted);
        }
        sorted.push(node.clone());
    }
}
use crate::backend::node_renderer::NodeRenderer;
use crate::node::{Node, NodeData};
use log::{debug, trace};
use std::collections::{HashMap, HashSet};

/// Represents a language-agnostic instruction for code generation.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Declares a new variable. e.g., `float v0 = ...;`
    DeclareVariable {
        name: String,
        dtype: String,
        value: String,
    },
    /// A standalone statement that doesn't declare a new variable. e.g., `c[i] = v0;`
    Statement { code: String },
    /// A loop construct. e.g., `for (int i = 0; i < count; ++i) { ... }`
    Loop {
        count: String,
        body: Vec<Instruction>,
    },
    /// A return statement. e.g., `return v0;`
    Return { value: String },
}

/// A code generation engine that traverses a computation graph and produces a
/// language-agnostic list of `Instruction`s.
pub struct CodeGenerator<'a> {
    renderer: &'a dyn NodeRenderer,
    node_to_var: HashMap<*const NodeData, String>,
    instructions: Vec<Instruction>,
    var_counter: usize,
}

impl<'a> CodeGenerator<'a> {
    pub fn new(renderer: &'a dyn NodeRenderer) -> Self {
        Self {
            renderer,
            node_to_var: HashMap::new(),
            instructions: Vec::new(),
            var_counter: 0,
        }
    }

    fn new_var(&mut self) -> String {
        let name = format!("v{}", self.var_counter);
        self.var_counter += 1;
        name
    }

    /// Generates a list of abstract `Instruction`s for the given graph.
    pub fn generate(&mut self, root: &Node) -> Vec<Instruction> {
        debug!("Starting code generation for root node: {:?}", root.op());
        let sorted_nodes = self.topological_sort(root);
        trace!("Topological sort completed. Node count: {}", sorted_nodes.len());

        for node in &sorted_nodes {
            self.render_node(node);
        }

        // Add a return statement if the root operation produces a value.
        if !root.op().as_any().is::<crate::op::Store>() && !root.op().as_any().is::<crate::op::Loop>() {
            if let Some(result_var) = self.node_to_var.get(&root.ptr()) {
                trace!("Adding return statement for variable: {result_var}");
                self.instructions.push(Instruction::Return {
                    value: result_var.clone(),
                });
            }
        }
        
        debug!("Finished code generation. Total instructions: {}", self.instructions.len());
        std::mem::take(&mut self.instructions)
    }

    /// Renders a single node into one or more `Instruction`s.
    fn render_node(&mut self, node: &Node) {
        if self.node_to_var.contains_key(&node.ptr()) {
            return;
        }
        debug!("Rendering node: {:?}", node.op());

        let rendered_operands: Vec<String> = node
            .src()
            .iter()
            .map(|child| self.node_to_var.get(&child.ptr()).unwrap().clone())
            .collect();

        if let Some(expr) = self.renderer.render_op(node.op().as_ref(), &rendered_operands) {
            if node.op().as_any().is::<crate::op::Const>()
                || node.op().as_any().is::<crate::op::Variable>()
                || node.op().as_any().is::<crate::op::LoopVariable>()
                || node.op().as_any().is::<crate::op::Input>()
            {
                trace!("Node is a literal or input, assigning expression directly: {expr}");
                self.node_to_var.insert(node.ptr(), expr);
            } else {
                let var_name = self.new_var();
                let instruction = Instruction::DeclareVariable {
                    name: var_name.clone(),
                    dtype: "float".to_string(), // TODO: Handle types properly
                    value: expr,
                };
                trace!("Generated instruction: {instruction:?}");
                self.instructions.push(instruction);
                self.node_to_var.insert(node.ptr(), var_name);
            }
        } else if let Some(_store_op) = node.op().as_any().downcast_ref::<crate::op::Store>() {
            let buffer_var = self.node_to_var.get(&node.src()[0].ptr()).unwrap();
            let index_var = self.node_to_var.get(&node.src()[1].ptr()).unwrap();
            let value_var = self.node_to_var.get(&node.src()[2].ptr()).unwrap();
            let instruction = Instruction::Statement {
                code: format!("{buffer_var}[{index_var}] = {value_var}"),
            };
            trace!("Generated instruction: {instruction:?}");
            self.instructions.push(instruction);
        } else if let Some(loop_op) = node.op().as_any().downcast_ref::<crate::op::Loop>() {
            let count_var = self.node_to_var.get(&loop_op.count.ptr()).unwrap();
            
            debug!("Entering loop body generation for node: {loop_op:?}");
            let mut body_codegen = CodeGenerator::new(self.renderer);
            let body_instructions = body_codegen.generate(&loop_op.body);
            debug!("Finished loop body generation.");

            let instruction = Instruction::Loop {
                count: count_var.clone(),
                body: body_instructions,
            };
            trace!("Generated instruction: {instruction:?}");
            self.instructions.push(instruction);
            self.node_to_var.insert(node.ptr(), "loop_result".to_string());
        } else if let Some(fused_op) = node.op().as_fused_op() {
            debug!("Operator '{}' not supported directly, falling back.", fused_op.name());
            let fallback_node = fused_op.fallback(node.src());
            let var_name = self.render_fallback_node(&fallback_node);
            self.node_to_var.insert(node.ptr(), var_name);
        } else {
            panic!("Rendering not implemented for operator: {}", node.op().name());
        }
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
            || node.op().as_any().is::<crate::op::Input>()
        {
            expr
        } else {
            let var_name = self.new_var();
            self.instructions.push(Instruction::DeclareVariable {
                name: var_name.clone(),
                dtype: "float".to_string(),
                value: expr,
            });
            self.node_to_var.insert(node.ptr(), var_name.clone());
            var_name
        }
    }

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

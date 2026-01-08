//! Main lowering logic for converting GraphNode to AST
//!
//! This module implements the core lowering algorithm that converts
//! computation graph nodes into executable AST kernels.

use std::collections::HashMap;

use crate::ast::{AstNode, DType, Literal};
use crate::graph::{GraphNode, GraphOp, ReduceOp, collect_inputs, topological_sort};

use super::fusion::AllFusions;
use super::fusion::FusionPass;
use super::index_gen::IndexGenerator;
use super::loop_gen::LoopGenerator;

/// Main lowerer for converting computation graphs to AST
pub struct Lowerer {
    /// Counter for generating unique buffer names
    buffer_counter: usize,
    /// Mapping from graph nodes to buffer names
    buffer_map: HashMap<*const crate::graph::GraphInner, String>,
    /// Generated kernels
    kernels: Vec<AstNode>,
    /// Loop generator
    loop_gen: LoopGenerator,
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowerer {
    /// Create a new lowerer
    pub fn new() -> Self {
        Self {
            buffer_counter: 0,
            buffer_map: HashMap::new(),
            kernels: Vec::new(),
            loop_gen: LoopGenerator::new(),
        }
    }

    /// Get the index generator
    pub fn index_gen(&self) -> &IndexGenerator {
        self.loop_gen.index_gen()
    }

    /// Generate a new unique buffer name
    fn next_buffer_name(&mut self) -> String {
        let name = format!("buf{}", self.buffer_counter);
        self.buffer_counter += 1;
        name
    }

    /// Get or create buffer name for a node
    fn get_buffer_name(&mut self, node: &GraphNode) -> String {
        let ptr = std::rc::Rc::as_ptr(&node.0);
        if let Some(name) = self.buffer_map.get(&ptr) {
            return name.clone();
        }

        // Check if node has a user-provided name
        let name = if let Some(user_name) = node.name() {
            format!("{}_{}", user_name, self.buffer_counter)
        } else {
            self.next_buffer_name()
        };
        self.buffer_counter += 1;

        self.buffer_map.insert(ptr, name.clone());
        name
    }

    /// Lower a computation graph to a Program AST
    ///
    /// This is the main entry point. It:
    /// 1. Applies fusion passes
    /// 2. Topologically sorts nodes
    /// 3. Generates a kernel for each non-input node
    /// 4. Wraps everything in a Program node
    pub fn lower(&mut self, roots: &[GraphNode]) -> AstNode {
        // Apply fusion passes
        let fused = AllFusions.apply(roots);

        // Get topological order
        let sorted = topological_sort(&fused);

        // Identify external inputs
        let inputs = collect_inputs(&fused);
        for input in &inputs {
            self.get_buffer_name(input);
        }

        // Generate kernels for each non-input node
        for node in sorted {
            if !node.is_external() {
                let kernel = self.lower_node(&node);
                self.kernels.push(kernel);
            }
        }

        // Create Program node
        AstNode::Program {
            functions: self.kernels.clone(),
            execution_waves: vec![], // TODO: dependency analysis
        }
    }

    /// Lower a single graph node to a Kernel AST
    fn lower_node(&mut self, node: &GraphNode) -> AstNode {
        let output_buf = self.get_buffer_name(node);

        match node.op() {
            GraphOp::View(_view) => {
                // View operation: just copy with index transformation
                self.lower_view(node, &output_buf)
            }
            GraphOp::MapReduce { map, reduce } => {
                if let Some((reduce_op, axis)) = reduce {
                    self.lower_reduce(node, &output_buf, map, *reduce_op, *axis)
                } else {
                    self.lower_elementwise(node, &output_buf, map)
                }
            }
        }
    }

    /// Lower a View operation
    fn lower_view(&mut self, node: &GraphNode, output_buf: &str) -> AstNode {
        let shape = node.shape();
        let output_idx = self.index_gen().view_to_index(node.view());

        // Get source buffer and index
        let src = &node.sources()[0];
        let src_buf = self.get_buffer_name(src);
        let src_idx = self.index_gen().view_to_index(src.view());

        // Load from source, store to output
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Var(src_buf)),
            offset: Box::new(src_idx),
            count: 1,
            dtype: node.dtype().clone(),
        };

        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(output_idx),
            value: Box::new(load),
        };

        let body = self.loop_gen.generate_loops(&shape, store);

        self.make_kernel(output_buf, node.dtype(), body)
    }

    /// Lower an elementwise (MapReduce with reduce=None) operation
    fn lower_elementwise(&mut self, node: &GraphNode, output_buf: &str, map: &AstNode) -> AstNode {
        let shape = node.shape();
        let output_idx = self.index_gen().view_to_index(node.view());

        // Substitute Wildcards with Load operations
        let element_expr = self.substitute_wildcards(node, map);

        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var(output_buf.to_string())),
            offset: Box::new(output_idx),
            value: Box::new(element_expr),
        };

        let body = self.loop_gen.generate_loops(&shape, store);

        self.make_kernel(output_buf, node.dtype(), body)
    }

    /// Lower a reduction operation
    fn lower_reduce(
        &mut self,
        node: &GraphNode,
        output_buf: &str,
        map: &AstNode,
        reduce_op: ReduceOp,
        axis: usize,
    ) -> AstNode {
        // Use source shape for iteration (before reduction)
        let src = &node.sources()[0];
        let src_shape = src.shape();

        // Output index excludes the reduced dimension (simplified: just use ridx0)
        let output_idx = self.index_gen().view_to_index(node.view());

        // Identity value
        let identity = self.reduce_identity(&reduce_op, node.dtype());

        // Substitute Wildcards in map expression
        let load_expr = self.substitute_wildcards(node, map);

        // Combine expression: acc = reduce_op(acc, val)
        let acc_var = AstNode::Var("acc".to_string());
        let combine_expr = reduce_op.combine(acc_var, load_expr);

        // Generate reduce loop structure
        let body = self.loop_gen.generate_reduce(
            &src_shape,
            axis,
            AstNode::Var(output_buf.to_string()),
            output_idx,
            "acc",
            identity,
            combine_expr,
        );

        self.make_kernel(output_buf, node.dtype(), body)
    }

    /// Create a Kernel AST node
    fn make_kernel(&self, name: &str, dtype: &DType, body: AstNode) -> AstNode {
        let one = Box::new(AstNode::Const(Literal::I64(1)));
        AstNode::Kernel {
            name: Some(format!("kernel_{}", name)),
            params: vec![],
            return_type: dtype.clone(),
            body: Box::new(body),
            default_grid_size: [one.clone(), one.clone(), one.clone()],
            default_thread_group_size: [one.clone(), one.clone(), one],
        }
    }

    /// Substitute Wildcard nodes with Load operations
    fn substitute_wildcards(&mut self, node: &GraphNode, expr: &AstNode) -> AstNode {
        match expr {
            AstNode::Wildcard(id) => {
                // Parse the wildcard ID to get source index
                let src_idx: usize = id.parse().unwrap_or(0);
                if src_idx < node.sources().len() {
                    let src = &node.sources()[src_idx];
                    let src_buf = self.get_buffer_name(src);
                    let idx = self.index_gen().view_to_index(src.view());

                    AstNode::Load {
                        ptr: Box::new(AstNode::Var(src_buf)),
                        offset: Box::new(idx),
                        count: 1,
                        dtype: src.dtype().clone(),
                    }
                } else {
                    expr.clone()
                }
            }
            // Recursively substitute in compound expressions
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Max(a, b) => AstNode::Max(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Recip(a) => AstNode::Recip(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Sqrt(a) => AstNode::Sqrt(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Log2(a) => AstNode::Log2(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Exp2(a) => AstNode::Exp2(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Sin(a) => AstNode::Sin(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Floor(a) => AstNode::Floor(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Cast(a, dtype) => {
                AstNode::Cast(Box::new(self.substitute_wildcards(node, a)), dtype.clone())
            }
            AstNode::Lt(a, b) => AstNode::Lt(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::And(a, b) => AstNode::And(
                Box::new(self.substitute_wildcards(node, a)),
                Box::new(self.substitute_wildcards(node, b)),
            ),
            AstNode::Not(a) => AstNode::Not(Box::new(self.substitute_wildcards(node, a))),
            AstNode::Select {
                cond,
                then_val,
                else_val,
            } => AstNode::Select {
                cond: Box::new(self.substitute_wildcards(node, cond)),
                then_val: Box::new(self.substitute_wildcards(node, then_val)),
                else_val: Box::new(self.substitute_wildcards(node, else_val)),
            },
            // For other node types, return as-is
            _ => expr.clone(),
        }
    }

    /// Get the identity value for a reduction operation
    fn reduce_identity(&self, op: &ReduceOp, dtype: &DType) -> AstNode {
        let lit = op.identity(dtype);
        AstNode::Const(lit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Expr, input};

    #[test]
    fn test_lower_simple_elementwise() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("x");
        let y = (&x + &x).with_name("y");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[y]);

        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_lower_reduction() {
        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("x");
        let y = x.sum(1).with_name("y");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[y]);

        assert!(matches!(program, AstNode::Program { .. }));
    }

    #[test]
    fn test_lower_complex_graph() {
        let a = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("a");
        let b = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32).with_name("b");

        // (a + b).sum(1)
        let c = (&a + &b).sum(1).with_name("c");

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&[c]);

        assert!(matches!(program, AstNode::Program { functions, .. } if !functions.is_empty()));
    }
}

use crate::ast::{AstNode, DType, Function, Scope, VariableDecl};
use crate::graph::{ElementwiseOp, Graph, GraphNode, GraphNodeData, GraphOp};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

pub struct Lowerer {
    node_to_var: HashMap<*const GraphNodeData, AstNode>,
    temp_var_count: usize,
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

impl Lowerer {
    pub fn new() -> Self {
        Lowerer {
            node_to_var: HashMap::new(),
            temp_var_count: 0,
        }
    }

    fn new_temp_var(&mut self) -> AstNode {
        let name = format!("_t{}", self.temp_var_count);
        self.temp_var_count += 1;
        AstNode::Var(name)
    }

    pub fn lower(&mut self, graph: &Graph) -> Function {
        let sorted_nodes = self.topological_sort(graph);

        let mut declarations = vec![];
        let mut statements = vec![];
        let mut arguments = vec![];

        for node in &sorted_nodes {
            let (decl, stmt) = self.compile_node(node, &mut arguments);
            if let Some(d) = decl {
                declarations.push(d);
            }
            if let Some(s) = stmt {
                statements.push(s);
            }
        }

        let body = AstNode::Block {
            scope: Scope { declarations },
            statements,
        };

        Function::new("harp_kernel".to_string(), arguments, DType::Void, body)
    }

    fn compile_node(
        &mut self,
        node: &GraphNode,
        arguments: &mut Vec<(String, DType)>,
    ) -> (Option<VariableDecl>, Option<AstNode>) {
        let node_ptr = Rc::as_ptr(&node.0);
        if self.node_to_var.contains_key(&node_ptr) {
            return (None, None);
        }

        match &node.op {
            GraphOp::Input(_) => {
                let arg_var = self.new_temp_var();
                if let AstNode::Var(name) = &arg_var {
                    arguments.push((name.clone(), node.dtype.clone()));
                    self.node_to_var.insert(node_ptr, arg_var);
                }
                (None, None)
            }
            GraphOp::Elementwise(op) => {
                let output_var = self.new_temp_var();
                let lhs_var = self
                    .node_to_var
                    .get(&Rc::as_ptr(&node.src[0].0))
                    .expect("LHS not found")
                    .clone();

                let init_expr = if node.src.len() == 2 {
                    let rhs_var = self
                        .node_to_var
                        .get(&Rc::as_ptr(&node.src[1].0))
                        .expect("RHS not found")
                        .clone();
                    match op {
                        ElementwiseOp::Add => AstNode::Add(Box::new(lhs_var), Box::new(rhs_var)),
                        ElementwiseOp::Mul => AstNode::Mul(Box::new(lhs_var), Box::new(rhs_var)),
                        ElementwiseOp::Rem => AstNode::Rem(Box::new(lhs_var), Box::new(rhs_var)),
                        ElementwiseOp::Max => AstNode::Max(Box::new(lhs_var), Box::new(rhs_var)),
                        _ => todo!("Unsupported binary op: {:?}", op),
                    }
                } else {
                    match op {
                        ElementwiseOp::Neg => AstNode::Neg(Box::new(lhs_var)),
                        ElementwiseOp::Recip => AstNode::Recip(Box::new(lhs_var)),
                        ElementwiseOp::Sin => AstNode::Sin(Box::new(lhs_var)),
                        ElementwiseOp::Sqrt => AstNode::Sqrt(Box::new(lhs_var)),
                        ElementwiseOp::Log2 => AstNode::Log2(Box::new(lhs_var)),
                        ElementwiseOp::Exp2 => AstNode::Exp2(Box::new(lhs_var)),
                        _ => todo!("Unsupported unary op: {:?}", op),
                    }
                };

                self.node_to_var.insert(node_ptr, output_var.clone());

                if let AstNode::Var(name) = &output_var {
                    let decl = VariableDecl {
                        name: name.clone(),
                        dtype: node.dtype.clone(),
                        constant: false,
                    };
                    let assign = AstNode::Assign(Box::new(output_var), Box::new(init_expr));
                    (Some(decl), Some(assign))
                } else {
                    (None, None)
                }
            }
            _ => todo!("Unsupported op: {:?}", node.op),
        }
    }

    fn topological_sort(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut all_nodes = vec![];
        let mut visited = HashMap::new();
        for output in &graph.outputs {
            collect_nodes(output, &mut all_nodes, &mut visited);
        }

        let mut in_degree = HashMap::new();
        let mut graph_edges = HashMap::new();

        for node in &all_nodes {
            let node_ptr = Rc::as_ptr(&node.0);
            in_degree.entry(node_ptr).or_insert(0);
            for src_node in &node.src {
                let src_ptr = Rc::as_ptr(&src_node.0);
                in_degree.entry(node_ptr).and_modify(|d| *d += 1);
                graph_edges
                    .entry(src_ptr)
                    .or_insert_with(Vec::new)
                    .push(node.clone());
            }
        }

        let mut queue = VecDeque::new();
        for node in &all_nodes {
            if *in_degree.get(&Rc::as_ptr(&node.0)).unwrap() == 0 {
                queue.push_back(node.clone());
            }
        }

        let mut sorted_nodes = Vec::new();
        while let Some(node) = queue.pop_front() {
            sorted_nodes.push(node.clone());
            if let Some(successors) = graph_edges.get(&Rc::as_ptr(&node.0)) {
                for successor in successors {
                    let succ_ptr = Rc::as_ptr(&successor.0);
                    let degree = in_degree.get_mut(&succ_ptr).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(successor.clone());
                    }
                }
            }
        }

        sorted_nodes
    }
}

fn collect_nodes(
    node: &GraphNode,
    nodes: &mut Vec<GraphNode>,
    visited: &mut HashMap<*const GraphNodeData, bool>,
) {
    let node_ptr = Rc::as_ptr(&node.0);
    if visited.contains_key(&node_ptr) {
        return;
    }

    for src_node in &node.src {
        collect_nodes(src_node, nodes, visited);
    }

    visited.insert(node_ptr, true);
    nodes.push(node.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::shape::Expr;

    #[test]
    fn test_lowerer_simple_graph() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(10)]);
        let c = &a + &b;
        graph.output(c);

        let mut lowerer = Lowerer::new();
        let function = lowerer.lower(&graph);

        assert_eq!(function.name(), "harp_kernel");
        assert_eq!(function.arguments().len(), 2);
        assert_eq!(function.arguments()[0].0, "_t0");
        assert_eq!(function.arguments()[1].0, "_t1");

        let body = function.body();
        if let AstNode::Block { scope, statements } = body {
            assert_eq!(scope.declarations.len(), 1);
            assert_eq!(scope.declarations[0].name, "_t2");

            let statements = &statements;
            assert_eq!(statements.len(), 1);
            assert!(matches!(statements[0], AstNode::Assign(_, _)));
        } else {
            panic!("Function body is not a block");
        }
    }
}

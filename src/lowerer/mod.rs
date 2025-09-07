use crate::ast::{AstNode, ConstLiteral, DType, Function, Scope, VariableDecl};
use crate::graph::shape::Expr;
use crate::graph::{ElementwiseOp, Graph, GraphNode, GraphNodeData, GraphOp};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

pub struct Lowerer {
    node_to_var: HashMap<*const GraphNodeData, AstNode>,
    var_count: usize,
    ridx_count: usize,
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
            var_count: 0,
            ridx_count: 0,
        }
    }

    fn new_var(&mut self) -> AstNode {
        let name = format!("var{}", self.var_count);
        self.var_count += 1;
        AstNode::Var(name)
    }

    fn new_ridx(&mut self) -> String {
        let name = format!("ridx{}", self.ridx_count);
        self.ridx_count += 1;
        name
    }

    fn shape_to_dtype(&self, base_dtype: &DType, shape: &[Expr]) -> DType {
        let mut dtype = base_dtype.clone();
        for dim in shape.iter().rev() {
            if let Expr::Const(size) = dim {
                dtype = DType::Vec(Box::new(dtype), *size as usize);
            } else {
                // 動的な形状はまだサポートしない
                unimplemented!("Dynamic shapes are not yet supported in lowerer");
            }
        }
        dtype
    }

    fn expr_to_ast(&self, expr: &Expr) -> AstNode {
        if let Expr::Const(val) = expr {
            AstNode::Const(ConstLiteral::Usize(*val as usize))
        } else {
            unimplemented!("Dynamic dimension expressions are not yet supported in lowerer");
        }
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
                let arg_var = self.new_var();
                if let AstNode::Var(name) = &arg_var {
                    let arg_dtype = self.shape_to_dtype(&node.dtype, node.shape());
                    arguments.push((name.clone(), arg_dtype));
                    self.node_to_var.insert(node_ptr, arg_var);
                }
                (None, None)
            }
            GraphOp::Elementwise(op) => {
                let output_var = self.new_var();
                let lhs_var = self
                    .node_to_var
                    .get(&Rc::as_ptr(&node.src[0].0))
                    .expect("LHS not found")
                    .clone();

                // --- ループ構造とインデックスアクセスASTの生成 ---
                let mut loop_indices = vec![];
                let mut loop_nest = vec![];

                for dim in node.shape() {
                    let ridx_name = self.new_ridx();
                    loop_indices.push(AstNode::Var(ridx_name.clone()));
                    loop_nest.push((ridx_name, self.expr_to_ast(dim)));
                }

                let mut indexed_lhs = lhs_var.clone();
                for index in &loop_indices {
                    indexed_lhs = AstNode::Index {
                        target: Box::new(indexed_lhs),
                        index: Box::new(index.clone()),
                    };
                }

                let scalar_op = if node.src.len() == 2 {
                    let rhs_var = self
                        .node_to_var
                        .get(&Rc::as_ptr(&node.src[1].0))
                        .expect("RHS not found")
                        .clone();

                    let mut indexed_rhs = rhs_var.clone();
                    for index in &loop_indices {
                        indexed_rhs = AstNode::Index {
                            target: Box::new(indexed_rhs),
                            index: Box::new(index.clone()),
                        };
                    }

                    match op {
                        ElementwiseOp::Add => {
                            AstNode::Add(Box::new(indexed_lhs), Box::new(indexed_rhs))
                        }
                        ElementwiseOp::Mul => {
                            AstNode::Mul(Box::new(indexed_lhs), Box::new(indexed_rhs))
                        }
                        ElementwiseOp::Rem => {
                            AstNode::Rem(Box::new(indexed_lhs), Box::new(indexed_rhs))
                        }
                        ElementwiseOp::Max => {
                            AstNode::Max(Box::new(indexed_lhs), Box::new(indexed_rhs))
                        }
                        _ => todo!("Unsupported binary op: {:?}", op),
                    }
                } else {
                    match op {
                        ElementwiseOp::Neg => AstNode::Neg(Box::new(indexed_lhs)),
                        ElementwiseOp::Recip => AstNode::Recip(Box::new(indexed_lhs)),
                        ElementwiseOp::Sin => AstNode::Sin(Box::new(indexed_lhs)),
                        ElementwiseOp::Sqrt => AstNode::Sqrt(Box::new(indexed_lhs)),
                        ElementwiseOp::Log2 => AstNode::Log2(Box::new(indexed_lhs)),
                        ElementwiseOp::Exp2 => AstNode::Exp2(Box::new(indexed_lhs)),
                        _ => todo!("Unsupported unary op: {:?}", op),
                    }
                };

                let mut indexed_output = output_var.clone();
                for index in &loop_indices {
                    indexed_output = AstNode::Index {
                        target: Box::new(indexed_output),
                        index: Box::new(index.clone()),
                    };
                }

                let mut loop_body = AstNode::Assign(Box::new(indexed_output), Box::new(scalar_op));

                for (ridx_name, max) in loop_nest.into_iter().rev() {
                    loop_body = AstNode::Range {
                        counter_name: ridx_name,
                        max: Box::new(max),
                        body: Box::new(loop_body),
                    };
                }
                // --- ここまで ---

                self.node_to_var.insert(node_ptr, output_var.clone());

                if let AstNode::Var(name) = &output_var {
                    let decl = VariableDecl {
                        name: name.clone(),
                        dtype: self.shape_to_dtype(&node.dtype, node.shape()),
                        constant: false,
                    };
                    (Some(decl), Some(loop_body))
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
        assert_eq!(function.arguments()[0].0, "var0");
        assert_eq!(
            function.arguments()[0].1,
            DType::Vec(Box::new(DType::F32), 10)
        );
        assert_eq!(function.arguments()[1].0, "var1");
        assert_eq!(
            function.arguments()[1].1,
            DType::Vec(Box::new(DType::F32), 10)
        );

        if let AstNode::Block { scope, statements } = function.body() {
            assert_eq!(scope.declarations.len(), 1);
            assert_eq!(scope.declarations[0].name, "var2");
            assert_eq!(
                scope.declarations[0].dtype,
                DType::Vec(Box::new(DType::F32), 10)
            );

            assert_eq!(statements.len(), 1);
            if let AstNode::Range {
                counter_name,
                max,
                body,
            } = &statements[0]
            {
                assert_eq!(counter_name, "ridx0");
                assert_eq!(**max, AstNode::from(10usize));
                // Check body of the loop
                if let AstNode::Assign(lhs, rhs) = &**body {
                    // Check lhs: var2[ridx0]
                    assert!(matches!(&**lhs, AstNode::Index { .. }));
                    // Check rhs: var0[ridx0] + var1[ridx0]
                    assert!(matches!(&**rhs, AstNode::Add { .. }));
                } else {
                    panic!("Loop body should be an assignment");
                }
            } else {
                panic!("Statement should be a Range loop");
            }
        } else {
            panic!("Function body is not a block");
        }
    }
}

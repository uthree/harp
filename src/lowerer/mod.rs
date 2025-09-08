use crate::ast::{AstNode, DType, Function, Program, Scope, VariableDecl};
use crate::graph::shape::Expr;
use crate::graph::{ElementwiseOp, Graph, GraphNode, GraphNodeData, GraphOp};
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

pub struct Lowerer {
    node_to_var: HashMap<*const GraphNodeData, AstNode>,
    shape_var_map: HashMap<String, AstNode>,
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
            shape_var_map: HashMap::new(),
            var_count: 0,
            ridx_count: 0,
        }
    }

    fn new_var(&mut self) -> AstNode {
        let name = format!("buf{}", self.var_count);
        self.var_count += 1;
        AstNode::Var(name)
    }

    fn new_ridx(&mut self) -> String {
        let name = format!("ridx{}", self.ridx_count);
        self.ridx_count += 1;
        name
    }

    fn get_base_dtype(dtype: &DType) -> &DType {
        match dtype {
            DType::Ptr(inner) | DType::Vec(inner, _) => Self::get_base_dtype(inner),
            _ => dtype,
        }
    }

    fn shape_to_dtype(&self, base_dtype: &DType, shape: &[Expr]) -> DType {
        if shape.iter().any(|d| !matches!(d, Expr::Const(_))) {
            return DType::Ptr(Box::new(base_dtype.clone()));
        }
        let mut dtype = base_dtype.clone();
        for dim in shape.iter().rev() {
            if let Expr::Const(size) = dim {
                dtype = DType::Vec(Box::new(dtype), *size as usize);
            }
        }
        dtype
    }

    fn expr_to_ast(&self, expr: &Expr) -> AstNode {
        match expr {
            Expr::Const(val) => (*val as usize).into(),
            Expr::Var(name) => self
                .shape_var_map
                .get(name)
                .unwrap_or_else(|| panic!("Shape variable '{}' not found in map", name))
                .clone(),
            _ => unimplemented!("Complex expressions in shapes are not yet supported in lowerer"),
        }
    }

    pub fn lower(&mut self, graph: &Graph) -> Program {
        let mut impl_arguments = vec![];

        // 1. Register buffer inputs as arguments
        for node in &graph.inputs {
            let arg_var = self.new_var();
            if let AstNode::Var(name) = &arg_var {
                let arg_dtype = self.shape_to_dtype(&node.dtype, node.shape());
                impl_arguments.push((name.clone(), arg_dtype));
                self.node_to_var.insert(Rc::as_ptr(&node.0), arg_var);
            }
        }

        // 2. Register buffer outputs as arguments
        for node in &graph.outputs {
            let node_ptr = Rc::as_ptr(&node.0);
            if self.node_to_var.contains_key(&node_ptr) {
                continue;
            }
            let arg_var = self.new_var();
            if let AstNode::Var(name) = &arg_var {
                let arg_dtype = self.shape_to_dtype(&node.dtype, node.shape());
                impl_arguments.push((name.clone(), arg_dtype));
                self.node_to_var.insert(node_ptr, arg_var);
            }
        }

        // 3. Register shape variables as arguments and create the map
        let mut shape_var_args = vec![];
        for (i, var_sig) in graph.shape_variables.iter().enumerate() {
            let arg_name = format!("shape_var_{}", i);
            impl_arguments.push((arg_name.clone(), DType::Usize));
            self.shape_var_map
                .insert(var_sig.name.clone(), AstNode::Var(arg_name.clone()));
            shape_var_args.push(AstNode::Index {
                target: Box::new(AstNode::Var("shape_vars".to_string())),
                index: Box::new(i.into()),
            });
        }

        // 4. Generate kernel body statements
        let sorted_nodes = self.topological_sort(graph);
        let mut impl_declarations = vec![];
        let mut impl_statements = vec![];

        for node in &sorted_nodes {
            if let GraphOp::Input(_) = &node.op {
                continue;
            }
            let (decl, stmt) = self.lower_node(node);
            if let Some(d) = decl {
                impl_declarations.push(d);
            }
            if let Some(s) = stmt {
                impl_statements.push(s);
            }
        }

        let kernel_impl_body = AstNode::Block {
            scope: Scope {
                declarations: impl_declarations,
            },
            statements: impl_statements,
        };

        let kernel_impl = Function::new(
            "kernel_impl".to_string(),
            impl_arguments.clone(),
            DType::Void,
            kernel_impl_body,
        );

        // 5. Generate `kernel_main` wrapper function
        let mut main_declarations = vec![];
        let mut main_statements = vec![];
        let mut call_args = vec![];
        let num_buffer_args = impl_arguments.len() - shape_var_args.len();

        for (i, (arg_name, arg_dtype)) in impl_arguments.iter().take(num_buffer_args).enumerate() {
            let base_dtype = Self::get_base_dtype(arg_dtype);
            let pointer_dtype = DType::Ptr(Box::new(base_dtype.clone()));
            main_declarations.push(VariableDecl {
                name: arg_name.clone(),
                dtype: pointer_dtype.clone(),
                constant: false,
            });

            let cast_expr = AstNode::Cast {
                dtype: pointer_dtype.clone(),
                expr: Box::new(AstNode::Index {
                    target: Box::new(AstNode::Var("buffers".to_string())),
                    index: Box::new(AstNode::from(i)),
                }),
            };
            main_statements.push(AstNode::Assign(
                Box::new(AstNode::Var(arg_name.clone())),
                Box::new(cast_expr),
            ));
            call_args.push(AstNode::Var(arg_name.clone()));
        }
        call_args.extend(shape_var_args);

        main_statements.push(AstNode::CallFunction {
            name: "kernel_impl".to_string(),
            args: call_args,
        });

        let kernel_main_body = AstNode::Block {
            scope: Scope {
                declarations: main_declarations,
            },
            statements: main_statements,
        };

        let kernel_main = Function::new(
            "kernel_main".to_string(),
            vec![
                (
                    "buffers".to_string(),
                    DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
                ),
                ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize))),
            ],
            DType::Void,
            kernel_main_body,
        );

        Program {
            functions: vec![kernel_main, kernel_impl],
            entry_point: "kernel_main".to_string(),
        }
    }

    fn lower_node(&mut self, node: &GraphNode) -> (Option<VariableDecl>, Option<AstNode>) {
        match &node.op {
            GraphOp::Elementwise(op) => {
                let node_ptr = Rc::as_ptr(&node.0);
                let (output_var, needs_decl) =
                    if let Some(existing_var) = self.node_to_var.get(&node_ptr) {
                        (existing_var.clone(), false)
                    } else {
                        let new_var = self.new_var();
                        self.node_to_var.insert(node_ptr, new_var.clone());
                        (new_var, true)
                    };

                let lhs_var = self
                    .node_to_var
                    .get(&Rc::as_ptr(&node.src[0].0))
                    .expect("LHS not found")
                    .clone();

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

                let decl = if needs_decl {
                    if let AstNode::Var(name) = &output_var {
                        Some(VariableDecl {
                            name: name.clone(),
                            dtype: self.shape_to_dtype(&node.dtype, node.shape()),
                            constant: false,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                };
                (decl, Some(loop_body))
            }
            GraphOp::View => {
                let source_var = self
                    .node_to_var
                    .get(&Rc::as_ptr(&node.src[0].0))
                    .unwrap()
                    .clone();
                self.node_to_var.insert(Rc::as_ptr(&node.0), source_var);
                (None, None) // No declaration or statement needed
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
        let _ = env_logger::try_init();
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(10)]);
        let c = &a + &b;
        graph.output(c);

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&graph);

        assert_eq!(program.entry_point, "kernel_main");
        assert_eq!(program.functions.len(), 2);

        let kernel_impl = program
            .functions
            .iter()
            .find(|f| f.name() == "kernel_impl")
            .unwrap();

        assert_eq!(kernel_impl.arguments().len(), 3); // a, b, c
        assert_eq!(
            kernel_impl.arguments()[0].1,
            DType::Vec(Box::new(DType::F32), 10)
        );
    }

    #[test]
    fn test_lowerer_dynamic_shape_graph() {
        let _ = env_logger::try_init();
        let mut graph = Graph::new();
        let n = graph.shape_var("N", 128);
        let a = graph.input(DType::F32, vec![n.clone()]);
        let b = graph.input(DType::F32, vec![n.clone()]);
        let c = &a + &b;
        graph.output(c);

        let mut lowerer = Lowerer::new();
        let program = lowerer.lower(&graph);

        let kernel_impl = program
            .functions
            .iter()
            .find(|f| f.name() == "kernel_impl")
            .unwrap();

        assert_eq!(kernel_impl.arguments().len(), 4); // a, b, c, shape_var_0
        assert_eq!(
            kernel_impl.arguments()[0].1,
            DType::Ptr(Box::new(DType::F32))
        );
        assert_eq!(kernel_impl.arguments()[3].0, "shape_var_0");
        assert_eq!(kernel_impl.arguments()[3].1, DType::Usize);

        if let AstNode::Block { statements, .. } = kernel_impl.body() {
            assert_eq!(statements.len(), 1);
            if let AstNode::Range { max, .. } = &statements[0] {
                assert!(matches!(**max, AstNode::Var(ref v) if v == "shape_var_0"));
            } else {
                panic!("Expected a Range node");
            }
        } else {
            panic!("kernel_impl body is not a block");
        }
    }
}

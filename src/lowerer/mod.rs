use crate::ast::{AstNode, DType, Function, Scope, VariableDecl};
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::{HashMap, HashSet, VecDeque};

pub struct Lowerer {
    next_temp_id: usize,
    node_to_var: HashMap<GraphNode, String>,
}

impl Lowerer {
    pub fn new() -> Self {
        Self {
            next_temp_id: 0,
            node_to_var: HashMap::new(),
        }
    }

    pub fn lower(&mut self, graph: &Graph) -> Function {
        // 1. トポロジカルソート
        let sorted_nodes = self.topological_sort(graph);

        // 2. 各ノードを処理してAST文を生成
        let mut statements = Vec::new();
        let mut declarations = Vec::new();

        for node in sorted_nodes {
            let ast_stmt = self.lower_node(&node, &mut declarations);
            if let Some(stmt) = ast_stmt {
                statements.push(stmt);
            }
        }

        // 3. 入力パラメータを作成
        let arguments = self.create_input_arguments(graph);

        // 4. Function構造体を構築
        Function::new(
            "lowered_function".to_string(),
            arguments,
            DType::Void,
            AstNode::Block {
                scope: Scope { declarations },
                statements,
            },
        )
    }

    fn topological_sort(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut in_degree: HashMap<GraphNode, usize> = HashMap::new();
        let mut adjacency: HashMap<GraphNode, Vec<GraphNode>> = HashMap::new();
        let mut all_nodes = HashSet::new();

        // グラフを走査して依存関係を構築
        let mut queue = VecDeque::new();
        for output in &graph.outputs {
            queue.push_back(output.clone());
        }

        while let Some(node) = queue.pop_front() {
            if all_nodes.contains(&node) {
                continue;
            }
            all_nodes.insert(node.clone());

            let deps = self.get_dependencies(&node);
            in_degree.insert(node.clone(), deps.len());

            for dep in deps {
                adjacency
                    .entry(dep.clone())
                    .or_insert_with(Vec::new)
                    .push(node.clone());
                queue.push_back(dep);
            }
        }

        // トポロジカルソート実行
        let mut result = Vec::new();
        let mut zero_in_degree: VecDeque<_> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(node, _)| node.clone())
            .collect();

        while let Some(node) = zero_in_degree.pop_front() {
            result.push(node.clone());

            if let Some(neighbors) = adjacency.get(&node) {
                for neighbor in neighbors {
                    if let Some(degree) = in_degree.get_mut(neighbor) {
                        *degree -= 1;
                        if *degree == 0 {
                            zero_in_degree.push_back(neighbor.clone());
                        }
                    }
                }
            }
        }

        result
    }

    fn get_dependencies(&self, node: &GraphNode) -> Vec<GraphNode> {
        match &node.op {
            GraphOp::Input => vec![],
            GraphOp::Const(_) => vec![],
            GraphOp::Elementwise(op) => {
                use crate::graph::ops::ElementwiseOp;
                match op {
                    ElementwiseOp::Add(lhs, rhs)
                    | ElementwiseOp::Mul(lhs, rhs)
                    | ElementwiseOp::Max(lhs, rhs)
                    | ElementwiseOp::Mod(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
                    ElementwiseOp::Neg(n)
                    | ElementwiseOp::Recip(n)
                    | ElementwiseOp::Sin(n)
                    | ElementwiseOp::Sqrt(n)
                    | ElementwiseOp::Log2(n)
                    | ElementwiseOp::Exp2(n) => vec![n.clone()],
                }
            }
            GraphOp::Reduce(_, _) => {
                // Reduce操作は通常、入力テンソルからの依存関係を持つ
                // ここでは仮実装として空を返す
                // TODO: 実際の入力ノードの依存関係を追加
                vec![]
            }
            GraphOp::ViewTransform(n) => vec![n.clone()],
        }
    }

    fn lower_node(&mut self, node: &GraphNode, declarations: &mut Vec<VariableDecl>) -> Option<AstNode> {
        match &node.op {
            GraphOp::Input => {
                // 入力ノードは引数として処理されるので、ここでは何もしない
                None
            }
            GraphOp::Const(lit) => {
                let var_name = self.get_or_create_var_name(node);
                declarations.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: node.dtype.clone(),
                    constant: true,
                });
                Some(AstNode::Assign(
                    Box::new(AstNode::Var(var_name)),
                    Box::new(AstNode::Const(lit.clone())),
                ))
            }
            GraphOp::Elementwise(op) => {
                self.lower_elementwise_op(node, op, declarations)
            }
            GraphOp::Reduce(op, axis) => {
                self.lower_reduce_op(node, op, *axis, declarations)
            }
            GraphOp::ViewTransform(_) => {
                // View変換は通常、メモリレイアウトの変更なので、
                // 新しい変数は作らずに既存の変数への参照として扱う
                None
            }
        }
    }

    fn get_or_create_var_name(&mut self, node: &GraphNode) -> String {
        if let Some(name) = self.node_to_var.get(node) {
            name.clone()
        } else {
            let name = format!("temp{}", self.next_temp_id);
            self.next_temp_id += 1;
            self.node_to_var.insert(node.clone(), name.clone());
            name
        }
    }

    fn create_input_arguments(&self, graph: &Graph) -> Vec<(String, DType)> {
        graph
            .inputs
            .iter()
            .enumerate()
            .filter_map(|(i, weak_ref)| {
                weak_ref.upgrade().map(|node_data| {
                    (format!("input{}", i), node_data.dtype.clone())
                })
            })
            .collect()
    }

    fn lower_elementwise_op(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::ElementwiseOp,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        use crate::graph::ops::ElementwiseOp;

        let result_var = self.get_or_create_var_name(node);
        declarations.push(VariableDecl {
            name: result_var.clone(),
            dtype: node.dtype.clone(),
            constant: false,
        });

        // ループでテンソルの各要素を処理
        let body = match op {
            ElementwiseOp::Add(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(&result_var, &lhs_var, &rhs_var, |l, r| {
                    AstNode::Add(Box::new(l), Box::new(r))
                })
            }
            ElementwiseOp::Mul(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(&result_var, &lhs_var, &rhs_var, |l, r| {
                    AstNode::Mul(Box::new(l), Box::new(r))
                })
            }
            ElementwiseOp::Max(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(&result_var, &lhs_var, &rhs_var, |l, r| {
                    AstNode::Max(Box::new(l), Box::new(r))
                })
            }
            ElementwiseOp::Mod(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(&result_var, &lhs_var, &rhs_var, |l, r| {
                    AstNode::Rem(Box::new(l), Box::new(r))
                })
            }
            ElementwiseOp::Neg(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(&result_var, &operand_var, |x| {
                    AstNode::Neg(Box::new(x))
                })
            }
            ElementwiseOp::Recip(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(&result_var, &operand_var, |x| {
                    x.recip()
                })
            }
            ElementwiseOp::Sin(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(&result_var, &operand_var, |x| {
                    x.sin()
                })
            }
            ElementwiseOp::Sqrt(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(&result_var, &operand_var, |x| {
                    x.sqrt()
                })
            }
            ElementwiseOp::Log2(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(&result_var, &operand_var, |x| {
                    x.log2()
                })
            }
            ElementwiseOp::Exp2(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(&result_var, &operand_var, |x| {
                    x.exp2()
                })
            }
        };

        Some(body)
    }

    fn create_elementwise_loop<F>(&self, result_var: &str, lhs_var: &str, rhs_var: &str, op: F) -> AstNode
    where
        F: Fn(AstNode, AstNode) -> AstNode,
    {
        // 簡単な実装: フラットなインデックスでループ
        let loop_var = "i";
        let loop_body = AstNode::Assign(
            Box::new(AstNode::Index {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(AstNode::Var(loop_var.to_string())),
            }),
            Box::new(op(
                AstNode::Index {
                    target: Box::new(AstNode::Var(lhs_var.to_string())),
                    index: Box::new(AstNode::Var(loop_var.to_string())),
                },
                AstNode::Index {
                    target: Box::new(AstNode::Var(rhs_var.to_string())),
                    index: Box::new(AstNode::Var(loop_var.to_string())),
                },
            )),
        );

        // TODO: 実際のサイズを計算する必要がある
        // とりあえず仮のサイズとして100を使用
        AstNode::Range {
            counter_name: loop_var.to_string(),
            max: Box::new(AstNode::Const(crate::ast::ConstLiteral::Usize(100))),
            body: Box::new(loop_body),
        }
    }

    fn create_unary_elementwise_loop<F>(&self, result_var: &str, operand_var: &str, op: F) -> AstNode
    where
        F: Fn(AstNode) -> AstNode,
    {
        let loop_var = "i";
        let loop_body = AstNode::Assign(
            Box::new(AstNode::Index {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(AstNode::Var(loop_var.to_string())),
            }),
            Box::new(op(AstNode::Index {
                target: Box::new(AstNode::Var(operand_var.to_string())),
                index: Box::new(AstNode::Var(loop_var.to_string())),
            })),
        );

        AstNode::Range {
            counter_name: loop_var.to_string(),
            max: Box::new(AstNode::Const(crate::ast::ConstLiteral::Usize(100))),
            body: Box::new(loop_body),
        }
    }

    fn lower_reduce_op(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::ReduceOp,
        axis: usize,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        use crate::graph::ops::ReduceOp;

        let result_var = self.get_or_create_var_name(node);
        declarations.push(VariableDecl {
            name: result_var.clone(),
            dtype: node.dtype.clone(),
            constant: false,
        });

        // Reduce操作: 指定された軸に沿って削減
        let reduce_op = match op {
            ReduceOp::Add => |acc: AstNode, val: AstNode| -> AstNode { acc + val },
            ReduceOp::Mul => |acc: AstNode, val: AstNode| -> AstNode { acc * val },
            ReduceOp::Max => |acc: AstNode, val: AstNode| -> AstNode {
                AstNode::Max(Box::new(acc), Box::new(val))
            },
        };

        // 簡単な実装: ネストしたループを作成
        // TODO: 実際のshapeとaxisを使用してより正確なループを生成
        let loop_var = format!("reduce_{}", axis);
        let inner_loop_body = AstNode::Assign(
            Box::new(AstNode::Index {
                target: Box::new(AstNode::Var(result_var.clone())),
                index: Box::new(AstNode::Var("out_idx".to_string())),
            }),
            Box::new(reduce_op(
                AstNode::Index {
                    target: Box::new(AstNode::Var(result_var.clone())),
                    index: Box::new(AstNode::Var("out_idx".to_string())),
                },
                AstNode::Index {
                    target: Box::new(AstNode::Var("input_data".to_string())),
                    index: Box::new(AstNode::Var("in_idx".to_string())),
                },
            )),
        );

        Some(AstNode::Range {
            counter_name: loop_var,
            max: Box::new(AstNode::Const(crate::ast::ConstLiteral::Usize(10))), // 仮の値
            body: Box::new(inner_loop_body),
        })
    }
}

impl Default for Lowerer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;

    #[test]
    fn test_simple_constant() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 単純な定数のみのグラフ
        let constant_node = GraphNode::f32(1.0);
        graph.output(constant_node);

        // lower処理
        let function = lowerer.lower(&graph);

        // 基本的なチェック
        assert_eq!(function.name(), "lowered_function");
        assert_eq!(function.return_type(), &DType::Void);
        assert_eq!(function.arguments().len(), 0); // 定数のみなので引数なし
    }

    #[test]
    fn test_input_only() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 入力のみのグラフ
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into()]);
        graph.output(input_node);

        // lower処理
        let function = lowerer.lower(&graph);

        // 基本的なチェック
        assert_eq!(function.name(), "lowered_function");
        assert_eq!(function.return_type(), &DType::Void);
        assert_eq!(function.arguments().len(), 1);
        assert_eq!(function.arguments()[0].0, "input0");
        assert_eq!(function.arguments()[0].1, DType::F32);
    }

    #[test]
    fn test_elementwise_negation() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 単項演算: -constant
        let constant_node = GraphNode::f32(1.0);
        let negated = -constant_node;
        graph.output(negated);

        // lower処理
        let function = lowerer.lower(&graph);

        // 基本的なチェック
        assert_eq!(function.name(), "lowered_function");
        assert_eq!(function.return_type(), &DType::Void);
        // 定数+単項演算で2つの文が生成される想定
        if let AstNode::Block { statements, .. } = function.body() {
            assert_eq!(statements.len(), 2); // const assignment + neg loop
        } else {
            panic!("Expected Block body");
        }
    }
}

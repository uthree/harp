use crate::ast::{AstNode, ConstLiteral, DType, Function, Program, Scope, VariableDecl};
use crate::graph::{Graph, GraphNode, GraphOp};
use std::collections::{HashMap, HashSet, VecDeque};

mod cumulative;
mod elementwise;
mod fused;
mod reduce;
mod utils;

use cumulative::CumulativeLowerer;
use elementwise::ElementwiseLowerer;
use fused::FusedLowerer;
use reduce::ReduceLowerer;
use utils::LowererUtils;

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

    pub fn lower(&mut self, graph: &Graph) -> Program {
        let kernel_function = self.create_kernel_function(graph);
        let entry_function = self.create_entry_function(graph, &kernel_function);

        Program {
            functions: vec![kernel_function, entry_function],
            entry_point: "kernel_main".to_string(),
        }
    }

    fn create_kernel_function(&mut self, graph: &Graph) -> Function {
        // 0.5. graphのinputsの順序通りに入力ノードをマッピング
        for weak_input in graph.inputs.iter() {
            if let Some(_input_rc) = weak_input.upgrade() {
                // GraphNodeを作成するには、トポロジカルソートで得られたノードと照合する必要がある
                // しかし、ここではまだトポロジカルソートしていないので、一旦保留
            }
        }

        // 1. トポロジカルソート
        let sorted_nodes = self.topological_sort(graph);

        // 1.5. 入力ノードと出力ノードに対して変数名を事前マッピング
        // graphのinputsと照合して正しい順序を維持
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
                // sorted_nodesから同じノードを探す
                // GraphNodeはRc<GraphNodeData>をラップしており、Eq/Hashが実装されている
                let input_node = GraphNode::from_rc(input_rc);
                let var_name = format!("input_{}", i);
                self.node_to_var.insert(input_node, var_name);
            }
        }

        // 出力ノードもマッピング
        for (i, output_node) in graph.outputs.iter().enumerate() {
            let var_name = format!("output_{}", i);
            self.node_to_var.insert(output_node.clone(), var_name);
        }

        // 2. 各ノードを処理してAST文を生成
        let mut statements = Vec::new();
        let mut declarations = Vec::new();

        for node in sorted_nodes {
            let ast_stmt = self.lower_node(&node, &mut declarations);
            if let Some(stmt) = ast_stmt {
                statements.push(stmt);
            }
        }

        // 3. 出力ノードに対して、必要に応じてコピーコードを生成
        for (output_idx, output_node) in graph.outputs.iter().enumerate() {
            let output_var = format!("output_{}", output_idx);
            let source_var = self.get_or_create_var_name(output_node);

            eprintln!(
                "Output {}: output_var={}, source_var={}",
                output_idx, output_var, source_var
            );

            // 出力変数とソース変数が異なる場合、コピーが必要
            if output_var != source_var {
                // メモリコピーを生成
                let output_shape = &output_node.view;
                let crate::graph::shape::view::View::Linear {
                    shape,
                    strides,
                    offset,
                } = output_shape;
                // シンプルなコピーループを生成
                let copy_stmt = ReduceLowerer::create_copy_loop(
                    shape,
                    strides,
                    offset,
                    &source_var,
                    &output_var,
                    0,
                );
                statements.push(copy_stmt);
            }
        }

        // 3. 入力パラメータを作成（入力+出力）
        let arguments = self.create_kernel_arguments(graph);

        // 4. カーネル関数を構築
        Function::new(
            "kernel_impl".to_string(),
            arguments,
            DType::Void,
            AstNode::Block {
                scope: Scope { declarations },
                statements,
            },
        )
    }

    fn create_entry_function(&self, graph: &Graph, _kernel_function: &Function) -> Function {
        let mut statements = Vec::new();
        let mut local_vars = Vec::new();

        // バッファポインタから具体的な型付きポインタを取得
        let mut arg_index = 0;

        // 入力バッファの型キャスト
        for (i, weak_ref) in graph.inputs.iter().enumerate() {
            if let Some(node_data) = weak_ref.upgrade() {
                let var_name = format!("input_{}", i);
                let _cast_type = LowererUtils::get_c_type(&node_data.dtype);

                local_vars.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: DType::Ptr(Box::new(node_data.dtype.clone())),
                    constant: false,
                    size_expr: None,
                });

                // cast: float* input_0 = (float*)bufs[0];
                statements.push(AstNode::Assign(
                    var_name,
                    Box::new(AstNode::Cast {
                        dtype: DType::Ptr(Box::new(node_data.dtype.clone())),
                        expr: Box::new(AstNode::Deref(Box::new(
                            AstNode::Var("bufs".to_string())
                                + AstNode::Const(ConstLiteral::Usize(arg_index)),
                        ))),
                    }),
                ));
                arg_index += 1;
            }
        }

        // 出力バッファの型キャスト
        for (i, output_node) in graph.outputs.iter().enumerate() {
            let var_name = format!("output_{}", i);
            let _cast_type = LowererUtils::get_c_type(&output_node.dtype);

            local_vars.push(VariableDecl {
                name: var_name.clone(),
                dtype: DType::Ptr(Box::new(output_node.dtype.clone())),
                constant: false,
                size_expr: None,
            });

            statements.push(AstNode::Assign(
                var_name,
                Box::new(AstNode::Cast {
                    dtype: DType::Ptr(Box::new(output_node.dtype.clone())),
                    expr: Box::new(AstNode::Deref(Box::new(
                        AstNode::Var("bufs".to_string())
                            + AstNode::Const(ConstLiteral::Usize(arg_index)),
                    ))),
                }),
            ));
            arg_index += 1;
        }

        // カーネル関数呼び出し
        let mut call_args = Vec::new();

        // 入力引数
        for (i, _) in graph.inputs.iter().enumerate() {
            call_args.push(AstNode::Var(format!("input_{}", i)));
        }

        // 出力引数
        for (i, _) in graph.outputs.iter().enumerate() {
            call_args.push(AstNode::Var(format!("output_{}", i)));
        }

        statements.push(AstNode::CallFunction {
            name: "kernel_impl".to_string(),
            args: call_args,
        });

        Function::new(
            "kernel_main".to_string(),
            vec![
                (
                    "bufs".to_string(),
                    DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
                ),
                ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize))),
            ],
            DType::Void,
            AstNode::Block {
                scope: Scope {
                    declarations: local_vars,
                },
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
                adjacency.entry(dep.clone()).or_default().push(node.clone());
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
            GraphOp::Reduce(_, _, input) => vec![input.clone()],
            GraphOp::Cumulative(_, _, input) => vec![input.clone()],
            GraphOp::View(n) => vec![n.clone()],
            GraphOp::Contiguous(input) => vec![input.clone()],
            GraphOp::FusedElementwise(_, nodes) => nodes.clone(),
            GraphOp::FusedReduce(_, _, input) => vec![input.clone()],
            GraphOp::FusedElementwiseReduce(_, nodes, _, _) => nodes.clone(),
            GraphOp::FusedElementwiseCumulative(_, nodes, _) => nodes.clone(),
        }
    }

    fn lower_node(
        &mut self,
        node: &GraphNode,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        match &node.op {
            GraphOp::Input => {
                // 入力ノードは引数として処理される
                // get_or_create_var_nameで適切な名前が生成されるようにする
                self.get_or_create_var_name(node);
                None
            }
            GraphOp::Const(lit) => {
                let var_name = self.get_or_create_var_name(node);
                declarations.push(VariableDecl {
                    name: var_name.clone(),
                    dtype: node.dtype.clone(),
                    constant: false, // 現在は初期化と代入を分けているため、constにできない
                    size_expr: None,
                });
                Some(AstNode::Assign(
                    var_name,
                    Box::new(AstNode::Const(lit.clone())),
                ))
            }
            GraphOp::Elementwise(op) => ElementwiseLowerer::lower(
                node,
                op,
                |n| self.get_or_create_var_name(n),
                declarations,
            ),
            GraphOp::Reduce(op, axis, input) => ReduceLowerer::lower(
                node,
                op,
                *axis,
                input,
                |n| self.get_or_create_var_name(n),
                declarations,
            ),
            GraphOp::Cumulative(op, axis, input) => CumulativeLowerer::lower(
                node,
                op,
                *axis,
                input,
                |n| self.get_or_create_var_name(n),
                declarations,
            ),
            GraphOp::View(source_node) => {
                // Viewノードは単にview情報を変更するだけで、メモリコピーは不要
                // 変数名はsourceと同じものを使い、view情報（stride/offset）だけが変わる
                let source_var = self.get_or_create_var_name(source_node);

                // Viewノードの変数名をsourceと同じにする（コピー不要）
                self.node_to_var.insert(node.clone(), source_var);

                // コピーループは生成しない
                None
            }
            GraphOp::Contiguous(input) => {
                // Contiguous操作: 非連続なメモリレイアウトを連続に変換
                let result_var = self.get_or_create_var_name(node);
                let input_var = self.get_or_create_var_name(input);

                // 出力ノードの場合は配列を宣言しない
                if !result_var.starts_with("output_") {
                    let total_size = LowererUtils::compute_total_size(&node.view);
                    let result_dtype = if let Some(size) = total_size {
                        DType::Vec(Box::new(node.dtype.clone()), size)
                    } else {
                        todo!("Dynamic size arrays not yet supported")
                    };

                    declarations.push(VariableDecl {
                        name: result_var.clone(),
                        dtype: result_dtype,
                        constant: false,
                        size_expr: None,
                    });
                }

                // 入力のview（非連続の可能性あり）と出力のview（連続）を取得
                let input_view = &input.view;
                let result_view = &node.view;

                let (
                    crate::graph::shape::view::View::Linear {
                        shape,
                        strides: input_strides,
                        offset: input_offset,
                    },
                    crate::graph::shape::view::View::Linear {
                        strides: result_strides,
                        offset: result_offset,
                        ..
                    },
                ) = (input_view, result_view);

                // 入力から連続な出力へコピーするループを生成
                Some(Self::create_contiguous_copy_loop(
                    shape,
                    input_strides,
                    input_offset,
                    result_strides,
                    result_offset,
                    &input_var,
                    &result_var,
                    0,
                ))
            }
            GraphOp::FusedElementwise(ast, inputs) => {
                FusedLowerer::lower_fused_elementwise(node, ast, inputs, declarations, |n| {
                    self.get_or_create_var_name(n)
                })
            }
            GraphOp::FusedReduce(op, axes, input) => {
                FusedLowerer::lower_fused_reduce(node, op, axes, input, declarations, |n| {
                    self.get_or_create_var_name(n)
                })
            }
            GraphOp::FusedElementwiseReduce(ast, inputs, op, axes) => {
                FusedLowerer::lower_fused_elementwise_reduce(
                    node,
                    ast,
                    inputs,
                    op,
                    axes,
                    declarations,
                    |n| self.get_or_create_var_name(n),
                )
            }
            GraphOp::FusedElementwiseCumulative(ast, inputs, op) => {
                FusedLowerer::lower_fused_elementwise_cumulative(
                    node,
                    ast,
                    inputs,
                    op,
                    declarations,
                    |n| self.get_or_create_var_name(n),
                )
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

    /// Contiguous変換のためのコピーループを作成
    #[allow(clippy::too_many_arguments)]
    fn create_contiguous_copy_loop(
        shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        input_var: &str,
        result_var: &str,
        dim: usize,
    ) -> AstNode {
        if dim >= shape.len() {
            // 最内レベル: コピーを実行
            let input_index = LowererUtils::compute_memory_index(input_strides, input_offset, dim);
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Deref(Box::new(
                    AstNode::Var(input_var.to_string()) + input_index,
                ))),
            }
        } else {
            // ループを生成
            let loop_var = format!("i{}", dim);
            let inner_body = Self::create_contiguous_copy_loop(
                shape,
                input_strides,
                input_offset,
                result_strides,
                result_offset,
                input_var,
                result_var,
                dim + 1,
            );

            let shape_size = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(shape_size),
                body: Box::new(inner_body),
            }
        }
    }

    fn create_kernel_arguments(&self, graph: &Graph) -> Vec<(String, DType)> {
        let mut arguments = Vec::new();

        // 入力引数
        for (i, weak_ref) in graph.inputs.iter().enumerate() {
            if let Some(node_data) = weak_ref.upgrade() {
                arguments.push((
                    format!("input_{}", i),
                    DType::Ptr(Box::new(node_data.dtype.clone())),
                ));
            }
        }

        // 出力引数
        for (i, output_node) in graph.outputs.iter().enumerate() {
            arguments.push((
                format!("output_{}", i),
                DType::Ptr(Box::new(output_node.dtype.clone())),
            ));
        }

        arguments
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
        let program = lowerer.lower(&graph);

        // 基本的なチェック
        assert_eq!(program.entry_point, "kernel_main");
        assert_eq!(program.functions.len(), 2); // kernel_impl + kernel_main

        // エントリーポイント関数のチェック
        let entry_func = &program.functions[1];
        assert_eq!(entry_func.name(), "kernel_main");
        assert_eq!(entry_func.return_type(), &DType::Void);
        assert_eq!(entry_func.arguments().len(), 2); // bufs, shape_vars
    }

    #[test]
    fn test_input_only() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 入力のみのグラフ
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into()]);
        graph.output(input_node);

        // lower処理
        let program = lowerer.lower(&graph);

        // 基本的なチェック
        assert_eq!(program.entry_point, "kernel_main");
        assert_eq!(program.functions.len(), 2);

        // カーネル実装関数のチェック
        let kernel_func = &program.functions[0];
        assert_eq!(kernel_func.name(), "kernel_impl");
        assert_eq!(kernel_func.arguments().len(), 2); // input_0 + output_0
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
        let program = lowerer.lower(&graph);

        // 基本的なチェック
        assert_eq!(program.entry_point, "kernel_main");
        assert_eq!(program.functions.len(), 2);

        // カーネル実装関数のチェック
        let kernel_func = &program.functions[0];
        if let AstNode::Block { statements, .. } = kernel_func.body() {
            assert_eq!(statements.len(), 2); // const assignment + neg loop
        } else {
            panic!("Expected Block body");
        }
    }

    #[test]
    fn test_entry_point_structure() {
        let mut graph = Graph::new();
        let mut lowerer = Lowerer::new();

        // 入力と出力があるグラフ
        let input_node = graph.input(DType::F32, vec![4.into()]);
        let _constant = GraphNode::f32(2.0);
        let result = -input_node; // 単項演算
        graph.output(result);

        let program = lowerer.lower(&graph);

        // エントリーポイント関数の詳細チェック
        let entry_func = &program.functions[1];
        assert_eq!(entry_func.name(), "kernel_main");

        // 引数チェック: (void** bufs, size_t* shape_vars)
        let args = entry_func.arguments();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].0, "bufs");
        assert_eq!(args[1].0, "shape_vars");

        // エントリー関数の本体をチェック
        if let AstNode::Block { statements, scope } = entry_func.body() {
            // 入力と出力バッファの型キャストがある
            assert!(statements.len() >= 3); // 最低でも input cast + output cast + kernel call

            // 変数宣言をチェック
            assert!(scope.declarations.len() >= 2); // input_0, output_0

            // 最後の文はkernel_impl呼び出し
            if let AstNode::CallFunction { name, args } = statements.last().unwrap() {
                assert_eq!(name, "kernel_impl");
                assert_eq!(args.len(), 2); // input_0, output_0
            } else {
                panic!("Expected kernel call as last statement");
            }
        } else {
            panic!("Expected Block body in entry function");
        }
    }
}

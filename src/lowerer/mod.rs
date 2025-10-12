use crate::ast::{AstNode, ConstLiteral, DType, Program, Scope, VariableDecl};
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

    fn create_kernel_function(&mut self, graph: &Graph) -> AstNode {
        // 0.5. graphのinputsの順序通りに入力ノードをマッピング
        for weak_input in graph.inputs.iter() {
            if let Some(_input_rc) = weak_input.upgrade() {
                // GraphNodeを作成するには、トポロジカルソートで得られたノードと照合する必要がある
                // しかし、ここではまだトポロジカルソートしていないので、一旦保留
            }
        }

        // 1. トポロジカルソート（世代別）
        let generations = self.topological_sort_by_generation(graph);

        // 1.5. 入力ノードと出力ノードに対して変数名を事前マッピング
        // graphのinputsと照合して正しい順序を維持
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
                // generationsから同じノードを探す
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

        // 2. 各世代のノードを処理してAST文を生成
        // 世代間にBarrierを挿入
        let mut statements = Vec::new();
        let mut declarations = Vec::new();

        for (gen_idx, generation) in generations.iter().enumerate() {
            // 世代内の各ノードを処理
            for node in generation {
                let ast_stmt = self.lower_node(node, &mut declarations);
                if let Some(stmt) = ast_stmt {
                    statements.push(stmt);
                }
            }

            // 最後の世代でなければ、Barrierを挿入
            if gen_idx < generations.len() - 1 {
                statements.push(AstNode::Barrier);
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
        AstNode::function(
            "kernel_impl".to_string(),
            arguments,
            DType::Void,
            Scope { declarations },
            statements,
        )
    }

    fn create_entry_function(&self, graph: &Graph, _kernel_function: &AstNode) -> AstNode {
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

        AstNode::function(
            "kernel_main".to_string(),
            vec![
                (
                    "bufs".to_string(),
                    DType::Ptr(Box::new(DType::Ptr(Box::new(DType::Void)))),
                ),
                ("shape_vars".to_string(), DType::Ptr(Box::new(DType::Usize))),
            ],
            DType::Void,
            Scope {
                declarations: local_vars,
            },
            statements,
        )
    }

    /// トポロジカルソートを実行し、世代（レベル）ごとにノードをグループ化
    /// 各世代は並列実行可能なノードのグループを表す
    fn topological_sort_by_generation(&self, graph: &Graph) -> Vec<Vec<GraphNode>> {
        let mut in_degree: HashMap<GraphNode, usize> = HashMap::new();
        let mut adjacency: HashMap<GraphNode, Vec<GraphNode>> = HashMap::new();
        let mut all_nodes = HashSet::new();
        let mut node_level: HashMap<GraphNode, usize> = HashMap::new();

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

        // レベル（世代）を計算しながらトポロジカルソート実行
        let mut generations: Vec<Vec<GraphNode>> = Vec::new();
        let mut zero_in_degree: VecDeque<_> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(node, _)| node.clone())
            .collect();

        // 初期ノード（入力ノード等）のレベルを0に設定
        for node in &zero_in_degree {
            node_level.insert(node.clone(), 0);
        }

        while !zero_in_degree.is_empty() {
            // 現在の世代のノードを収集
            let current_generation: Vec<GraphNode> = zero_in_degree.drain(..).collect();

            // 次の世代の候補を収集
            let mut next_generation = Vec::new();

            for node in &current_generation {
                if let Some(neighbors) = adjacency.get(node) {
                    let current_level = *node_level.get(node).unwrap_or(&0);

                    for neighbor in neighbors {
                        if let Some(degree) = in_degree.get_mut(neighbor) {
                            *degree -= 1;

                            // ノードのレベルを更新（全ての依存元の最大レベル+1）
                            let neighbor_level = node_level.entry(neighbor.clone()).or_insert(0);
                            *neighbor_level = (*neighbor_level).max(current_level + 1);

                            if *degree == 0 {
                                next_generation.push(neighbor.clone());
                            }
                        }
                    }
                }
            }

            generations.push(current_generation);
            zero_in_degree = next_generation.into_iter().collect();
        }

        generations
    }

    fn get_dependencies(&self, node: &GraphNode) -> Vec<GraphNode> {
        match &node.op {
            GraphOp::Input(_) => vec![],
            GraphOp::Const(_) => vec![],
            GraphOp::Elementwise(op) => {
                use crate::graph::ops::ElementwiseOp;
                match op {
                    ElementwiseOp::Add(lhs, rhs)
                    | ElementwiseOp::Mul(lhs, rhs)
                    | ElementwiseOp::Max(lhs, rhs)
                    | ElementwiseOp::Mod(lhs, rhs)
                    | ElementwiseOp::LessThan(lhs, rhs)
                    | ElementwiseOp::Eq(lhs, rhs) => vec![lhs.clone(), rhs.clone()],
                    ElementwiseOp::Neg(n)
                    | ElementwiseOp::Recip(n)
                    | ElementwiseOp::Sin(n)
                    | ElementwiseOp::Sqrt(n)
                    | ElementwiseOp::Log2(n)
                    | ElementwiseOp::Exp2(n) => vec![n.clone()],
                    ElementwiseOp::Select(cond, true_val, false_val) => {
                        vec![cond.clone(), true_val.clone(), false_val.clone()]
                    }
                }
            }
            GraphOp::Reduce(_, _, input) => vec![input.clone()],
            GraphOp::Cumulative(_, _, input) => vec![input.clone()],
            GraphOp::View(n) => vec![n.clone()],
            GraphOp::Contiguous(input) => vec![input.clone()],
            GraphOp::Cast(input, _) => vec![input.clone()],
            GraphOp::Fold(_, _, _, _, input) => vec![input.clone()],
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
            GraphOp::Input(_) => {
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
                    let (result_dtype, size_expr) = if let Some(size) = total_size {
                        (DType::Vec(Box::new(node.dtype.clone()), size), None)
                    } else {
                        let size_expr = LowererUtils::compute_total_size_expr(&node.view);
                        (
                            DType::Ptr(Box::new(node.dtype.clone())),
                            Some(Box::new(size_expr)),
                        )
                    };

                    declarations.push(VariableDecl {
                        name: result_var.clone(),
                        dtype: result_dtype,
                        constant: false,
                        size_expr,
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
            GraphOp::Cast(input, target_dtype) => {
                // Cast操作: 型変換
                // 注意: 同じ型へのキャストでも、出力ノードへのコピーが必要な場合があるため、
                // Viewのように変数を共有するのではなく、常にコピーループを生成する

                let result_var = self.get_or_create_var_name(node);
                let input_var = self.get_or_create_var_name(input);

                // 出力ノードの場合は配列を宣言しない
                if !result_var.starts_with("output_") {
                    let total_size = LowererUtils::compute_total_size(&node.view);
                    let (result_dtype, size_expr) = if let Some(size) = total_size {
                        (DType::Vec(Box::new(target_dtype.clone()), size), None)
                    } else {
                        let size_expr = LowererUtils::compute_total_size_expr(&node.view);
                        (
                            DType::Ptr(Box::new(target_dtype.clone())),
                            Some(Box::new(size_expr)),
                        )
                    };

                    declarations.push(VariableDecl {
                        name: result_var.clone(),
                        dtype: result_dtype,
                        constant: false,
                        size_expr,
                    });
                }

                // キャストループを生成
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

                Some(Self::create_cast_loop(
                    shape,
                    input_strides,
                    input_offset,
                    result_strides,
                    result_offset,
                    &input_var,
                    &result_var,
                    target_dtype,
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
            GraphOp::Fold(dim, window_size, stride, dilation, input) => {
                // Fold operation (col2im): combines overlapping windows
                // Input:  [..., L', K] where last dim is window dimension
                // Output: [..., L] where L = (L'-1)*stride + (K-1)*dilation + 1
                let result_var = self.get_or_create_var_name(node);
                let input_var = self.get_or_create_var_name(input);

                // Declare output array if needed
                if !result_var.starts_with("output_") {
                    let total_size = LowererUtils::compute_total_size(&node.view);
                    let (result_dtype, size_expr) = if let Some(size) = total_size {
                        (DType::Vec(Box::new(node.dtype.clone()), size), None)
                    } else {
                        let size_expr = LowererUtils::compute_total_size_expr(&node.view);
                        (
                            DType::Ptr(Box::new(node.dtype.clone())),
                            Some(Box::new(size_expr)),
                        )
                    };

                    declarations.push(VariableDecl {
                        name: result_var.clone(),
                        dtype: result_dtype,
                        constant: false,
                        size_expr,
                    });
                }

                let input_view = &input.view;
                let result_view = &node.view;

                let (
                    crate::graph::shape::view::View::Linear {
                        shape: input_shape,
                        strides: input_strides,
                        offset: input_offset,
                    },
                    crate::graph::shape::view::View::Linear {
                        shape: result_shape,
                        strides: result_strides,
                        offset: result_offset,
                    },
                ) = (input_view, result_view);

                // Generate fold loops: initialize to zero, then accumulate
                Some(Self::create_fold_loops(
                    input_shape,
                    input_strides,
                    input_offset,
                    result_shape,
                    result_strides,
                    result_offset,
                    *dim,
                    *window_size,
                    *stride,
                    *dilation,
                    &input_var,
                    &result_var,
                ))
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
            let loop_var = format!("ridx{}", dim);
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
                start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
            }
        }
    }

    /// Foldのためのループを作成 (col2im operation)
    #[allow(clippy::too_many_arguments)]
    fn create_fold_loops(
        input_shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        dim: usize,
        window_size: usize,
        stride: usize,
        dilation: usize,
        input_var: &str,
        result_var: &str,
    ) -> AstNode {
        // Phase 1: Initialize output to zero
        let init_loop =
            Self::create_fold_init_loop(result_shape, result_strides, result_offset, result_var, 0);

        // Phase 2: Accumulate values from input windows
        let accum_loop = Self::create_fold_accumulate_loop(
            input_shape,
            input_strides,
            input_offset,
            result_shape,
            result_strides,
            result_offset,
            dim,
            window_size,
            stride,
            dilation,
            input_var,
            result_var,
            0,
        );

        // Combine init and accumulate in a block
        AstNode::Block {
            scope: crate::ast::Scope {
                declarations: vec![],
            },
            statements: vec![init_loop, accum_loop],
        }
    }

    /// Initialize output buffer to zero for fold operation
    fn create_fold_init_loop(
        result_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        result_var: &str,
        dim: usize,
    ) -> AstNode {
        if dim >= result_shape.len() {
            // Initialize to zero
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);
            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Const(ConstLiteral::F32(0.0))),
            }
        } else {
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_fold_init_loop(
                result_shape,
                result_strides,
                result_offset,
                result_var,
                dim + 1,
            );
            let shape_size = LowererUtils::shape_expr_to_ast_node(&result_shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
            }
        }
    }

    /// Accumulate values from input windows into output for fold operation
    #[allow(clippy::too_many_arguments)]
    fn create_fold_accumulate_loop(
        input_shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        _result_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        fold_dim: usize,
        _window_size: usize,
        stride: usize,
        dilation: usize,
        input_var: &str,
        result_var: &str,
        current_dim: usize,
    ) -> AstNode {
        let window_dim = input_shape.len() - 1; // Last dimension is window dimension

        if current_dim >= input_shape.len() {
            // All dimensions processed: perform accumulation
            // input[..., i_fold_dim, i_window_dim] accumulates to
            // output[..., i_fold_dim * stride + i_window_dim * dilation]

            let input_index =
                LowererUtils::compute_memory_index(input_strides, input_offset, input_shape.len());

            // Compute result index with stride and dilation adjustment
            // For fold_dim: use i_fold_dim * stride + i_window_dim * dilation instead of i_fold_dim
            let result_index = Self::compute_fold_result_index(
                result_strides,
                result_offset,
                fold_dim,
                window_dim,
                stride,
                dilation,
                input_shape.len(),
            );

            // result[idx] += input[idx]
            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index.clone()),
                value: Box::new(AstNode::Add(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index,
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                )),
            }
        } else {
            // Generate loop for current dimension
            let loop_var = format!("ridx{}", current_dim);
            let inner_body = Self::create_fold_accumulate_loop(
                input_shape,
                input_strides,
                input_offset,
                _result_shape,
                result_strides,
                result_offset,
                fold_dim,
                _window_size,
                stride,
                dilation,
                input_var,
                result_var,
                current_dim + 1,
            );
            let shape_size = LowererUtils::shape_expr_to_ast_node(&input_shape[current_dim]);

            AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
            }
        }
    }

    /// Compute result index for fold operation
    /// Maps input[..., i_fold_dim, ..., i_window_dim] to output[..., i_fold_dim * stride + i_window_dim * dilation, ...]
    fn compute_fold_result_index(
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        fold_dim: usize,
        window_dim: usize,
        stride: usize,
        dilation: usize,
        num_input_dims: usize,
    ) -> AstNode {
        let mut index = LowererUtils::shape_expr_to_ast_node(result_offset);

        for dim in 0..num_input_dims {
            if dim == window_dim {
                // Skip window dimension (it's been folded into fold_dim)
                continue;
            }

            let loop_var = format!("ridx{}", dim);
            let result_dim = if dim > fold_dim { dim - 1 } else { dim };

            if dim == fold_dim {
                // For fold_dim: result_index = i_fold_dim * stride + i_window_dim * dilation
                let fold_index = AstNode::Add(
                    Box::new(AstNode::Mul(
                        Box::new(AstNode::Var(loop_var)),
                        Box::new(AstNode::Const(ConstLiteral::Isize(stride as isize))),
                    )),
                    Box::new(AstNode::Mul(
                        Box::new(AstNode::Var(format!("ridx{}", window_dim))),
                        Box::new(AstNode::Const(ConstLiteral::Isize(dilation as isize))),
                    )),
                );
                index += LowererUtils::shape_expr_to_ast_node(&result_strides[result_dim].clone())
                    * fold_index;
            } else {
                index += LowererUtils::shape_expr_to_ast_node(&result_strides[result_dim].clone())
                    * AstNode::Var(loop_var);
            }
        }

        index
    }

    /// Castのためのループを作成
    #[allow(clippy::too_many_arguments)]
    fn create_cast_loop(
        shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        input_var: &str,
        result_var: &str,
        _target_dtype: &DType,
        dim: usize,
    ) -> AstNode {
        if dim >= shape.len() {
            // 最内レベル: キャストを実行
            let input_index = LowererUtils::compute_memory_index(input_strides, input_offset, dim);
            let result_index =
                LowererUtils::compute_memory_index(result_strides, result_offset, dim);

            // Cast AstNodeを使用して型変換
            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(AstNode::Cast {
                    dtype: _target_dtype.clone(),
                    expr: Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                }),
            }
        } else {
            // ループを生成
            let loop_var = format!("ridx{}", dim);
            let inner_body = Self::create_cast_loop(
                shape,
                input_strides,
                input_offset,
                result_strides,
                result_offset,
                input_var,
                result_var,
                _target_dtype,
                dim + 1,
            );

            let shape_size = LowererUtils::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                max: Box::new(shape_size),
                step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                body: Box::new(inner_body),
                unroll: None,
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
        if let AstNode::Function {
            name,
            return_type,
            arguments,
            ..
        } = &program.functions[1]
        {
            assert_eq!(name, "kernel_main");
            assert_eq!(return_type, &DType::Void);
            assert_eq!(arguments.len(), 2); // bufs, shape_vars
        } else {
            panic!("Expected Function node");
        }
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
        if let AstNode::Function {
            name, arguments, ..
        } = &program.functions[0]
        {
            assert_eq!(name, "kernel_impl");
            assert_eq!(arguments.len(), 2); // input_0 + output_0
        } else {
            panic!("Expected Function node");
        }
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
        if let AstNode::Function { statements, .. } = &program.functions[0] {
            // const assignment + barrier + neg loop
            assert_eq!(statements.len(), 3);
            // 2番目のステートメントがBarrierであることを確認
            assert!(matches!(statements[1], AstNode::Barrier));
        } else {
            panic!("Expected Function node");
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
        if let AstNode::Function {
            name,
            arguments,
            scope,
            statements,
            ..
        } = &program.functions[1]
        {
            assert_eq!(name, "kernel_main");

            // 引数チェック: (void** bufs, size_t* shape_vars)
            assert_eq!(arguments.len(), 2);
            assert_eq!(arguments[0].0, "bufs");
            assert_eq!(arguments[1].0, "shape_vars");

            // エントリー関数の本体をチェック
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
            panic!("Expected Function node");
        }
    }
}

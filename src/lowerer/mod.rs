use crate::ast::{AstNode, ConstLiteral, DType, Function, Program, Scope, VariableDecl};
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
        for (i, weak_input) in graph.inputs.iter().enumerate() {
            if let Some(input_rc) = weak_input.upgrade() {
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
                if let crate::graph::shape::view::View::Linear {
                    shape,
                    strides,
                    offset,
                } = output_shape
                {
                    // シンプルなコピーループを生成
                    let copy_stmt =
                        self.create_copy_loop(shape, strides, offset, &source_var, &output_var, 0);
                    statements.push(copy_stmt);
                }
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
                let _cast_type = self.get_c_type(&node_data.dtype);

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
            let _cast_type = self.get_c_type(&output_node.dtype);

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
            GraphOp::Contiguous => vec![],
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
            GraphOp::Elementwise(op) => self.lower_elementwise_op(node, op, declarations),
            GraphOp::Reduce(op, axis, input) => {
                self.lower_reduce_op(node, op, *axis, input, declarations)
            }
            GraphOp::Cumulative(op, axis, input) => {
                self.lower_cumulative_op(node, op, *axis, input, declarations)
            }
            GraphOp::View(source_node) => {
                // View変換は通常、メモリレイアウトの変更なので、
                // 新しい変数は作らずに既存の変数への参照として扱う
                // ソースノードの変数名をこのノードにもマッピングする
                let source_var = self.get_or_create_var_name(source_node);
                self.node_to_var.insert(node.clone(), source_var);
                None
            }
            GraphOp::Contiguous => {
                // TODO: Implement contiguous memory layout conversion
                todo!("Contiguous operation not yet implemented in lowerer")
            }
            GraphOp::FusedElementwise(_, _) => {
                // TODO: Implement fused elementwise operations
                todo!("FusedElementwise operation not yet implemented in lowerer")
            }
            GraphOp::FusedReduce(_, _, _) => {
                // TODO: Implement fused reduce operations
                todo!("FusedReduce operation not yet implemented in lowerer")
            }
            GraphOp::FusedElementwiseReduce(_, _, _, _) => {
                // TODO: Implement fused elementwise-reduce operations
                todo!("FusedElementwiseReduce operation not yet implemented in lowerer")
            }
            GraphOp::FusedElementwiseCumulative(_, _, _) => {
                // TODO: Implement fused elementwise-cumulative operations
                todo!("FusedElementwiseCumulative operation not yet implemented in lowerer")
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

    fn get_c_type(&self, dtype: &DType) -> &'static str {
        match dtype {
            DType::F32 => "float",
            DType::Usize => "size_t",
            DType::Isize => "ssize_t",
            DType::Void => "void",
            DType::Ptr(_) => "void*",
            DType::Vec(_, _) => "void*", // ベクトル型も一旦void*として扱う
        }
    }

    fn compute_total_size(&self, view: &crate::graph::shape::view::View) -> Option<usize> {
        if let crate::graph::shape::view::View::Linear { shape, .. } = view {
            let mut total = 1;
            for dim in shape {
                if let crate::graph::shape::Expr::Const(n) = dim {
                    total *= *n as usize;
                } else {
                    return None; // 動的サイズの場合
                }
            }
            Some(total)
        } else {
            None
        }
    }

    fn lower_elementwise_op(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::ElementwiseOp,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        use crate::graph::ops::ElementwiseOp;

        let result_var = self.get_or_create_var_name(node);

        // 出力ノードの場合は配列を宣言しない（引数として渡される）
        if !result_var.starts_with("output_") {
            // テンソルの場合は配列として宣言する必要がある
            let total_size = self.compute_total_size(&node.view);
            let result_dtype = if let Some(size) = total_size {
                // サイズが静的に決定できる場合は固定サイズ配列型
                DType::Vec(Box::new(node.dtype.clone()), size)
            } else {
                // 動的サイズの場合はポインタ型（将来的にmallocで対応）
                todo!("Dynamic size arrays not yet supported")
            };

            declarations.push(VariableDecl {
                name: result_var.clone(),
                dtype: result_dtype,
                constant: false,
                size_expr: None,
            });
        }

        // ループでテンソルの各要素を処理
        let body = match op {
            ElementwiseOp::Add(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Add(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Mul(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Mul(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Max(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Max(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Mod(lhs, rhs) => {
                let lhs_var = self.get_or_create_var_name(lhs);
                let rhs_var = self.get_or_create_var_name(rhs);
                self.create_elementwise_loop(
                    node,
                    lhs,
                    rhs,
                    &result_var,
                    &lhs_var,
                    &rhs_var,
                    |l, r| AstNode::Rem(Box::new(l), Box::new(r)),
                )
            }
            ElementwiseOp::Neg(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    AstNode::Neg(Box::new(x))
                })
            }
            ElementwiseOp::Recip(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.recip()
                })
            }
            ElementwiseOp::Sin(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.sin()
                })
            }
            ElementwiseOp::Sqrt(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.sqrt()
                })
            }
            ElementwiseOp::Log2(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.log2()
                })
            }
            ElementwiseOp::Exp2(operand) => {
                let operand_var = self.get_or_create_var_name(operand);
                self.create_unary_elementwise_loop(node, operand, &result_var, &operand_var, |x| {
                    x.exp2()
                })
            }
        };

        Some(body)
    }

    #[allow(clippy::too_many_arguments)]
    fn create_elementwise_loop<F>(
        &self,
        result_node: &GraphNode,
        lhs_node: &GraphNode,
        rhs_node: &GraphNode,
        result_var: &str,
        lhs_var: &str,
        rhs_var: &str,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode, AstNode) -> AstNode + Clone,
    {
        // viewから形状情報を取得
        let result_view = &result_node.view;
        let lhs_view = &lhs_node.view;
        let rhs_view = &rhs_node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: _result_shape,
                strides: result_strides,
                offset: result_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: lhs_strides,
                offset: lhs_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: rhs_strides,
                offset: rhs_offset,
            },
        ) = (result_view, lhs_view, rhs_view);

        // 多重ループを生成
        self.create_nested_loops(
            _result_shape,
            result_strides,
            result_offset,
            lhs_strides,
            lhs_offset,
            rhs_strides,
            rhs_offset,
            result_var,
            lhs_var,
            rhs_var,
            0,
            op,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_nested_loops<F>(
        &self,
        shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        lhs_strides: &[crate::graph::shape::Expr],
        lhs_offset: &crate::graph::shape::Expr,
        rhs_strides: &[crate::graph::shape::Expr],
        rhs_offset: &crate::graph::shape::Expr,
        result_var: &str,
        lhs_var: &str,
        rhs_var: &str,
        dim: usize,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode, AstNode) -> AstNode + Clone,
    {
        if dim >= shape.len() {
            // 最内ループ: 実際の計算を実行
            let result_index = self.compute_memory_index(result_strides, result_offset, dim);
            let lhs_index = self.compute_memory_index(lhs_strides, lhs_offset, dim);
            let rhs_index = self.compute_memory_index(rhs_strides, rhs_offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(op(
                    AstNode::Deref(Box::new(AstNode::Var(lhs_var.to_string()) + lhs_index)),
                    AstNode::Deref(Box::new(AstNode::Var(rhs_var.to_string()) + rhs_index)),
                )),
            }
        } else {
            // 再帰的にネストしたループを作成
            let loop_var = format!("i{}", dim);
            let inner_body = self.create_nested_loops(
                shape,
                result_strides,
                result_offset,
                lhs_strides,
                lhs_offset,
                rhs_strides,
                rhs_offset,
                result_var,
                lhs_var,
                rhs_var,
                dim + 1,
                op,
            );

            // shape[dim]をAstNodeに変換
            let max_iter = Self::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(max_iter),
                body: Box::new(inner_body),
            }
        }
    }

    fn compute_memory_index(
        &self,
        strides: &[crate::graph::shape::Expr],
        offset: &crate::graph::shape::Expr,
        num_dims: usize,
    ) -> AstNode {
        use crate::graph::shape::Expr;

        // Exprレベルで計算してからAstNodeに変換することで、simplifyが適用される
        let mut index_expr = offset.clone();

        for (i, stride) in strides.iter().enumerate().take(num_dims) {
            let loop_var = Expr::Var(format!("i{}", i));
            let term = loop_var * stride.clone();
            index_expr += term;
        }

        // 最終的にsimplifyしてからAstNodeに変換
        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
    }

    fn shape_expr_to_ast_node(expr: &crate::graph::shape::Expr) -> AstNode {
        use crate::graph::shape::Expr;
        match expr {
            Expr::Const(n) => AstNode::Const(crate::ast::ConstLiteral::Usize(*n as usize)),
            Expr::Var(name) => AstNode::Var(name.clone()),
            Expr::Add(left, right) => AstNode::Add(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(Self::shape_expr_to_ast_node(right)),
            ),
            Expr::Mul(left, right) => AstNode::Mul(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(Self::shape_expr_to_ast_node(right)),
            ),
            Expr::Div(left, right) => AstNode::Div(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(Self::shape_expr_to_ast_node(right)),
            ),
            Expr::Sub(left, right) => AstNode::Add(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(AstNode::Neg(Box::new(Self::shape_expr_to_ast_node(right)))),
            ),
            Expr::Rem(left, right) => AstNode::Rem(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(Self::shape_expr_to_ast_node(right)),
            ),
        }
    }

    fn create_unary_elementwise_loop<F>(
        &self,
        result_node: &GraphNode,
        operand_node: &GraphNode,
        result_var: &str,
        operand_var: &str,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode) -> AstNode + Clone,
    {
        // viewから形状情報を取得
        let result_view = &result_node.view;
        let operand_view = &operand_node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: _result_shape,
                strides: result_strides,
                offset: result_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _,
                strides: operand_strides,
                offset: operand_offset,
            },
        ) = (result_view, operand_view);

        // 多重ループを生成
        self.create_unary_nested_loops(
            _result_shape,
            result_strides,
            result_offset,
            operand_strides,
            operand_offset,
            result_var,
            operand_var,
            0,
            op,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn create_unary_nested_loops<F>(
        &self,
        shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        operand_strides: &[crate::graph::shape::Expr],
        operand_offset: &crate::graph::shape::Expr,
        result_var: &str,
        operand_var: &str,
        dim: usize,
        op: F,
    ) -> AstNode
    where
        F: Fn(AstNode) -> AstNode + Clone,
    {
        if dim >= shape.len() {
            // 最内ループ: 実際の計算を実行
            let result_index = self.compute_memory_index(result_strides, result_offset, dim);
            let operand_index = self.compute_memory_index(operand_strides, operand_offset, dim);

            AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(op(AstNode::Deref(Box::new(
                    AstNode::Var(operand_var.to_string()) + operand_index,
                )))),
            }
        } else {
            // 再帰的にネストしたループを作成
            let loop_var = format!("i{}", dim);
            let inner_body = self.create_unary_nested_loops(
                shape,
                result_strides,
                result_offset,
                operand_strides,
                operand_offset,
                result_var,
                operand_var,
                dim + 1,
                op,
            );

            // shape[dim]をAstNodeに変換
            let max_iter = Self::shape_expr_to_ast_node(&shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(max_iter),
                body: Box::new(inner_body),
            }
        }
    }

    fn lower_reduce_op(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::ReduceOp,
        axis: usize,
        input: &GraphNode,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        use crate::graph::ops::ReduceOp;

        let result_var = self.get_or_create_var_name(node);
        let input_var = self.get_or_create_var_name(input);

        // 出力ノードの場合は配列を宣言しない（引数として渡される）
        if !result_var.starts_with("output_") {
            // テンソルの場合は配列として宣言する必要がある
            let total_size = self.compute_total_size(&node.view);
            let result_dtype = if let Some(size) = total_size {
                // サイズが静的に決定できる場合は固定サイズ配列型
                DType::Vec(Box::new(node.dtype.clone()), size)
            } else {
                // 動的サイズの場合はポインタ型（将来的にmallocで対応）
                todo!("Dynamic size arrays not yet supported")
            };

            declarations.push(VariableDecl {
                name: result_var.clone(),
                dtype: result_dtype,
                constant: false,
                size_expr: None,
            });
        }

        // view情報を取得
        let input_view = &input.view;
        let result_view = &node.view;

        let (
            crate::graph::shape::view::View::Linear {
                shape: input_shape,
                strides: input_strides,
                offset: input_offset,
            },
            crate::graph::shape::view::View::Linear {
                shape: _result_shape,
                strides: result_strides,
                offset: result_offset,
            },
        ) = (input_view, result_view);

        // 縮約操作の初期値を定義
        let initial_value = match op {
            ReduceOp::Add => AstNode::Const(crate::ast::ConstLiteral::F32(0.0)),
            ReduceOp::Mul => AstNode::Const(crate::ast::ConstLiteral::F32(1.0)),
            ReduceOp::Max => AstNode::Const(crate::ast::ConstLiteral::F32(f32::NEG_INFINITY)),
        };

        // 多重ループでreduce操作を実行
        Some(self.create_reduce_loops(
            input_shape,
            input_strides,
            input_offset,
            _result_shape,
            result_strides,
            result_offset,
            axis,
            &input_var,
            &result_var,
            op,
            initial_value,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_reduce_loops(
        &self,
        input_shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        _result_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        reduce_axis: usize,
        input_var: &str,
        result_var: &str,
        reduce_op: &crate::graph::ops::ReduceOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        if dim >= input_shape.len() {
            // 全ての次元を処理した：縮約軸のループ本体を生成
            // この時点でdim == input_shape.len()なので、全てのループ変数が定義されている
            let input_index =
                self.compute_memory_index(input_strides, input_offset, input_shape.len());
            let result_index = self.compute_reduce_result_index(
                result_strides,
                result_offset,
                input_shape.len(),
                reduce_axis,
            );

            // 縮約操作: result[...] = result[...] op input[...]
            let operation_result = match reduce_op {
                crate::graph::ops::ReduceOp::Add => AstNode::Add(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
                crate::graph::ops::ReduceOp::Mul => AstNode::Mul(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
                crate::graph::ops::ReduceOp::Max => AstNode::Max(
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(result_var.to_string()) + result_index.clone(),
                    ))),
                    Box::new(AstNode::Deref(Box::new(
                        AstNode::Var(input_var.to_string()) + input_index,
                    ))),
                ),
            };

            return AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(operation_result),
            };
        }

        if dim == reduce_axis {
            // 縮約する次元: 初期化 + ループで累積
            // 縮約軸以降の次元のループを再帰的に生成
            let inner_body = self.create_reduce_loops(
                input_shape,
                input_strides,
                input_offset,
                _result_shape,
                result_strides,
                result_offset,
                reduce_axis,
                input_var,
                result_var,
                reduce_op,
                initial_value.clone(),
                dim + 1,
            );

            let loop_var = format!("i{}", dim);
            let shape_size = Self::shape_expr_to_ast_node(&input_shape[dim]);

            // 結果の初期化（縮約軸をスキップしたインデックスで計算）
            let result_index =
                self.compute_reduce_result_index(result_strides, result_offset, dim, reduce_axis);
            let init_stmt = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(initial_value),
            };

            // 縮約ループ: for (i_reduce) { inner_body }
            let reduce_loop = AstNode::Range {
                counter_name: loop_var,
                max: Box::new(shape_size),
                body: Box::new(inner_body),
            };

            // 初期化 + 縮約ループをブロックにまとめる
            AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![],
                },
                statements: vec![init_stmt, reduce_loop],
            }
        } else {
            // 縮約しない次元: 通常のループ
            let loop_var = format!("i{}", dim);
            let inner_body = self.create_reduce_loops(
                input_shape,
                input_strides,
                input_offset,
                _result_shape,
                result_strides,
                result_offset,
                reduce_axis,
                input_var,
                result_var,
                reduce_op,
                initial_value,
                dim + 1,
            );

            let shape_size = Self::shape_expr_to_ast_node(&input_shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(shape_size),
                body: Box::new(inner_body),
            }
        }
    }

    fn create_copy_loop(
        &self,
        shape: &[crate::graph::shape::Expr],
        strides: &[crate::graph::shape::Expr],
        offset: &crate::graph::shape::Expr,
        source_var: &str,
        dest_var: &str,
        dim: usize,
    ) -> AstNode {
        if dim >= shape.len() {
            // 最内レベル: コピーを実行
            let index = self.compute_memory_index(strides, offset, shape.len());
            return AstNode::Store {
                target: Box::new(AstNode::Var(dest_var.to_string())),
                index: Box::new(index.clone()),
                value: Box::new(AstNode::Deref(Box::new(
                    AstNode::Var(source_var.to_string()) + index,
                ))),
            };
        }

        // ループを生成
        let loop_var = format!("i{}", dim);
        let inner_body =
            self.create_copy_loop(shape, strides, offset, source_var, dest_var, dim + 1);

        let max_iter = Self::shape_expr_to_ast_node(&shape[dim]);

        AstNode::Range {
            counter_name: loop_var,
            max: Box::new(max_iter),
            body: Box::new(inner_body),
        }
    }

    fn lower_cumulative_op(
        &mut self,
        node: &GraphNode,
        op: &crate::graph::ops::CumulativeOp,
        axis: usize,
        input: &GraphNode,
        declarations: &mut Vec<VariableDecl>,
    ) -> Option<AstNode> {
        use crate::graph::ops::CumulativeOp;

        let result_var = self.get_or_create_var_name(node);
        let input_var = self.get_or_create_var_name(input);

        // 出力ノードの場合は配列を宣言しない（引数として渡される）
        if !result_var.starts_with("output_") {
            // テンソルの場合は配列として宣言する必要がある
            let total_size = self.compute_total_size(&node.view);
            let result_dtype = if let Some(size) = total_size {
                // サイズが静的に決定できる場合は固定サイズ配列型
                DType::Vec(Box::new(node.dtype.clone()), size)
            } else {
                // 動的サイズの場合はポインタ型（将来的にmallocで対応）
                todo!("Dynamic size arrays not yet supported")
            };

            declarations.push(VariableDecl {
                name: result_var.clone(),
                dtype: result_dtype,
                constant: false,
                size_expr: None,
            });
        }

        // view情報を取得
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

        // 累積演算のための初期値を定義
        let initial_value = match op {
            CumulativeOp::Add => AstNode::Const(crate::ast::ConstLiteral::F32(0.0)),
            CumulativeOp::Mul => AstNode::Const(crate::ast::ConstLiteral::F32(1.0)),
            CumulativeOp::Max => AstNode::Const(crate::ast::ConstLiteral::F32(f32::NEG_INFINITY)),
        };

        // 多重ループでcumulative操作を実行
        Some(self.create_cumulative_loops(
            input_shape,
            input_strides,
            input_offset,
            result_shape,
            result_strides,
            result_offset,
            axis,
            &input_var,
            &result_var,
            op,
            initial_value,
            0,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn create_cumulative_loops(
        &self,
        input_shape: &[crate::graph::shape::Expr],
        input_strides: &[crate::graph::shape::Expr],
        input_offset: &crate::graph::shape::Expr,
        result_shape: &[crate::graph::shape::Expr],
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        cumulative_axis: usize,
        input_var: &str,
        result_var: &str,
        cumulative_op: &crate::graph::ops::CumulativeOp,
        initial_value: AstNode,
        dim: usize,
    ) -> AstNode {
        use crate::graph::ops::CumulativeOp;

        if dim >= input_shape.len() {
            // 全ての次元を処理した：ここには到達しない（累積軸のループ内で処理される）
            unreachable!()
        } else if dim == cumulative_axis {
            // 累積軸: 特別な処理
            let loop_var = format!("i{}", dim);
            let shape_size = Self::shape_expr_to_ast_node(&input_shape[dim]);

            // 累積軸の最初の要素を初期化
            // result[..., 0, ...] = input[..., 0, ...]
            let first_input_index = {
                // i{axis} = 0 の状態でインデックスを計算
                let zero_ast = AstNode::from(0usize);

                // 一時的に i{axis} を 0 に設定してインデックスを計算
                self.compute_memory_index_with_override(
                    input_strides,
                    input_offset,
                    input_shape.len(),
                    cumulative_axis,
                    zero_ast.clone(),
                )
            };

            let first_result_index = {
                let zero_ast = AstNode::from(0usize);
                self.compute_memory_index_with_override(
                    result_strides,
                    result_offset,
                    result_shape.len(),
                    cumulative_axis,
                    zero_ast,
                )
            };

            let first_init = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(first_result_index),
                value: Box::new(AstNode::Deref(Box::new(
                    AstNode::Var(input_var.to_string()) + first_input_index,
                ))),
            };

            // 累積ループ: i = 1 から開始
            // result[..., i, ...] = result[..., i-1, ...] op input[..., i, ...]
            let input_index =
                self.compute_memory_index(input_strides, input_offset, input_shape.len());
            let result_index =
                self.compute_memory_index(result_strides, result_offset, result_shape.len());

            // 前の要素のインデックス（i-1）
            let prev_index = {
                let prev_offset = result_offset.clone()
                    + result_strides
                        .iter()
                        .enumerate()
                        .map(|(d, stride)| {
                            if d == cumulative_axis {
                                stride.clone() * crate::graph::shape::Expr::Var(format!("i{}", d))
                                    - stride.clone()
                            } else {
                                stride.clone() * crate::graph::shape::Expr::Var(format!("i{}", d))
                            }
                        })
                        .fold(crate::graph::shape::Expr::Const(0), |acc, x| acc + x);
                Self::shape_expr_to_ast_node(&prev_offset.simplify())
            };

            let prev_value =
                AstNode::Deref(Box::new(AstNode::Var(result_var.to_string()) + prev_index));
            let input_value =
                AstNode::Deref(Box::new(AstNode::Var(input_var.to_string()) + input_index));

            let cumulative_value = match cumulative_op {
                CumulativeOp::Add => prev_value + input_value,
                CumulativeOp::Mul => prev_value * input_value,
                CumulativeOp::Max => AstNode::Max(Box::new(prev_value), Box::new(input_value)),
            };

            let cumulative_stmt = AstNode::Store {
                target: Box::new(AstNode::Var(result_var.to_string())),
                index: Box::new(result_index),
                value: Box::new(cumulative_value),
            };

            // i=1から開始するループを作成
            // for (size_t i = 1; i < shape_size; i++)
            let cumulative_loop = AstNode::RangeFrom {
                counter_name: loop_var,
                start: Box::new(AstNode::from(1usize)),
                max: Box::new(shape_size),
                body: Box::new(cumulative_stmt),
            };

            // 初期化 + 累積ループをブロックにまとめる
            AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![],
                },
                statements: vec![first_init, cumulative_loop],
            }
        } else {
            // 累積軸以外の次元: 通常のループ
            let loop_var = format!("i{}", dim);
            let inner_body = self.create_cumulative_loops(
                input_shape,
                input_strides,
                input_offset,
                result_shape,
                result_strides,
                result_offset,
                cumulative_axis,
                input_var,
                result_var,
                cumulative_op,
                initial_value,
                dim + 1,
            );

            let shape_size = Self::shape_expr_to_ast_node(&input_shape[dim]);

            AstNode::Range {
                counter_name: loop_var,
                max: Box::new(shape_size),
                body: Box::new(inner_body),
            }
        }
    }

    fn compute_memory_index_with_override(
        &self,
        strides: &[crate::graph::shape::Expr],
        offset: &crate::graph::shape::Expr,
        ndim: usize,
        override_dim: usize,
        override_value: AstNode,
    ) -> AstNode {
        use crate::graph::shape::Expr;

        let mut index_expr = offset.clone();
        for d in 0..ndim {
            let loop_var = if d == override_dim {
                // オーバーライドされた値を使用
                match override_value {
                    AstNode::Const(ref lit) => match lit {
                        crate::ast::ConstLiteral::Usize(v) => Expr::Const(*v as isize),
                        crate::ast::ConstLiteral::Isize(v) => Expr::Const(*v),
                        _ => Expr::Var(format!("i{}", d)),
                    },
                    _ => Expr::Var(format!("i{}", d)),
                }
            } else {
                Expr::Var(format!("i{}", d))
            };
            let term = loop_var * strides[d].clone();
            index_expr += term;
        }

        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
    }

    fn compute_reduce_result_index(
        &self,
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        current_dim: usize,
        reduce_axis: usize,
    ) -> AstNode {
        use crate::graph::shape::Expr;

        // Exprレベルで計算してからAstNodeに変換することで、simplifyが適用される
        let mut index_expr = result_offset.clone();

        let mut result_dim = 0;
        for input_dim in 0..current_dim {
            if input_dim != reduce_axis {
                let loop_var = Expr::Var(format!("i{}", input_dim));
                let term = loop_var * result_strides[result_dim].clone();
                index_expr += term;
                result_dim += 1;
            }
        }

        // 最終的にsimplifyしてからAstNodeに変換
        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
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
        let constant = GraphNode::f32(2.0);
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

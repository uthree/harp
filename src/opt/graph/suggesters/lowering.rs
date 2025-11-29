//! Lowering Suggester
//!
//! GraphOpをCustomノードに変換するSuggester。
//! 各GraphOpに対して、対応するAstNode::Functionを生成し、
//! Custom演算として統合します。

use crate::ast::{AstNode, DType as AstDType, Mutability, Scope, helper::*};
use crate::graph::ops::custom_placeholders as ph;
use crate::graph::{
    CumulativeOp, DType as GraphDType, Graph, GraphNode, GraphNodeData, GraphOp, ReduceOp,
};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// GraphOpをCustomノードに変換するSuggester
///
/// 各計算ノードを等価なCustomノード（AstNode::Functionを保持）に変換します。
/// これにより、すべての計算がAST関数として統一され、
/// ASTレベルの最適化が可能になります。
pub struct LoweringSuggester;

impl LoweringSuggester {
    pub fn new() -> Self {
        LoweringSuggester
    }

    /// グラフ内の全ノードを収集（トポロジカル順）
    fn collect_all_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const GraphNodeData>,
            nodes: &mut Vec<GraphNode>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in &node.src {
                visit(src, visited, nodes);
            }

            nodes.push(node.clone());
        }

        for output in graph.outputs().values() {
            visit(output, &mut visited, &mut nodes);
        }

        nodes
    }

    /// ノードをCustomノードに変換可能かチェック
    fn can_lower(&self, node: &GraphNode) -> bool {
        // 基本的なノードタイプをチェック
        if matches!(
            node.op,
            GraphOp::Buffer { .. }
                | GraphOp::Const(_)
                | GraphOp::ComplexConst { .. }
                | GraphOp::View(_)
                | GraphOp::Custom { .. }
        ) {
            return false;
        }

        // Elementwise演算の場合、すべての入力が連続している必要がある
        // 非連続な入力（転置など）がある場合、正しいオフセット計算ができないためスキップ
        if matches!(node.op, GraphOp::Elementwise { .. }) {
            for src in &node.src {
                // すべての入力のViewが連続しているかチェック
                // (View経由でなくても、入力自体のviewが非連続な場合がある)
                if !src.view.is_contiguous() {
                    log::debug!(
                        "LoweringSuggester: skipping Elementwise due to non-contiguous input view"
                    );
                    return false;
                }
            }
        }

        true
    }

    /// GraphOpをCustomノードに変換
    fn lower_to_custom(&self, node: &GraphNode) -> Option<GraphNode> {
        let function = match &node.op {
            GraphOp::Elementwise { op, .. } => self.build_elementwise_function(node, op),
            GraphOp::Reduce { op, axis, .. } => self.build_reduce_function(node, op, *axis),
            GraphOp::Cumulative { op, axis, .. } => self.build_cumulative_function(node, op, *axis),
            GraphOp::Contiguous => self.build_contiguous_function(node),
            GraphOp::FusedElementwise { expr, .. } => {
                self.build_fused_elementwise_function(node, expr)
            }
            GraphOp::FusedElementwiseReduce {
                expr,
                reduce_op,
                axis,
                ..
            } => self.build_fused_elementwise_reduce_function(node, expr, reduce_op, *axis),
            GraphOp::FusedElementwiseCumulative {
                expr,
                cumulative_op,
                axis,
                ..
            } => self.build_fused_elementwise_cumulative_function(node, expr, cumulative_op, *axis),
            GraphOp::Pad { padding, value } => self.build_pad_function(node, padding, *value),
            GraphOp::Slice { ranges } => self.build_slice_function(node, ranges),
            GraphOp::Concat { axis } => self.build_concat_function(node, *axis),
            GraphOp::Rand => self.build_rand_function(node),
            GraphOp::Arange => self.build_arange_function(node),
            GraphOp::Cast { target_dtype, .. } => self.build_cast_function(node, target_dtype),
            GraphOp::Real => self.build_real_function(node),
            GraphOp::Imag => self.build_imag_function(node),
            GraphOp::ComplexFromParts => self.build_complex_from_parts_function(node),
            GraphOp::Fold { .. } => {
                // Foldは複雑なので後で実装
                return None;
            }
            GraphOp::FusedReduce { .. } => {
                // FusedReduceはタプル出力が必要なので後で実装
                return None;
            }
            _ => return None,
        }?;

        // Customノードを作成
        // 出力バッファーも含めたsrcを構築
        // 構造: [input0, input1, ..., output_buffer]
        // これにより、lowerer側でsrcとASTパラメータの対応が明確になる
        let mut new_src = node.src.clone();

        // 出力バッファーを作成
        // 名前は "output" とし、ASTの "output" パラメータに対応
        let output_buffer = GraphNode::new(
            node.dtype.clone(),
            GraphOp::Buffer {
                name: "output".to_string(),
            },
            vec![],
            node.view.clone(),
        );
        new_src.push(output_buffer);

        Some(GraphNode::new(
            node.dtype.clone(),
            GraphOp::Custom { ast: function },
            new_src,
            node.view.clone(),
        ))
    }

    /// グラフ内の特定ノードを置き換えた新しいグラフを作成
    fn replace_node_in_graph(
        &self,
        graph: &Graph,
        old_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
        node_map.insert(old_node.as_ptr(), new_node);

        let mut visited = HashSet::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            visited: &mut HashSet<*const GraphNodeData>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            if matches!(node.op, GraphOp::Buffer { .. }) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            if visited.contains(&ptr) {
                return node.clone();
            }
            visited.insert(ptr);

            let new_src: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, visited))
                .collect();

            let src_changed = new_src
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            if !src_changed {
                return node.clone();
            }

            GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
            )
        }

        let mut new_graph = Graph::new();

        for (name, weak_input) in graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                new_graph.register_input(name.clone(), input_node);
            }
        }

        let mut outputs: Vec<_> = graph.outputs().iter().collect();
        outputs.sort_by_key(|(name, _)| name.as_str());

        for (name, output_node) in outputs {
            let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
            new_graph.output(name, rebuilt);
        }

        new_graph
    }

    // ========================================================================
    // 関数ビルダー
    // ========================================================================

    /// Elementwise演算の関数を生成
    fn build_elementwise_function(
        &self,
        node: &GraphNode,
        op: &crate::graph::ops::ElementwiseOp,
    ) -> Option<AstNode> {
        use crate::graph::ops::ElementwiseOp;

        let shape = node.view.shape();
        let ndim = shape.len();

        // 演算式を構築
        let expr = match op {
            ElementwiseOp::Add => wildcard("0") + wildcard("1"),
            ElementwiseOp::Mul => wildcard("0") * wildcard("1"),
            ElementwiseOp::Neg => const_f32(-1.0) * wildcard("0"),
            ElementwiseOp::Max => max(wildcard("0"), wildcard("1")),
            ElementwiseOp::Rem => wildcard("0") % wildcard("1"),
            ElementwiseOp::Idiv => idiv(wildcard("0"), wildcard("1")),
            ElementwiseOp::Recip => recip(wildcard("0")),
            ElementwiseOp::Log2 => log2(wildcard("0")),
            ElementwiseOp::Exp2 => exp2(wildcard("0")),
            ElementwiseOp::Sin => sin(wildcard("0")),
            ElementwiseOp::Sqrt => sqrt(wildcard("0")),
        };

        // 入力数を計算（Constノードを除く）
        let num_inputs = node
            .src
            .iter()
            .filter(|s| !matches!(s.op, GraphOp::Const(_)))
            .count();

        // 定数を埋め込んだ式を構築
        let expr_with_consts = self.embed_constants(&expr, &node.src);

        Some(self.build_elementwise_function_impl(ndim, num_inputs, expr_with_consts, &node.dtype))
    }

    /// 定数ノードを式に埋め込む
    fn embed_constants(&self, expr: &AstNode, srcs: &[GraphNode]) -> AstNode {
        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;

        for (i, src) in srcs.iter().enumerate() {
            if let GraphOp::Const(lit) = &src.op {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            } else {
                // 非Constノードは元のインデックスを維持
                if non_const_idx != i {
                    mappings.insert(i.to_string(), wildcard(non_const_idx.to_string()));
                }
                non_const_idx += 1;
            }
        }

        if mappings.is_empty() {
            expr.clone()
        } else {
            expr.substitute(&mappings)
        }
    }

    /// Elementwise関数の実装
    fn build_elementwise_function_impl(
        &self,
        ndim: usize,
        num_inputs: usize,
        expr: AstNode,
        output_dtype: &GraphDType,
    ) -> AstNode {
        let offset = self.build_contiguous_offset(ndim);
        let load_dtype = self.graph_dtype_to_ast(output_dtype);

        // 入力のロードを含む式を構築
        let mut mappings = HashMap::new();
        for i in 0..num_inputs {
            let load_node = load(var(ph::input(i)), offset.clone(), load_dtype.clone());
            mappings.insert(i.to_string(), load_node);
        }
        let final_expr = expr.substitute(&mappings);

        let store_stmt = store(var(ph::OUTPUT), offset, final_expr);
        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        function(None::<String>, vec![], AstDType::Tuple(vec![]), body)
    }

    /// Reduce演算の関数を生成
    fn build_reduce_function(
        &self,
        node: &GraphNode,
        op: &ReduceOp,
        axis: usize,
    ) -> Option<AstNode> {
        let input = node.src.first()?;
        let input_shape = input.view.shape();
        let ndim = input_shape.len();

        let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
            match op {
                ReduceOp::Sum => (
                    self.get_reduce_init(&node.dtype, op),
                    Box::new(|acc, val| acc + val),
                ),
                ReduceOp::Prod => (
                    self.get_reduce_init(&node.dtype, op),
                    Box::new(|acc, val| acc * val),
                ),
                ReduceOp::Max => (self.get_reduce_init(&node.dtype, op), Box::new(max)),
            };

        let input_offset = self.build_contiguous_offset(ndim);
        let load_dtype = self.graph_dtype_to_ast(&input.dtype);
        let value_expr = load(var(ph::input(0)), input_offset, load_dtype);

        let output_offset = self.build_contiguous_offset_excluding_axis(ndim, axis);

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

        let reduce_loop = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            var(ph::shape(axis)),
            block(vec![acc_update], Scope::new()),
        );

        let mut scope = Scope::new();
        let _ = scope.declare(
            acc_var.to_string(),
            self.graph_dtype_to_ast(&node.dtype),
            Mutability::Mutable,
        );
        let acc_init = assign(acc_var, init_value);
        let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

        let inner_body = vec![acc_init, reduce_loop, store_stmt];
        let body = self.wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Cumulative演算の関数を生成
    fn build_cumulative_function(
        &self,
        node: &GraphNode,
        op: &CumulativeOp,
        axis: usize,
    ) -> Option<AstNode> {
        let input = node.src.first()?;
        let ndim = input.view.shape().len();

        let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
            match op {
                CumulativeOp::Sum => (const_f32(0.0), Box::new(|acc, val| acc + val)),
                CumulativeOp::Prod => (const_f32(1.0), Box::new(|acc, val| acc * val)),
            };

        let offset = self.build_contiguous_offset(ndim);
        let load_dtype = self.graph_dtype_to_ast(&input.dtype);
        let value_expr = load(var(ph::input(0)), offset.clone(), load_dtype);

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
        let store_stmt = store(var(ph::OUTPUT), offset, var(acc_var));

        let cum_loop = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            var(ph::shape(axis)),
            block(vec![acc_update, store_stmt], Scope::new()),
        );

        let mut scope = Scope::new();
        let _ = scope.declare(
            acc_var.to_string(),
            self.graph_dtype_to_ast(&node.dtype),
            Mutability::Mutable,
        );
        let acc_init = assign(acc_var, init_value);

        let inner_body = vec![acc_init, cum_loop];
        let body = self.wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Contiguous演算の関数を生成
    fn build_contiguous_function(&self, node: &GraphNode) -> Option<AstNode> {
        let input = node.src.first()?;
        let shape = node.view.shape();
        let ndim = shape.len();

        // 入力のオフセット計算（Viewを考慮）
        let input_offset = self.build_strided_offset(&input.view, ndim);
        let output_offset = self.build_contiguous_offset(ndim);

        let load_dtype = self.graph_dtype_to_ast(&input.dtype);
        let load_expr = load(var(ph::input(0)), input_offset, load_dtype);
        let store_stmt = store(var(ph::OUTPUT), output_offset, load_expr);

        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// FusedElementwise演算の関数を生成
    fn build_fused_elementwise_function(
        &self,
        node: &GraphNode,
        expr: &AstNode,
    ) -> Option<AstNode> {
        let shape = node.view.shape();
        let ndim = shape.len();
        let num_inputs = node
            .src
            .iter()
            .filter(|s| !matches!(s.op, GraphOp::Const(_)))
            .count();

        let expr_with_consts = self.embed_constants(expr, &node.src);

        Some(self.build_elementwise_function_impl(ndim, num_inputs, expr_with_consts, &node.dtype))
    }

    /// FusedElementwiseReduce演算の関数を生成
    fn build_fused_elementwise_reduce_function(
        &self,
        node: &GraphNode,
        expr: &AstNode,
        reduce_op: &ReduceOp,
        axis: usize,
    ) -> Option<AstNode> {
        let input = node.src.first()?;
        let input_shape = input.view.shape();
        let ndim = input_shape.len();
        let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
            match reduce_op {
                ReduceOp::Sum => (
                    self.get_reduce_init(&node.dtype, reduce_op),
                    Box::new(|acc, val| acc + val),
                ),
                ReduceOp::Prod => (
                    self.get_reduce_init(&node.dtype, reduce_op),
                    Box::new(|acc, val| acc * val),
                ),
                ReduceOp::Max => (self.get_reduce_init(&node.dtype, reduce_op), Box::new(max)),
            };

        // 入力のロードを含む式を構築
        let input_offset = self.build_contiguous_offset(ndim);
        let load_dtype = self.graph_dtype_to_ast(&input.dtype);

        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;
        for (i, src) in node.src.iter().enumerate() {
            if let GraphOp::Const(lit) = &src.op {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            } else {
                let load_node = load(
                    var(ph::input(non_const_idx)),
                    input_offset.clone(),
                    load_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                non_const_idx += 1;
            }
        }
        let value_expr = expr.substitute(&mappings);

        let output_offset = self.build_contiguous_offset_excluding_axis(ndim, axis);

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));

        let reduce_loop = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            var(ph::shape(axis)),
            block(vec![acc_update], Scope::new()),
        );

        let mut scope = Scope::new();
        let _ = scope.declare(
            acc_var.to_string(),
            self.graph_dtype_to_ast(&node.dtype),
            Mutability::Mutable,
        );
        let acc_init = assign(acc_var, init_value);
        let store_stmt = store(var(ph::OUTPUT), output_offset, var(acc_var));

        let inner_body = vec![acc_init, reduce_loop, store_stmt];
        let body = self.wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// FusedElementwiseCumulative演算の関数を生成
    fn build_fused_elementwise_cumulative_function(
        &self,
        node: &GraphNode,
        expr: &AstNode,
        cum_op: &CumulativeOp,
        axis: usize,
    ) -> Option<AstNode> {
        let input = node.src.first()?;
        let ndim = input.view.shape().len();

        let (init_value, accumulate_fn): (AstNode, Box<dyn Fn(AstNode, AstNode) -> AstNode>) =
            match cum_op {
                CumulativeOp::Sum => (const_f32(0.0), Box::new(|acc, val| acc + val)),
                CumulativeOp::Prod => (const_f32(1.0), Box::new(|acc, val| acc * val)),
            };

        let offset = self.build_contiguous_offset(ndim);
        let load_dtype = self.graph_dtype_to_ast(&input.dtype);

        let mut mappings = HashMap::new();
        let mut non_const_idx = 0;
        for (i, src) in node.src.iter().enumerate() {
            if let GraphOp::Const(lit) = &src.op {
                mappings.insert(i.to_string(), AstNode::Const(lit.clone()));
            } else {
                let load_node = load(
                    var(ph::input(non_const_idx)),
                    offset.clone(),
                    load_dtype.clone(),
                );
                mappings.insert(i.to_string(), load_node);
                non_const_idx += 1;
            }
        }
        let value_expr = expr.substitute(&mappings);

        let acc_var = "acc";
        let acc_update = assign(acc_var, accumulate_fn(var(acc_var), value_expr));
        let store_stmt = store(var(ph::OUTPUT), offset, var(acc_var));

        let cum_loop = range(
            ph::ridx(axis),
            const_int(0),
            const_int(1),
            var(ph::shape(axis)),
            block(vec![acc_update, store_stmt], Scope::new()),
        );

        let mut scope = Scope::new();
        let _ = scope.declare(
            acc_var.to_string(),
            self.graph_dtype_to_ast(&node.dtype),
            Mutability::Mutable,
        );
        let acc_init = assign(acc_var, init_value);

        let inner_body = vec![acc_init, cum_loop];
        let body = self.wrap_with_loops_excluding_axis_with_scope(ndim, axis, inner_body, scope);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Pad演算の関数を生成
    /// 注意: 現在のASTには条件式がないため、Padのloweringは未サポート
    fn build_pad_function(
        &self,
        _node: &GraphNode,
        _padding: &[(usize, usize)],
        _value: f32,
    ) -> Option<AstNode> {
        // TODO: ASTに条件式を追加したら実装
        None
    }

    /// Slice演算の関数を生成
    fn build_slice_function(&self, node: &GraphNode, ranges: &[(usize, usize)]) -> Option<AstNode> {
        let input = node.src.first()?;
        let shape = node.view.shape();
        let ndim = shape.len();

        // 出力のオフセット
        let output_offset = self.build_contiguous_offset(ndim);

        // 入力のオフセット（スライス開始位置を考慮）
        let mut input_offset_parts = Vec::new();
        for (axis, &(start, _)) in ranges.iter().enumerate().take(ndim) {
            let idx = var(ph::ridx(axis)) + const_int(start as isize);
            input_offset_parts.push(idx);
        }

        // ストライドを計算して入力オフセットを構築
        let input_shape = input.view.shape();
        let mut input_offset = input_offset_parts[ndim - 1].clone();
        for axis in (0..ndim - 1).rev() {
            let mut stride: AstNode = input_shape[axis + 1].clone().into();
            for dim in input_shape.iter().take(ndim).skip(axis + 2) {
                let s: AstNode = dim.clone().into();
                stride = stride * s;
            }
            input_offset = input_offset_parts[axis].clone() * stride + input_offset;
        }

        let load_dtype = self.graph_dtype_to_ast(&input.dtype);
        let load_expr = load(var(ph::input(0)), input_offset, load_dtype);
        let store_stmt = store(var(ph::OUTPUT), output_offset, load_expr);

        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Concat演算の関数を生成
    /// 注意: 現在のASTには条件式がないため、Concatのloweringは未サポート
    fn build_concat_function(&self, _node: &GraphNode, _axis: usize) -> Option<AstNode> {
        // TODO: ASTに条件式を追加したら実装
        None
    }

    /// Rand演算の関数を生成
    fn build_rand_function(&self, node: &GraphNode) -> Option<AstNode> {
        let shape = node.view.shape();
        let ndim = shape.len();

        let offset = self.build_contiguous_offset(ndim);

        // 簡易的な乱数生成（実際にはシード管理が必要）
        // ここではプレースホルダーとしてrand()呼び出しを生成
        let rand_expr = AstNode::Call {
            name: "rand_f32".to_string(),
            args: vec![],
        };

        let store_stmt = store(var(ph::OUTPUT), offset, rand_expr);
        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Arange演算の関数を生成
    fn build_arange_function(&self, node: &GraphNode) -> Option<AstNode> {
        let shape = node.view.shape();
        let ndim = shape.len();

        if ndim != 1 {
            return None; // Arangeは1次元のみ
        }

        let idx = var(ph::ridx(0));
        let offset = idx.clone();

        // 型に応じてキャスト
        let value = match &node.dtype {
            GraphDType::I32 => idx,
            GraphDType::F32 => cast(idx, AstDType::F32),
            _ => return None,
        };

        let store_stmt = store(var(ph::OUTPUT), offset, value);
        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Cast演算の関数を生成
    fn build_cast_function(&self, node: &GraphNode, target_dtype: &GraphDType) -> Option<AstNode> {
        let input = node.src.first()?;
        let shape = node.view.shape();
        let ndim = shape.len();

        let offset = self.build_contiguous_offset(ndim);
        let load_dtype = self.graph_dtype_to_ast(&input.dtype);
        let target_ast_dtype = self.graph_dtype_to_ast(target_dtype);

        let load_expr = load(var(ph::input(0)), offset.clone(), load_dtype);
        let cast_expr = cast(load_expr, target_ast_dtype);
        let store_stmt = store(var(ph::OUTPUT), offset, cast_expr);

        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Real演算の関数を生成
    fn build_real_function(&self, node: &GraphNode) -> Option<AstNode> {
        let _input = node.src.first()?;
        let shape = node.view.shape();
        let ndim = shape.len();

        // 複素数は2つのf32として格納されている
        let offset = self.build_contiguous_offset(ndim);
        let complex_offset = offset * const_int(2); // 実部は偶数インデックス

        let load_expr = load(var(ph::input(0)), complex_offset, AstDType::F32);
        let store_stmt = store(
            var(ph::OUTPUT),
            self.build_contiguous_offset(ndim),
            load_expr,
        );

        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// Imag演算の関数を生成
    fn build_imag_function(&self, node: &GraphNode) -> Option<AstNode> {
        let _input = node.src.first()?;
        let shape = node.view.shape();
        let ndim = shape.len();

        let offset = self.build_contiguous_offset(ndim);
        let complex_offset = offset * const_int(2) + const_int(1); // 虚部は奇数インデックス

        let load_expr = load(var(ph::input(0)), complex_offset, AstDType::F32);
        let store_stmt = store(
            var(ph::OUTPUT),
            self.build_contiguous_offset(ndim),
            load_expr,
        );

        let body = self.wrap_with_loops(ndim, vec![store_stmt]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    /// ComplexFromParts演算の関数を生成
    fn build_complex_from_parts_function(&self, node: &GraphNode) -> Option<AstNode> {
        if node.src.len() < 2 {
            return None;
        }

        let shape = node.view.shape();
        let ndim = shape.len();

        let offset = self.build_contiguous_offset(ndim);
        let complex_offset = offset.clone() * const_int(2);

        let real_load = load(var(ph::input(0)), offset.clone(), AstDType::F32);
        let imag_load = load(var(ph::input(1)), offset, AstDType::F32);

        let store_real = store(var(ph::OUTPUT), complex_offset.clone(), real_load);
        let store_imag = store(var(ph::OUTPUT), complex_offset + const_int(1), imag_load);

        let body = self.wrap_with_loops(ndim, vec![store_real, store_imag]);

        Some(function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            body,
        ))
    }

    // ========================================================================
    // ヘルパー関数
    // ========================================================================

    fn graph_dtype_to_ast(&self, dtype: &GraphDType) -> AstDType {
        match dtype {
            GraphDType::Bool => AstDType::Bool,
            GraphDType::I32 => AstDType::Int,
            GraphDType::F32 => AstDType::F32,
            GraphDType::Complex => AstDType::F32, // 複素数は2つのf32として扱う
            GraphDType::Unknown => AstDType::F32,
        }
    }

    fn get_reduce_init(&self, dtype: &GraphDType, op: &ReduceOp) -> AstNode {
        match op {
            ReduceOp::Sum => match dtype {
                GraphDType::Bool => AstNode::Const(false.into()),
                GraphDType::I32 => const_int(0),
                _ => const_f32(0.0),
            },
            ReduceOp::Prod => match dtype {
                GraphDType::Bool => AstNode::Const(true.into()),
                GraphDType::I32 => const_int(1),
                _ => const_f32(1.0),
            },
            ReduceOp::Max => match dtype {
                GraphDType::Bool => AstNode::Const(false.into()),
                GraphDType::I32 => const_int(i32::MIN as isize),
                _ => const_f32(f32::NEG_INFINITY),
            },
        }
    }

    fn build_contiguous_offset(&self, ndim: usize) -> AstNode {
        if ndim == 0 {
            return const_int(0);
        }

        let mut offset = var(ph::ridx(ndim - 1));

        for axis in (0..ndim - 1).rev() {
            let mut stride = var(ph::shape(axis + 1));
            for inner_axis in (axis + 2)..ndim {
                stride = stride * var(ph::shape(inner_axis));
            }
            offset = var(ph::ridx(axis)) * stride + offset;
        }

        offset
    }

    fn build_contiguous_offset_excluding_axis(&self, ndim: usize, exclude_axis: usize) -> AstNode {
        if ndim <= 1 {
            return const_int(0);
        }

        let output_ndim = ndim - 1;
        if output_ndim == 0 {
            return const_int(0);
        }

        let mut output_axes = Vec::new();
        for axis in 0..ndim {
            if axis != exclude_axis {
                output_axes.push(axis);
            }
        }

        let mut offset = var(ph::ridx(output_axes[output_ndim - 1]));

        for (out_axis, &in_axis) in output_axes.iter().enumerate().take(output_ndim - 1).rev() {
            let stride = if out_axis + 1 < output_axes.len() {
                let next_in_axis = output_axes[out_axis + 1];
                let mut s = var(ph::shape(next_in_axis));
                for &inner_in_axis in &output_axes[out_axis + 2..] {
                    s = s * var(ph::shape(inner_in_axis));
                }
                s
            } else {
                const_int(1)
            };

            offset = var(ph::ridx(in_axis)) * stride + offset;
        }

        offset
    }

    fn build_strided_offset(&self, view: &crate::graph::View, ndim: usize) -> AstNode {
        use crate::graph::View;

        if ndim == 0 {
            return const_int(0);
        }

        match view {
            View::Linear {
                strides, offset, ..
            } => {
                let mut result: AstNode = offset.clone().into();

                for (axis, stride_expr) in strides.iter().enumerate().take(ndim) {
                    let stride: AstNode = stride_expr.clone().into();
                    result = result + var(ph::ridx(axis)) * stride;
                }

                result
            }
        }
    }

    fn wrap_with_loops(&self, ndim: usize, inner_body: Vec<AstNode>) -> AstNode {
        if ndim == 0 {
            return block(inner_body, Scope::new());
        }

        let mut body = block(inner_body, Scope::new());

        for axis in (0..ndim).rev() {
            body = range(
                ph::ridx(axis),
                const_int(0),
                const_int(1),
                var(ph::shape(axis)),
                body,
            );
        }

        block(vec![body], Scope::new())
    }

    fn wrap_with_loops_excluding_axis_with_scope(
        &self,
        ndim: usize,
        exclude_axis: usize,
        inner_body: Vec<AstNode>,
        scope: Scope,
    ) -> AstNode {
        if ndim == 0 {
            return block(inner_body, scope);
        }

        let mut body = block(inner_body, scope);

        for axis in (0..ndim).rev() {
            if axis == exclude_axis {
                continue;
            }
            body = range(
                ph::ridx(axis),
                const_int(0),
                const_int(1),
                var(ph::shape(axis)),
                body,
            );
        }

        block(vec![body], Scope::new())
    }
}

impl Default for LoweringSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for LoweringSuggester {
    fn name(&self) -> &'static str {
        "Lowering"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        let mut lowerable_count = 0;
        let mut already_custom = 0;
        let mut lowered_count = 0;

        for node in &nodes {
            if matches!(node.op, GraphOp::Custom { .. }) {
                already_custom += 1;
                continue;
            }

            if !self.can_lower(node) {
                continue;
            }

            lowerable_count += 1;

            if let Some(custom_node) = self.lower_to_custom(node) {
                let new_graph = self.replace_node_in_graph(graph, node, custom_node);
                suggestions.push(new_graph);
                lowered_count += 1;
            } else {
                log::debug!(
                    "LoweringSuggester: failed to lower {:?}",
                    std::mem::discriminant(&node.op)
                );
            }
        }

        log::debug!(
            "LoweringSuggester: {} nodes total, {} already custom, {} lowerable, {} lowered",
            nodes.len(),
            already_custom,
            lowerable_count,
            lowered_count
        );

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType;

    #[test]
    fn test_lower_elementwise_add() {
        let suggester = LoweringSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        let suggestions = suggester.suggest(&graph);

        // Elementwise Addが1つあるので、1つの候補が生成される
        assert_eq!(suggestions.len(), 1);

        // 候補のグラフでCustomノードが使われていることを確認
        let new_graph = &suggestions[0];
        let output = new_graph.outputs().get("c").unwrap();
        assert!(matches!(output.op, GraphOp::Custom { .. }));
    }

    #[test]
    fn test_lower_reduce_sum() {
        let suggester = LoweringSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = a.reduce_sum(1);
        graph.output("b", b);

        let suggestions = suggester.suggest(&graph);

        assert_eq!(suggestions.len(), 1);

        let new_graph = &suggestions[0];
        let output = new_graph.outputs().get("b").unwrap();
        assert!(matches!(output.op, GraphOp::Custom { .. }));
    }

    #[test]
    fn test_skip_already_custom() {
        let suggester = LoweringSuggester::new();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);

        // 既にCustomノードを使用
        let custom_func = function(
            None::<String>,
            vec![],
            AstDType::Tuple(vec![]),
            block(vec![], Scope::new()),
        );
        let b = a.custom_function(custom_func);
        graph.output("b", b);

        let suggestions = suggester.suggest(&graph);

        // Customノードはスキップされるので候補なし
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_beam_search_with_lowering() {
        use crate::opt::graph::{
            BeamSearchGraphOptimizer, CompositeSuggester, GraphCostEstimator, SimpleCostEstimator,
        };

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = a + b;
        graph.output("c", c);

        // 初期コストを確認
        let estimator = SimpleCostEstimator::new();
        let initial_cost = estimator.estimate(&graph);
        println!("Initial cost: {}", initial_cost);

        // LoweringSuggesterのみでBeamSearch
        let composite = CompositeSuggester::new(vec![Box::new(LoweringSuggester::new())]);

        let optimizer = BeamSearchGraphOptimizer::new(composite, SimpleCostEstimator::new())
            .with_beam_width(4)
            .with_max_steps(10);

        let (optimized, history) = optimizer.optimize_with_history(graph);

        println!("Optimization steps: {}", history.len());
        for (i, snapshot) in history.snapshots().iter().enumerate() {
            println!("  Step {}: cost = {}", i, snapshot.cost);
        }

        // 最適化後のグラフを確認
        let output = optimized.outputs().get("c").unwrap();
        println!("Final output op: {:?}", std::mem::discriminant(&output.op));

        // Customノードに変換されているはず
        assert!(
            matches!(output.op, GraphOp::Custom { .. }),
            "Output should be Custom node, but got {:?}",
            output.op
        );
    }

    #[test]
    fn test_beam_search_with_fusion_and_lowering() {
        use crate::opt::graph::{
            BeamSearchGraphOptimizer, CompositeSuggester, FusionSuggester, SimpleCostEstimator,
        };

        // (a + b) * c + d というElementwiseチェーンを作成
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10, 20]);
        let b = graph.input("b", DType::F32, vec![10, 20]);
        let c = graph.input("c", DType::F32, vec![10, 20]);
        let d = graph.input("d", DType::F32, vec![10, 20]);

        let sum = a + b;
        let mul = sum * c;
        let result = mul + d;
        graph.output("result", result);

        // FusionとLoweringの両方を含むSuggester
        let suggesters: Vec<Box<dyn crate::opt::graph::GraphSuggester>> = vec![
            Box::new(FusionSuggester::new()),
            Box::new(LoweringSuggester::new()),
        ];
        let composite = CompositeSuggester::new(suggesters);

        let optimizer = BeamSearchGraphOptimizer::new(composite, SimpleCostEstimator::new())
            .with_beam_width(4)
            .with_max_steps(50);

        let (optimized, history) = optimizer.optimize_with_history(graph);

        println!("Optimization steps: {}", history.len());
        for (i, snapshot) in history.snapshots().iter().enumerate() {
            println!("  Step {}: cost = {}", i, snapshot.cost);
        }

        // 最適化後のグラフを確認
        let output = optimized.outputs().get("result").unwrap();
        println!("Final output op: {:?}", std::mem::discriminant(&output.op));
        println!("Final output src count: {}", output.src.len());

        // Customノードに変換されているはず
        assert!(
            matches!(output.op, GraphOp::Custom { .. }),
            "Output should be Custom node, but got {:?}",
            output.op
        );

        // 全ての入力が単一のCustomノードに融合されているはず (4入力 + 1出力Buffer = 5)
        // ただしBeamSearchの挙動により、部分的な融合になる可能性もある
        // 注: src = [input0, input1, ..., output_buffer] の構造
        assert!(
            output.src.len() <= 5,
            "Custom node should have at most 5 src nodes (4 inputs + 1 output buffer), but got {}",
            output.src.len()
        );
    }
}

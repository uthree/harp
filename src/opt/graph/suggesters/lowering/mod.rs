//! Lowering Suggester
//!
//! GraphOpをKernelノードに変換するSuggester。
//! 各GraphOpに対して、対応するAstNode::Functionを生成し、
//! Kernel演算として統合します。

mod elementwise;
mod helpers;
mod other;
mod parallel;
mod reduce;

use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

// サブモジュールから関数を再エクスポート
pub use elementwise::is_pure_const_node;
pub use parallel::ParallelizationStrategy;

/// GraphOpをKernelノードに変換するSuggester
///
/// 各計算ノードを等価なKernelノード（AstNode::Functionを保持）に変換します。
/// これにより、すべての計算がAST関数として統一され、
/// ASTレベルの最適化が可能になります。
///
/// # 並列化戦略
///
/// デフォルトでは複数の並列化戦略（Sequential, FlatParallel, MultiDimParallel）で
/// 候補を生成しますが、`sequential_only()`で逐次実行のみに制限できます。
/// これは実行時間の実測など、軽量なloweringが必要な場合に有用です。
///
/// # 設定可能なパラメータ
///
/// - `thread_group_sizes`: スレッドグループサイズの候補リスト
/// - `vector_widths`: ベクトル幅の候補リスト（空の場合はベクトル化無効）
pub struct LoweringSuggester {
    /// Sequentialのみを使用するかどうか
    sequential_only: bool,
    /// スレッドグループサイズの候補リスト
    thread_group_sizes: Vec<usize>,
    /// ベクトル幅の候補リスト（空ならベクトル化無効）
    vector_widths: Vec<usize>,
}

/// カーネル/関数の種類を表すプレフィックス
#[derive(Debug, Clone, Copy)]
enum KernelKind {
    /// Elementwise演算
    Elementwise,
    /// ElementwiseReduce演算 (FusedElementwiseReduceを含む)
    ElementwiseReduce,
    /// Cumulative演算 (FusedElementwiseCumulativeを含む)
    Cumulative,
    /// Reduce演算
    Reduce,
    /// その他の演算 (Contiguous, Cast, etc.)
    Other,
}

impl KernelKind {
    /// プレフィックス文字列を取得
    fn prefix(&self) -> &'static str {
        match self {
            KernelKind::Elementwise => "E",
            KernelKind::ElementwiseReduce => "ER",
            KernelKind::Cumulative => "C",
            KernelKind::Reduce => "R",
            KernelKind::Other => "O",
        }
    }
}

impl LoweringSuggester {
    /// 新しいLoweringSuggesterを作成
    ///
    /// デフォルトでは複数の並列化戦略で候補を生成します。
    /// - スレッドグループサイズ: [64, 128, 256, 512]
    /// - ベクトル幅: [2, 4, 8]
    pub fn new() -> Self {
        LoweringSuggester {
            sequential_only: false,
            thread_group_sizes: vec![64, 128, 256, 512],
            vector_widths: vec![2, 4, 8],
        }
    }

    /// Sequential戦略のみを使用するLoweringSuggesterを作成
    ///
    /// 並列化候補を生成せず、逐次実行のみでloweringします。
    /// 実行時間の実測など、軽量なloweringが必要な場合に使用します。
    ///
    /// # Example
    ///
    /// ```ignore
    /// let suggester = LoweringSuggester::sequential_only();
    /// ```
    pub fn sequential_only() -> Self {
        LoweringSuggester {
            sequential_only: true,
            thread_group_sizes: vec![],
            vector_widths: vec![],
        }
    }

    /// スレッドグループサイズの候補を設定
    ///
    /// # Example
    ///
    /// ```ignore
    /// let suggester = LoweringSuggester::new()
    ///     .with_thread_group_sizes(vec![128, 256]);
    /// ```
    pub fn with_thread_group_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.thread_group_sizes = sizes;
        self
    }

    /// ベクトル幅の候補を設定
    ///
    /// # Example
    ///
    /// ```ignore
    /// let suggester = LoweringSuggester::new()
    ///     .with_vector_widths(vec![4, 8]);
    /// ```
    pub fn with_vector_widths(mut self, widths: Vec<usize>) -> Self {
        self.vector_widths = widths;
        self
    }

    /// ベクトル化を無効にする
    ///
    /// # Example
    ///
    /// ```ignore
    /// let suggester = LoweringSuggester::new()
    ///     .without_vectorization();
    /// ```
    pub fn without_vectorization(mut self) -> Self {
        self.vector_widths = vec![];
        self
    }

    /// Sequential専用モードかどうかを返す
    pub fn is_sequential_only(&self) -> bool {
        self.sequential_only
    }

    /// ノードの種類とshapeからカーネル/関数名を生成
    ///
    /// 命名規則:
    /// - プレフィックス: E (Elementwise), ER (ElementwiseReduce), C (Cumulative), R (Reduce), O (Other)
    /// - 出力shape: `_`区切りで追加
    /// - 例: shape [2, 4] のElementwise演算 → `E_2_4`
    fn generate_kernel_name(
        &self,
        kind: KernelKind,
        shape: &[crate::graph::shape::Expr],
    ) -> String {
        let mut name = kind.prefix().to_string();

        for dim in shape {
            name.push('_');
            match dim {
                crate::graph::shape::Expr::Const(val) => {
                    name.push_str(&val.to_string());
                }
                crate::graph::shape::Expr::Var(var_name) => {
                    name.push_str(var_name);
                }
                _ => {
                    name.push_str("dyn");
                }
            }
        }

        name
    }

    /// ノードからカーネル種類を判定
    fn get_kernel_kind(&self, op: &GraphOp) -> KernelKind {
        match op {
            GraphOp::Elementwise { .. } | GraphOp::FusedElementwise { .. } => {
                KernelKind::Elementwise
            }
            GraphOp::FusedElementwiseReduce { .. } => KernelKind::ElementwiseReduce,
            GraphOp::Cumulative { .. } | GraphOp::FusedElementwiseCumulative { .. } => {
                KernelKind::Cumulative
            }
            GraphOp::Reduce { .. } => KernelKind::Reduce,
            _ => KernelKind::Other,
        }
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

    /// ノードをKernelノードに変換可能かチェック
    fn can_lower(&self, node: &GraphNode) -> bool {
        // 基本的なノードタイプをチェック
        if matches!(
            node.op,
            GraphOp::Buffer { .. }
                | GraphOp::Const(_)
                | GraphOp::ComplexConst { .. }
                | GraphOp::View(_)
                | GraphOp::Kernel { .. }
        ) {
            return false;
        }

        // Elementwise演算の場合、すべての入力が連続している必要がある
        if matches!(node.op, GraphOp::Elementwise { .. }) {
            for src in &node.src {
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

    /// GraphOpをKernelノードに変換（戦略指定版）
    fn lower_to_custom(
        &self,
        node: &GraphNode,
        strategy: &ParallelizationStrategy,
    ) -> Option<GraphNode> {
        let kind = self.get_kernel_kind(&node.op);
        let name = self.generate_kernel_name(kind, node.view.shape());

        let ast = match &node.op {
            GraphOp::Elementwise { op, .. } => {
                self.build_elementwise_ast(node, op, &name, strategy)
            }
            GraphOp::Reduce { op, axis, .. } => {
                self.build_reduce_ast(node, op, *axis, &name, strategy)
            }
            GraphOp::Cumulative { op, axis, .. } => {
                // Cumulativeは現時点では逐次のみサポート
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                reduce::build_cumulative_function(node, op, *axis, &name)
            }
            GraphOp::Contiguous => {
                // Contiguousは現時点では逐次のみサポート
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_contiguous_function(node, &name)
            }
            GraphOp::FusedElementwise { expr, .. } => {
                self.build_fused_elementwise_ast(node, expr, &name, strategy)
            }
            GraphOp::FusedElementwiseReduce {
                expr,
                reduce_op,
                axes,
                ..
            } => match strategy {
                ParallelizationStrategy::Sequential => {
                    reduce::build_fused_elementwise_reduce_function(
                        node, expr, reduce_op, axes, &name,
                    )
                }
                ParallelizationStrategy::FlatParallel {
                    thread_group_size: _,
                    vector_width: _,
                } => parallel::build_flat_parallel_fused_elementwise_reduce_kernel(
                    node, expr, reduce_op, axes, &name,
                ),
                ParallelizationStrategy::MultiDimParallel { .. } => None,
            },
            GraphOp::FusedElementwiseCumulative {
                expr,
                cumulative_op,
                axis,
                ..
            } => {
                // FusedElementwiseCumulativeは現時点では逐次のみサポート
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                reduce::build_fused_elementwise_cumulative_function(
                    node,
                    expr,
                    cumulative_op,
                    *axis,
                    &name,
                )
            }
            GraphOp::Pad { padding, value } => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_pad_function(node, padding, *value)
            }
            GraphOp::Slice { ranges } => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_slice_function(node, ranges, &name)
            }
            GraphOp::Concat { axis } => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_concat_function(node, *axis)
            }
            GraphOp::Rand => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_rand_function(node, &name)
            }
            GraphOp::Arange => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_arange_function(node, &name)
            }
            GraphOp::Cast { target_dtype, .. } => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_cast_function(node, target_dtype, &name)
            }
            GraphOp::Real => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_real_function(node, &name)
            }
            GraphOp::Imag => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_imag_function(node, &name)
            }
            GraphOp::ComplexFromParts => {
                if !matches!(strategy, ParallelizationStrategy::Sequential) {
                    return None;
                }
                other::build_complex_from_parts_function(node, &name)
            }
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

        // Kernelノードを作成
        // srcからView経由でInputまで辿り、対応するBufferノードを収集
        let non_const_src: Vec<_> = node
            .src
            .iter()
            .filter(|s| !matches!(s.op, GraphOp::Const(_)) && !is_pure_const_node(s))
            .cloned()
            .collect();
        let mut new_src = helpers::collect_input_buffers(&non_const_src);

        // 出力バッファーを作成
        let output_buffer_name = format!("output_{}", name);
        let output_buffer = GraphNode::new(
            node.dtype.clone(),
            GraphOp::Buffer {
                name: output_buffer_name,
            },
            vec![],
            node.view.clone(),
        );
        new_src.push(output_buffer);

        Some(GraphNode::new(
            node.dtype.clone(),
            GraphOp::Kernel {
                ast,
                input_buffers: None,
            },
            new_src,
            node.view.clone(),
        ))
    }

    /// Elementwise演算のAST生成（戦略に応じて分岐）
    fn build_elementwise_ast(
        &self,
        node: &GraphNode,
        op: &crate::graph::ElementwiseOp,
        name: &str,
        strategy: &ParallelizationStrategy,
    ) -> Option<crate::ast::AstNode> {
        match strategy {
            ParallelizationStrategy::Sequential => {
                elementwise::build_elementwise_function(node, op, name)
            }
            _ => {
                // 並列版: parallel モジュールを使用
                let shape = node.view.shape();
                let ndim = shape.len();
                let num_inputs = node
                    .src
                    .iter()
                    .filter(|s| !matches!(s.op, GraphOp::Const(_)) && !is_pure_const_node(s))
                    .count();

                let expr = elementwise::build_elementwise_expr(op);
                let expr_with_consts = elementwise::embed_constants(&expr, &node.src);

                Some(parallel::build_parallel_elementwise_kernel(
                    ndim,
                    num_inputs,
                    expr_with_consts,
                    &node.dtype,
                    name,
                    strategy,
                ))
            }
        }
    }

    /// FusedElementwise演算のAST生成（戦略に応じて分岐）
    fn build_fused_elementwise_ast(
        &self,
        node: &GraphNode,
        expr: &crate::ast::AstNode,
        name: &str,
        strategy: &ParallelizationStrategy,
    ) -> Option<crate::ast::AstNode> {
        match strategy {
            ParallelizationStrategy::Sequential => {
                elementwise::build_fused_elementwise_function(node, expr, name)
            }
            _ => {
                let shape = node.view.shape();
                let ndim = shape.len();
                let num_inputs = node
                    .src
                    .iter()
                    .filter(|s| !matches!(s.op, GraphOp::Const(_)) && !is_pure_const_node(s))
                    .count();

                let expr_with_consts = elementwise::embed_constants(expr, &node.src);

                Some(parallel::build_parallel_elementwise_kernel(
                    ndim,
                    num_inputs,
                    expr_with_consts,
                    &node.dtype,
                    name,
                    strategy,
                ))
            }
        }
    }

    /// Reduce演算のAST生成（戦略に応じて分岐）
    fn build_reduce_ast(
        &self,
        node: &GraphNode,
        op: &crate::graph::ReduceOp,
        axis: usize,
        name: &str,
        strategy: &ParallelizationStrategy,
    ) -> Option<crate::ast::AstNode> {
        match strategy {
            ParallelizationStrategy::Sequential => {
                reduce::build_reduce_function(node, op, axis, name)
            }
            _ => parallel::build_parallel_reduce_kernel(node, op, axis, name, strategy),
        }
    }

    /// 利用可能な並列化戦略のリストを取得
    fn available_strategies(&self, node: &GraphNode) -> Vec<ParallelizationStrategy> {
        // Sequential専用モードの場合は逐次のみ
        if self.sequential_only {
            return vec![ParallelizationStrategy::Sequential];
        }

        let mut strategies = vec![ParallelizationStrategy::Sequential];

        // 総要素数を計算（ベクトル化の可否判定に使用）
        let total_elements = self.calculate_total_elements(node);

        // Elementwise系とReduce系のみ並列化をサポート
        match &node.op {
            GraphOp::Elementwise { .. } | GraphOp::FusedElementwise { .. } => {
                // 各スレッドグループサイズで候補を生成
                for &tg_size in &self.thread_group_sizes {
                    // スカラー版
                    strategies.push(ParallelizationStrategy::FlatParallel {
                        thread_group_size: tg_size,
                        vector_width: None,
                    });

                    // ベクトル化版（要素数が割り切れる場合のみ）
                    if let Some(total) = total_elements {
                        for &vec_width in &self.vector_widths {
                            if total % vec_width == 0 && total >= vec_width {
                                strategies.push(ParallelizationStrategy::FlatParallel {
                                    thread_group_size: tg_size,
                                    vector_width: Some(vec_width),
                                });
                            }
                        }
                    }
                }

                // 多次元並列化は次元数に応じて追加
                let ndim = node.view.shape().len();
                for &tg_size in &self.thread_group_sizes {
                    if ndim >= 1 {
                        strategies.push(ParallelizationStrategy::MultiDimParallel {
                            parallel_dims: 1,
                            thread_group_size: tg_size,
                            vector_width: None,
                        });
                    }
                    if ndim >= 2 {
                        strategies.push(ParallelizationStrategy::MultiDimParallel {
                            parallel_dims: 2,
                            thread_group_size: tg_size,
                            vector_width: None,
                        });
                    }
                }
            }
            GraphOp::Reduce { .. } => {
                // Reduceは現時点ではベクトル化をサポートしない
                for &tg_size in &self.thread_group_sizes {
                    strategies.push(ParallelizationStrategy::FlatParallel {
                        thread_group_size: tg_size,
                        vector_width: None,
                    });

                    let ndim = node.view.shape().len();
                    if ndim >= 1 {
                        strategies.push(ParallelizationStrategy::MultiDimParallel {
                            parallel_dims: 1,
                            thread_group_size: tg_size,
                            vector_width: None,
                        });
                    }
                    if ndim >= 2 {
                        strategies.push(ParallelizationStrategy::MultiDimParallel {
                            parallel_dims: 2,
                            thread_group_size: tg_size,
                            vector_width: None,
                        });
                    }
                }
            }
            GraphOp::FusedElementwiseReduce { .. } => {
                // FusedElementwiseReduceはFlatParallelのみサポート
                // 出力軸を並列化し、縮約軸は逐次ループで処理
                for &tg_size in &self.thread_group_sizes {
                    strategies.push(ParallelizationStrategy::FlatParallel {
                        thread_group_size: tg_size,
                        vector_width: None,
                    });
                }
            }
            _ => {
                // その他の演算は逐次のみ
            }
        }

        strategies
    }

    /// ノードの総要素数を計算（定数の場合のみ）
    fn calculate_total_elements(&self, node: &GraphNode) -> Option<usize> {
        use crate::graph::shape::Expr;

        let shape = node.view.shape();
        let mut total = 1usize;
        for dim in shape {
            match dim {
                Expr::Const(val) => {
                    total *= *val as usize;
                }
                _ => return None, // シンボリックな軸がある場合は計算不可
            }
        }
        Some(total)
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

        // 入力・出力メタデータをコピー
        new_graph.copy_input_metas_from(graph);
        new_graph.copy_output_metas_from(graph);

        // ProgramRootノードがある場合は、Program構造を保持しながらsrcを再構築
        if let Some(old_sink) = graph.program_root() {
            let new_sink_src: Vec<GraphNode> = old_sink
                .src
                .iter()
                .map(|src| rebuild_node(src, &node_map, &mut visited))
                .collect();

            if let GraphOp::ProgramRoot { ast, outputs } = &old_sink.op {
                let new_sink = GraphNode::new(
                    old_sink.dtype.clone(),
                    GraphOp::ProgramRoot {
                        ast: ast.clone(),
                        outputs: outputs.clone(),
                    },
                    new_sink_src,
                    old_sink.view.clone(),
                );
                new_graph.set_program_root(new_sink);
            }
        } else {
            // ProgramRootがない場合は従来通りoutputsを使用
            let outputs_map = graph.outputs();
            let mut outputs: Vec<_> = outputs_map.iter().collect();
            outputs.sort_by_key(|(name, _)| name.as_str());

            for (name, output_node) in outputs {
                let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
                new_graph.output(name, rebuilt);
            }
        }

        // shape変数のデフォルト値をコピー
        for (name, value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(name.clone(), *value);
        }

        new_graph
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
            if matches!(node.op, GraphOp::Kernel { .. }) {
                already_custom += 1;
                continue;
            }

            if !self.can_lower(node) {
                continue;
            }

            lowerable_count += 1;

            // 利用可能な全戦略で候補を生成
            let strategies = self.available_strategies(node);
            for strategy in &strategies {
                if let Some(custom_node) = self.lower_to_custom(node, strategy) {
                    let new_graph = self.replace_node_in_graph(graph, node, custom_node);
                    suggestions.push(new_graph);
                    lowered_count += 1;
                } else {
                    log::debug!(
                        "LoweringSuggester: failed to lower {:?} with strategy {:?}",
                        std::mem::discriminant(&node.op),
                        strategy
                    );
                }
            }
        }

        log::debug!(
            "LoweringSuggester: {} nodes total, {} already custom, {} lowerable, {} lowered candidates",
            nodes.len(),
            already_custom,
            lowerable_count,
            lowered_count
        );

        suggestions
    }
}

#[cfg(test)]
mod tests;

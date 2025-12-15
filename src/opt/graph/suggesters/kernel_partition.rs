//! カーネルパーティションSuggester
//!
//! 1D並列化されたGraphOp::Kernel内のAstNode::Kernelを多次元グリッドに変換するSuggester。
//! グラフレベルで操作することで、dispatch設定の一貫性を保証します。

use crate::ast::{AstNode, DType, Mutability, VarDecl, VarKind, helper::*};
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::graph::{GraphSuggester, SuggestResult};
use log::{debug, trace};
use std::collections::HashSet;

/// スレッドグループサイズを指定した次元数に分配する
///
/// 各軸に均等に分配し、2のべき乗を維持します。
///
/// # Arguments
/// * `total_size` - 総スレッドグループサイズ（例: 256）
/// * `dims` - 分配する次元数（1, 2, or 3）
///
/// # Returns
/// 各軸のスレッドグループサイズ [x, y, z]
pub fn distribute_thread_group_size(total_size: usize, dims: usize) -> [usize; 3] {
    match dims {
        1 => [total_size, 1, 1],
        2 => {
            // 2次元: できるだけ正方形に近い分配
            let sqrt_approx = (total_size as f64).sqrt() as usize;
            let x = sqrt_approx.next_power_of_two() / 2;
            let x = x.max(1);
            let y = total_size / x;
            [x, y, 1]
        }
        3 => {
            // 3次元: 立方根に近い分配
            let cbrt_approx = (total_size as f64).cbrt() as usize;
            let x = cbrt_approx.next_power_of_two() / 2;
            let x = x.max(1);
            let remaining = total_size / x;
            let y = ((remaining as f64).sqrt() as usize).max(1);
            let y = y.next_power_of_two() / 2;
            let y = y.max(1);
            let z = remaining / y;
            [x, y, z]
        }
        _ => [total_size, 1, 1],
    }
}

/// 切り上げ計算: ceil(a / b)
fn ceil_div(a: AstNode, b: AstNode) -> AstNode {
    idiv(a + b.clone() - const_int(1), b)
}

/// GraphOp::Kernel内の1D並列Kernelを多次元グリッドに分割するSuggester
///
/// LoweringSuggesterで生成されたFlatParallel Kernelをより効率的な
/// 多次元並列化構成に変換します。
///
/// # 変換内容
///
/// ```text
/// // 変換前 (1D FlatParallel)
/// GraphOp::Kernel { ast: AstNode::Kernel {
///     params: [tid: ThreadId(0), ...],
///     body: { if (tid < total) { ... } },
///     grid_size: [ceil_div(N, 256) * 256, 1, 1],
///     thread_group_size: [256, 1, 1],
/// }}
///
/// // 変換後 (2D Grid)
/// GraphOp::Kernel { ast: AstNode::Kernel {
///     params: [tid_0: ThreadId(0), tid_1: ThreadId(1), ...],
///     body: { if (tid_0 < shape_0 && tid_1 < shape_1) { ... } },
///     grid_size: [ceil_div(shape_0, 16) * 16, ceil_div(shape_1, 16) * 16, 1],
///     thread_group_size: [16, 16, 1],
/// }}
/// ```
pub struct KernelPartitionSuggester {
    /// 並列化する軸数の候補
    parallel_dims_options: Vec<usize>,
    /// スレッドグループサイズの候補
    thread_group_sizes: Vec<usize>,
}

impl KernelPartitionSuggester {
    /// 新しいKernelPartitionSuggesterを作成
    pub fn new() -> Self {
        Self {
            parallel_dims_options: vec![2, 3],
            thread_group_sizes: vec![64, 128, 256],
        }
    }

    /// 並列化する軸数を設定
    pub fn with_parallel_dims(mut self, dims: Vec<usize>) -> Self {
        self.parallel_dims_options = dims;
        self
    }

    /// スレッドグループサイズを設定
    pub fn with_thread_group_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.thread_group_sizes = sizes;
        self
    }

    /// グラフ内のKernelノードを収集
    fn collect_kernel_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut kernels = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const GraphNodeData>,
            kernels: &mut Vec<GraphNode>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in &node.src {
                visit(src, visited, kernels);
            }

            if matches!(node.op, GraphOp::Kernel { .. }) {
                kernels.push(node.clone());
            }
        }

        // outputsから走査
        for output in graph.outputs().values() {
            visit(output, &mut visited, &mut kernels);
        }

        kernels
    }

    /// 1D FlatParallel Kernelかどうかをチェック
    fn is_1d_flat_parallel_kernel(&self, node: &GraphNode) -> bool {
        match &node.op {
            GraphOp::Kernel {
                ast: AstNode::Kernel { params, .. },
                ..
            } => {
                // ThreadId(0)のパラメータが"tid"という名前で存在するかチェック
                params
                    .iter()
                    .any(|p| p.name == "tid" && matches!(p.kind, VarKind::ThreadId(0)))
            }
            _ => false,
        }
    }

    /// Kernel本体から次元数を推測
    fn infer_ndim(&self, node: &GraphNode) -> usize {
        match &node.op {
            GraphOp::Kernel {
                ast: AstNode::Kernel { body, .. },
                ..
            } => self.infer_ndim_from_body(body),
            _ => 1,
        }
    }

    /// Kernel本体から次元数を推測
    fn infer_ndim_from_body(&self, body: &AstNode) -> usize {
        let mut max_dim = 0;
        self.collect_shape_vars(body, &mut max_dim);
        max_dim + 1
    }

    /// shape_N変数を再帰的に収集
    fn collect_shape_vars(&self, node: &AstNode, max_dim: &mut usize) {
        match node {
            AstNode::Var(name) => {
                if let Some(suffix) = name.strip_prefix("shape_")
                    && let Ok(dim) = suffix.parse::<usize>()
                {
                    *max_dim = (*max_dim).max(dim);
                }
            }
            // 再帰的に子ノードを探索
            AstNode::Add(a, b)
            | AstNode::Mul(a, b)
            | AstNode::Max(a, b)
            | AstNode::Rem(a, b)
            | AstNode::Idiv(a, b) => {
                self.collect_shape_vars(a, max_dim);
                self.collect_shape_vars(b, max_dim);
            }
            AstNode::Lt(a, b)
            | AstNode::Le(a, b)
            | AstNode::Gt(a, b)
            | AstNode::Ge(a, b)
            | AstNode::Eq(a, b)
            | AstNode::Ne(a, b) => {
                self.collect_shape_vars(a, max_dim);
                self.collect_shape_vars(b, max_dim);
            }
            AstNode::Recip(a) | AstNode::Sqrt(a) | AstNode::Log2(a) | AstNode::Exp2(a) => {
                self.collect_shape_vars(a, max_dim);
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    self.collect_shape_vars(stmt, max_dim);
                }
            }
            AstNode::Range {
                body,
                start,
                stop,
                step,
                ..
            } => {
                self.collect_shape_vars(body, max_dim);
                self.collect_shape_vars(start, max_dim);
                self.collect_shape_vars(stop, max_dim);
                self.collect_shape_vars(step, max_dim);
            }
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                self.collect_shape_vars(condition, max_dim);
                self.collect_shape_vars(then_body, max_dim);
                if let Some(e) = else_body {
                    self.collect_shape_vars(e, max_dim);
                }
            }
            AstNode::Store { ptr, offset, value } => {
                self.collect_shape_vars(ptr, max_dim);
                self.collect_shape_vars(offset, max_dim);
                self.collect_shape_vars(value, max_dim);
            }
            AstNode::Load { ptr, offset, .. } => {
                self.collect_shape_vars(ptr, max_dim);
                self.collect_shape_vars(offset, max_dim);
            }
            AstNode::Assign { value, .. } => {
                self.collect_shape_vars(value, max_dim);
            }
            AstNode::BitwiseAnd(a, b)
            | AstNode::BitwiseOr(a, b)
            | AstNode::BitwiseXor(a, b)
            | AstNode::LeftShift(a, b)
            | AstNode::RightShift(a, b) => {
                self.collect_shape_vars(a, max_dim);
                self.collect_shape_vars(b, max_dim);
            }
            AstNode::BitwiseNot(a) | AstNode::Sin(a) | AstNode::Cast(a, _) => {
                self.collect_shape_vars(a, max_dim);
            }
            _ => {}
        }
    }

    /// Kernelを多次元グリッドに分割
    fn partition_kernel(
        &self,
        node: &GraphNode,
        parallel_dims: usize,
        thread_group_size: usize,
    ) -> Option<GraphNode> {
        let GraphOp::Kernel { ast, input_buffers } = &node.op else {
            return None;
        };

        let AstNode::Kernel {
            name,
            params,
            return_type,
            body,
            ..
        } = ast
        else {
            return None;
        };

        // 新しいパラメータを構築（tidをtid_0, tid_1, ...に置換）
        let mut new_params = Vec::new();
        for param in params {
            if param.name == "tid" && matches!(param.kind, VarKind::ThreadId(0)) {
                // 多次元スレッドIDに置換
                for dim in 0..parallel_dims {
                    new_params.push(VarDecl {
                        name: format!("tid_{}", dim),
                        dtype: DType::Int,
                        mutability: Mutability::Immutable,
                        kind: VarKind::ThreadId(dim),
                    });
                }
            } else {
                new_params.push(param.clone());
            }
        }

        // 本体を変換
        let ndim = self.infer_ndim_from_body(body);
        let new_body = self.transform_body(body, parallel_dims, ndim);

        // スレッドグループサイズを分配
        let tg_dist = distribute_thread_group_size(thread_group_size, parallel_dims);

        // グリッドサイズを計算（各軸で切り上げ）
        let grid_size = self.build_multidim_grid_size(parallel_dims, &tg_dist);

        // 新しいスレッドグループサイズ
        let new_tg_size = [
            Box::new(const_int(tg_dist[0] as isize)),
            Box::new(const_int(tg_dist[1] as isize)),
            Box::new(const_int(tg_dist[2] as isize)),
        ];

        let new_ast = AstNode::Kernel {
            name: name.clone(),
            params: new_params,
            return_type: return_type.clone(),
            body: Box::new(new_body),
            default_grid_size: grid_size,
            default_thread_group_size: new_tg_size,
        };

        Some(GraphNode::new(
            node.dtype.clone(),
            GraphOp::Kernel {
                ast: new_ast,
                input_buffers: input_buffers.clone(),
            },
            node.src.clone(),
            node.view.clone(),
        ))
    }

    /// 本体を変換（tid -> 多次元インデックス）
    fn transform_body(&self, body: &AstNode, parallel_dims: usize, ndim: usize) -> AstNode {
        self.transform_node(body, parallel_dims, ndim)
    }

    /// ノードを再帰的に変換
    fn transform_node(&self, node: &AstNode, parallel_dims: usize, ndim: usize) -> AstNode {
        match node {
            // tid参照を多次元オフセットに変換
            AstNode::Var(name) if name == "tid" => self.build_linear_offset(parallel_dims, ndim),

            // 再帰的に子ノードを変換
            AstNode::Lt(a, b) => AstNode::Lt(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Max(a, b) => AstNode::Max(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Rem(a, b) => AstNode::Rem(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Idiv(a, b) => AstNode::Idiv(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Recip(a) => {
                AstNode::Recip(Box::new(self.transform_node(a, parallel_dims, ndim)))
            }
            AstNode::Sqrt(a) => {
                AstNode::Sqrt(Box::new(self.transform_node(a, parallel_dims, ndim)))
            }
            AstNode::Log2(a) => {
                AstNode::Log2(Box::new(self.transform_node(a, parallel_dims, ndim)))
            }
            AstNode::Exp2(a) => {
                AstNode::Exp2(Box::new(self.transform_node(a, parallel_dims, ndim)))
            }
            AstNode::Le(a, b) => AstNode::Le(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Gt(a, b) => AstNode::Gt(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Ge(a, b) => AstNode::Ge(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Eq(a, b) => AstNode::Eq(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::Ne(a, b) => AstNode::Ne(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::BitwiseAnd(a, b) => AstNode::BitwiseAnd(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::BitwiseOr(a, b) => AstNode::BitwiseOr(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::BitwiseXor(a, b) => AstNode::BitwiseXor(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::LeftShift(a, b) => AstNode::LeftShift(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::RightShift(a, b) => AstNode::RightShift(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),
            AstNode::BitwiseNot(a) => {
                AstNode::BitwiseNot(Box::new(self.transform_node(a, parallel_dims, ndim)))
            }
            AstNode::Sin(a) => AstNode::Sin(Box::new(self.transform_node(a, parallel_dims, ndim))),
            AstNode::Cast(a, dtype) => AstNode::Cast(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                dtype.clone(),
            ),

            AstNode::Block { statements, scope } => {
                let new_stmts = statements
                    .iter()
                    .map(|s| self.transform_node(s, parallel_dims, ndim))
                    .collect();
                AstNode::Block {
                    statements: new_stmts,
                    scope: scope.clone(),
                }
            }
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                // 境界チェックパターンを検出: if (tid < total) { body }
                if self.is_boundary_check_condition(condition) && else_body.is_none() {
                    let transformed_body = self.transform_node(then_body, parallel_dims, ndim);
                    self.build_multidim_boundary_check_if(parallel_dims, transformed_body)
                } else {
                    AstNode::If {
                        condition: Box::new(self.transform_node(condition, parallel_dims, ndim)),
                        then_body: Box::new(self.transform_node(then_body, parallel_dims, ndim)),
                        else_body: else_body
                            .as_ref()
                            .map(|e| Box::new(self.transform_node(e, parallel_dims, ndim))),
                    }
                }
            }
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => AstNode::Range {
                var: var.clone(),
                start: Box::new(self.transform_node(start, parallel_dims, ndim)),
                step: Box::new(self.transform_node(step, parallel_dims, ndim)),
                stop: Box::new(self.transform_node(stop, parallel_dims, ndim)),
                body: Box::new(self.transform_node(body, parallel_dims, ndim)),
            },
            AstNode::Store { ptr, offset, value } => AstNode::Store {
                ptr: Box::new(self.transform_node(ptr, parallel_dims, ndim)),
                offset: Box::new(self.transform_node(offset, parallel_dims, ndim)),
                value: Box::new(self.transform_node(value, parallel_dims, ndim)),
            },
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => AstNode::Load {
                ptr: Box::new(self.transform_node(ptr, parallel_dims, ndim)),
                offset: Box::new(self.transform_node(offset, parallel_dims, ndim)),
                count: *count,
                dtype: dtype.clone(),
            },
            AstNode::Assign { var, value } => AstNode::Assign {
                var: var.clone(),
                value: Box::new(self.transform_node(value, parallel_dims, ndim)),
            },

            // その他のノードはそのまま返す
            _ => node.clone(),
        }
    }

    /// 境界チェック条件（tid < total_elements）かどうかをチェック
    fn is_boundary_check_condition(&self, condition: &AstNode) -> bool {
        match condition {
            AstNode::Lt(left, right) => self.is_tid_ref(left) && self.is_total_elements(right),
            _ => false,
        }
    }

    /// tidへの参照かどうかをチェック
    fn is_tid_ref(&self, node: &AstNode) -> bool {
        matches!(node, AstNode::Var(name) if name == "tid")
    }

    /// total_elements（shape_0 * shape_1 * ...）かどうかをチェック
    fn is_total_elements(&self, node: &AstNode) -> bool {
        match node {
            AstNode::Var(name) => name.starts_with("shape_"),
            AstNode::Mul(a, b) => self.is_total_elements(a) || self.is_total_elements(b),
            _ => false,
        }
    }

    /// 多次元インデックスから線形オフセットを計算
    fn build_linear_offset(&self, parallel_dims: usize, ndim: usize) -> AstNode {
        if parallel_dims == 0 || ndim == 0 {
            return const_int(0);
        }

        let actual_dims = parallel_dims.min(ndim);
        let mut offset = var(format!("tid_{}", actual_dims - 1));

        for axis in (0..actual_dims - 1).rev() {
            let mut stride = var(format!("shape_{}", axis + 1));
            for inner_axis in (axis + 2)..actual_dims {
                stride = stride * var(format!("shape_{}", inner_axis));
            }
            offset = var(format!("tid_{}", axis)) * stride + offset;
        }

        offset
    }

    /// 多次元境界チェックのネストしたIf文を構築
    fn build_multidim_boundary_check_if(
        &self,
        parallel_dims: usize,
        inner_body: AstNode,
    ) -> AstNode {
        if parallel_dims == 0 {
            return inner_body;
        }

        let mut result = inner_body;
        for dim in (0..parallel_dims).rev() {
            let condition = lt(var(format!("tid_{}", dim)), var(format!("shape_{}", dim)));
            result = if_then(condition, result);
        }

        result
    }

    /// 多次元グリッドサイズを構築
    fn build_multidim_grid_size(
        &self,
        parallel_dims: usize,
        tg_size: &[usize; 3],
    ) -> [Box<AstNode>; 3] {
        let mut grid = [
            Box::new(const_int(1)),
            Box::new(const_int(1)),
            Box::new(const_int(1)),
        ];

        for dim in 0..parallel_dims {
            let shape_var = var(format!("shape_{}", dim));
            let tg = const_int(tg_size[dim] as isize);
            let num_groups = ceil_div(shape_var, tg.clone());
            *grid[dim] = num_groups * tg;
        }

        grid
    }

    /// グラフ内の指定ノードを新しいノードで置換
    fn replace_node_in_graph(
        &self,
        graph: &Graph,
        old_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        use std::collections::HashMap;

        let old_ptr = old_node.as_ptr();
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
        node_map.insert(old_ptr, new_node);

        // グラフを再構築
        fn rebuild_node(
            node: &GraphNode,
            node_map: &mut HashMap<*const GraphNodeData, GraphNode>,
        ) -> GraphNode {
            let ptr = node.as_ptr();
            if let Some(mapped) = node_map.get(&ptr) {
                return mapped.clone();
            }

            // 入力ノードを再帰的に再構築
            let new_src: Vec<GraphNode> =
                node.src.iter().map(|s| rebuild_node(s, node_map)).collect();

            // ポインタ比較で変更があるかチェック
            let src_changed = new_src.len() != node.src.len()
                || new_src
                    .iter()
                    .zip(node.src.iter())
                    .any(|(a, b)| a.as_ptr() != b.as_ptr());

            let new_node = if !src_changed {
                node.clone()
            } else {
                GraphNode::new(
                    node.dtype.clone(),
                    node.op.clone(),
                    new_src,
                    node.view.clone(),
                )
            };

            node_map.insert(ptr, new_node.clone());
            new_node
        }

        // 新しいグラフを作成
        let mut new_graph = Graph::new();

        // 出力ノードを再構築
        for (name, output) in graph.outputs() {
            let new_output = rebuild_node(output, &mut node_map);
            new_graph.set_output_node(name.clone(), new_output);
        }

        new_graph
    }
}

impl Default for KernelPartitionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for KernelPartitionSuggester {
    fn name(&self) -> &'static str {
        "KernelPartition"
    }

    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult> {
        trace!("KernelPartitionSuggester: Generating partition suggestions");
        let mut suggestions = Vec::new();

        let kernel_nodes = self.collect_kernel_nodes(graph);
        trace!(
            "KernelPartitionSuggester: Found {} kernel nodes",
            kernel_nodes.len()
        );

        for kernel_node in kernel_nodes {
            // 1D FlatParallel Kernelのみを対象
            if !self.is_1d_flat_parallel_kernel(&kernel_node) {
                continue;
            }

            // ndimを取得
            let ndim = self.infer_ndim(&kernel_node);
            trace!(
                "KernelPartitionSuggester: Processing 1D kernel with ndim={}",
                ndim
            );

            for &parallel_dims in &self.parallel_dims_options {
                if parallel_dims > ndim || parallel_dims < 2 {
                    continue;
                }

                for &tg_size in &self.thread_group_sizes {
                    if let Some(partitioned) =
                        self.partition_kernel(&kernel_node, parallel_dims, tg_size)
                    {
                        let new_graph =
                            self.replace_node_in_graph(graph, &kernel_node, partitioned);
                        suggestions.push(SuggestResult::new(new_graph, self.name()));
                    }
                }
            }
        }

        debug!(
            "KernelPartitionSuggester: Generated {} suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::DType as GraphDType;
    use crate::graph::shape::View;

    #[test]
    fn test_distribute_thread_group_size_1d() {
        let result = distribute_thread_group_size(256, 1);
        assert_eq!(result, [256, 1, 1]);
    }

    #[test]
    fn test_distribute_thread_group_size_2d() {
        let result = distribute_thread_group_size(256, 2);
        assert_eq!(result[0] * result[1], 256);
        assert_eq!(result[2], 1);
        assert!(result[0].is_power_of_two());
    }

    #[test]
    fn test_distribute_thread_group_size_3d() {
        let result = distribute_thread_group_size(64, 3);
        assert_eq!(result[0] * result[1] * result[2], 64);
        assert!(result[0].is_power_of_two());
        assert!(result[1].is_power_of_two());
    }

    #[test]
    fn test_is_1d_flat_parallel_kernel() {
        let suggester = KernelPartitionSuggester::new();

        // 1D FlatParallel Kernelを含むGraphNodeを作成
        let body = AstNode::If {
            condition: Box::new(lt(var("tid"), var("shape_0") * var("shape_1"))),
            then_body: Box::new(store(var("output"), var("tid"), var("value"))),
            else_body: None,
        };

        let kernel_ast = AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "tid".to_string(),
                    dtype: DType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::ThreadId(0),
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(body),
            default_grid_size: [
                Box::new(const_int(1024)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
            default_thread_group_size: [
                Box::new(const_int(256)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
        };

        let kernel_node = GraphNode::new(
            GraphDType::F32,
            GraphOp::Kernel {
                ast: kernel_ast,
                input_buffers: None,
            },
            vec![],
            View::contiguous::<i64, _>([]),
        );

        assert!(suggester.is_1d_flat_parallel_kernel(&kernel_node));
    }

    #[test]
    fn test_infer_ndim() {
        let suggester = KernelPartitionSuggester::new();

        // shape_0, shape_1 を含むKernel
        let body = AstNode::Mul(
            Box::new(AstNode::Var("shape_0".to_string())),
            Box::new(AstNode::Var("shape_1".to_string())),
        );

        let kernel_ast = AstNode::Kernel {
            name: Some("test".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(body),
            default_grid_size: [
                Box::new(const_int(1)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
            default_thread_group_size: [
                Box::new(const_int(1)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
        };

        let kernel_node = GraphNode::new(
            GraphDType::F32,
            GraphOp::Kernel {
                ast: kernel_ast,
                input_buffers: None,
            },
            vec![],
            View::contiguous::<i64, _>([]),
        );

        let ndim = suggester.infer_ndim(&kernel_node);
        assert_eq!(ndim, 2); // shape_1が最大なのでndim = 2
    }

    #[test]
    fn test_build_linear_offset() {
        let suggester = KernelPartitionSuggester::new();

        // 2D: tid_0 * shape_1 + tid_1
        let offset = suggester.build_linear_offset(2, 2);
        assert!(matches!(offset, AstNode::Add(_, _)));
    }

    #[test]
    fn test_partition_kernel_basic() {
        let suggester = KernelPartitionSuggester::new();

        // 1D Kernelを含むGraphを作成
        let body = AstNode::If {
            condition: Box::new(lt(var("tid"), var("shape_0") * var("shape_1"))),
            then_body: Box::new(store(var("output"), var("tid"), var("value"))),
            else_body: None,
        };

        let kernel_ast = AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "tid".to_string(),
                    dtype: DType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::ThreadId(0),
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(body),
            default_grid_size: [
                Box::new(const_int(1024)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
            default_thread_group_size: [
                Box::new(const_int(256)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
        };

        let kernel_node = GraphNode::new(
            GraphDType::F32,
            GraphOp::Kernel {
                ast: kernel_ast,
                input_buffers: None,
            },
            vec![],
            View::contiguous::<i64, _>([]),
        );

        // partition_kernelを実行
        let partitioned = suggester.partition_kernel(&kernel_node, 2, 256);
        assert!(partitioned.is_some());

        let partitioned = partitioned.unwrap();
        if let GraphOp::Kernel { ast, .. } = &partitioned.op
            && let AstNode::Kernel { params, .. } = ast
        {
            // tid_0, tid_1 が追加されているはず
            let has_tid_0 = params.iter().any(|p| p.name == "tid_0");
            let has_tid_1 = params.iter().any(|p| p.name == "tid_1");
            assert!(has_tid_0, "Should have tid_0 parameter");
            assert!(has_tid_1, "Should have tid_1 parameter");
        }
    }
}

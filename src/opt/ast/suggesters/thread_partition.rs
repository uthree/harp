//! スレッドパーティションSuggester
//!
//! 1D並列化されたKernelを多次元グリッドに変換するSuggesterを提供します。
//! LoweringSuggesterで生成されたFlatParallel Kernelをより効率的な
//! 多次元並列化構成に変換できます。

use crate::ast::{AstNode, DType, Mutability, VarDecl, VarKind, helper::*};
use crate::opt::ast::Suggester;
use log::{debug, trace};

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
            // sqrt を計算し、2のべき乗に丸める
            let sqrt_approx = (total_size as f64).sqrt() as usize;
            // 2のべき乗に切り下げ
            let x = sqrt_approx.next_power_of_two() / 2;
            let x = x.max(1);
            let y = total_size / x;
            [x, y, 1]
        }
        3 => {
            // 3次元: 立方根に近い分配
            let cbrt_approx = (total_size as f64).cbrt() as usize;
            // 2のべき乗に切り下げ
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

/// 1D並列KernelをN次元グリッドに分割するSuggester
///
/// FlatParallel戦略で生成されたKernel（tidによる1D並列）を
/// 多次元スレッドIDを使用した構成に変換します。
///
/// # 変換内容
///
/// ```text
/// // 変換前 (1D FlatParallel)
/// Kernel {
///     params: [tid: ThreadId(0), ...],
///     body: { if (tid < total) { ... } },
///     grid_size: [ceil_div(N, 256) * 256, 1, 1],
///     thread_group_size: [256, 1, 1],
/// }
///
/// // 変換後 (2D Grid)
/// Kernel {
///     params: [tid_0: ThreadId(0), tid_1: ThreadId(1), ...],
///     body: { if (tid_0 < shape_0 && tid_1 < shape_1) { ... } },
///     grid_size: [ceil_div(shape_0, 16) * 16, ceil_div(shape_1, 16) * 16, 1],
///     thread_group_size: [16, 16, 1],
/// }
/// ```
pub struct ThreadPartitionSuggester {
    /// 並列化する軸数の候補
    parallel_dims_options: Vec<usize>,
    /// スレッドグループサイズの候補
    thread_group_sizes: Vec<usize>,
}

impl ThreadPartitionSuggester {
    /// 新しいThreadPartitionSuggesterを作成
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

    /// ASTからKernelのパーティション候補を収集
    fn collect_partition_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        match ast {
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => {
                // 1D FlatParallel Kernelかどうかをチェック
                if self.is_1d_flat_parallel_kernel(params) {
                    // Kernel本体からndimを推測
                    let ndim = self.infer_ndim_from_body(body);
                    trace!(
                        "ThreadPartitionSuggester: Found 1D kernel '{}' with inferred ndim={}",
                        name.as_deref().unwrap_or("anonymous"),
                        ndim
                    );

                    // 各設定で候補を生成
                    for &parallel_dims in &self.parallel_dims_options {
                        if parallel_dims > ndim || parallel_dims < 2 {
                            continue;
                        }

                        for &tg_size in &self.thread_group_sizes {
                            if let Some(partitioned) = self.partition_kernel(
                                name,
                                params,
                                return_type,
                                body,
                                default_grid_size,
                                default_thread_group_size,
                                parallel_dims,
                                tg_size,
                            ) {
                                candidates.push(partitioned);
                            }
                        }
                    }
                }
            }
            AstNode::Program {
                functions,
                entry_point,
            } => {
                // 各関数/Kernelを再帰的に探索
                for (i, func) in functions.iter().enumerate() {
                    for partitioned in self.collect_partition_candidates(func) {
                        let mut new_functions = functions.clone();
                        new_functions[i] = partitioned;
                        candidates.push(AstNode::Program {
                            functions: new_functions,
                            entry_point: entry_point.clone(),
                        });
                    }
                }
            }
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                // 関数本体を再帰的に探索
                for partitioned in self.collect_partition_candidates(body) {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(partitioned),
                    });
                }
            }
            AstNode::Block { statements, scope } => {
                for (i, stmt) in statements.iter().enumerate() {
                    for partitioned in self.collect_partition_candidates(stmt) {
                        let mut new_stmts = statements.clone();
                        new_stmts[i] = partitioned;
                        candidates.push(AstNode::Block {
                            statements: new_stmts,
                            scope: scope.clone(),
                        });
                    }
                }
            }
            _ => {}
        }

        candidates
    }

    /// 1D FlatParallel Kernelかどうかをチェック
    fn is_1d_flat_parallel_kernel(&self, params: &[VarDecl]) -> bool {
        // ThreadId(0)のパラメータが"tid"という名前で存在するかチェック
        params
            .iter()
            .any(|p| p.name == "tid" && matches!(p.kind, VarKind::ThreadId(0)))
    }

    /// Kernel本体から次元数を推測
    ///
    /// shape_0, shape_1, ... の変数参照を探し、最大のインデックスから
    /// ndimを推測します。
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
                        && dim > *max_dim {
                            *max_dim = dim;
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
    #[allow(clippy::too_many_arguments)]
    fn partition_kernel(
        &self,
        name: &Option<String>,
        params: &[VarDecl],
        return_type: &DType,
        body: &AstNode,
        _default_grid_size: &[Box<AstNode>; 3],
        _default_thread_group_size: &[Box<AstNode>; 3],
        parallel_dims: usize,
        thread_group_size: usize,
    ) -> Option<AstNode> {
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

        Some(AstNode::Kernel {
            name: name.clone(),
            params: new_params,
            return_type: return_type.clone(),
            body: Box::new(new_body),
            default_grid_size: grid_size,
            default_thread_group_size: new_tg_size,
        })
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

            // 比較演算は再帰的に処理
            AstNode::Lt(a, b) => AstNode::Lt(
                Box::new(self.transform_node(a, parallel_dims, ndim)),
                Box::new(self.transform_node(b, parallel_dims, ndim)),
            ),

            // 再帰的に子ノードを変換
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
                    // 本体を変換してから多次元境界チェックでラップ
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
        // シンプルに: Mul または shape_N変数を含むかチェック
        match node {
            AstNode::Var(name) => name.starts_with("shape_"),
            AstNode::Mul(a, b) => self.is_total_elements(a) || self.is_total_elements(b),
            _ => false,
        }
    }

    /// 多次元インデックスから線形オフセットを計算
    ///
    /// offset = tid_0 * stride_0 + tid_1 * stride_1 + ... + tid_(n-1)
    fn build_linear_offset(&self, parallel_dims: usize, ndim: usize) -> AstNode {
        if parallel_dims == 0 || ndim == 0 {
            return const_int(0);
        }

        // 使用する軸数（parallel_dimsとndimの小さい方）
        let actual_dims = parallel_dims.min(ndim);

        // 最後の軸から開始
        let mut offset = var(format!("tid_{}", actual_dims - 1));

        for axis in (0..actual_dims - 1).rev() {
            // stride = shape_(axis+1) * shape_(axis+2) * ... * shape_(actual_dims-1)
            let mut stride = var(format!("shape_{}", axis + 1));
            for inner_axis in (axis + 2)..actual_dims {
                stride = stride * var(format!("shape_{}", inner_axis));
            }
            offset = var(format!("tid_{}", axis)) * stride + offset;
        }

        offset
    }

    /// 多次元境界チェックのネストしたIf文を構築
    ///
    /// if (tid_0 < shape_0) { if (tid_1 < shape_1) { ... } }
    fn build_multidim_boundary_check_if(
        &self,
        parallel_dims: usize,
        inner_body: AstNode,
    ) -> AstNode {
        if parallel_dims == 0 {
            return inner_body;
        }

        // 内側から外側へ構築
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
}

impl Default for ThreadPartitionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl Suggester for ThreadPartitionSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("ThreadPartitionSuggester: Generating partition suggestions");
        let candidates = self.collect_partition_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);
        debug!(
            "ThreadPartitionSuggester: Generated {} unique suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribute_thread_group_size_1d() {
        let result = distribute_thread_group_size(256, 1);
        assert_eq!(result, [256, 1, 1]);
    }

    #[test]
    fn test_distribute_thread_group_size_2d() {
        let result = distribute_thread_group_size(256, 2);
        // sqrt(256) = 16, so [8, 32] or [16, 16] depending on implementation
        assert_eq!(result[0] * result[1], 256);
        assert_eq!(result[2], 1);
        // 2の累乗であること
        assert!(result[0].is_power_of_two());
    }

    #[test]
    fn test_distribute_thread_group_size_3d() {
        let result = distribute_thread_group_size(64, 3);
        // cbrt(64) = 4, so approximately [4, 4, 4] or similar
        assert_eq!(result[0] * result[1] * result[2], 64);
        // 各軸が2の累乗であること
        assert!(result[0].is_power_of_two());
        assert!(result[1].is_power_of_two());
    }

    #[test]
    fn test_is_1d_flat_parallel_kernel() {
        let suggester = ThreadPartitionSuggester::new();

        let params_1d = vec![
            VarDecl {
                name: "tid".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(0),
            },
            VarDecl {
                name: "input_0".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            },
        ];
        assert!(suggester.is_1d_flat_parallel_kernel(&params_1d));

        // 2Dの場合
        let params_2d = vec![
            VarDecl {
                name: "tid_0".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(0),
            },
            VarDecl {
                name: "tid_1".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(1),
            },
        ];
        assert!(!suggester.is_1d_flat_parallel_kernel(&params_2d));
    }

    #[test]
    fn test_infer_ndim_from_body() {
        use crate::ast::Scope;

        let suggester = ThreadPartitionSuggester::new();

        // shape_0, shape_1 を含む本体
        let body = AstNode::Block {
            statements: vec![AstNode::Mul(
                Box::new(AstNode::Var("shape_0".to_string())),
                Box::new(AstNode::Var("shape_1".to_string())),
            )],
            scope: Box::new(Scope::new()),
        };

        let ndim = suggester.infer_ndim_from_body(&body);
        assert_eq!(ndim, 2); // shape_1 が最大なので ndim = 2
    }

    #[test]
    fn test_build_linear_offset() {
        let suggester = ThreadPartitionSuggester::new();

        // 2D: tid_0 * shape_1 + tid_1
        let offset = suggester.build_linear_offset(2, 2);

        // Mul(Var(tid_0), Var(shape_1)) + Var(tid_1) の形になるはず
        assert!(matches!(offset, AstNode::Add(_, _)));
    }

    #[test]
    fn test_build_multidim_boundary_check_if() {
        use crate::ast::Scope;

        let suggester = ThreadPartitionSuggester::new();

        // 2D境界チェック: if (tid_0 < shape_0) { if (tid_1 < shape_1) { body } }
        let inner_body = AstNode::Block {
            statements: vec![],
            scope: Box::new(Scope::new()),
        };
        let check = suggester.build_multidim_boundary_check_if(2, inner_body);

        // If(Lt(...), If(...)) の形になるはず
        assert!(matches!(check, AstNode::If { .. }));
    }

    #[test]
    fn test_partition_kernel_basic() {
        let suggester = ThreadPartitionSuggester::new();

        // 簡単な1D Kernelを作成
        let body = AstNode::If {
            condition: Box::new(lt(var("tid"), var("shape_0") * var("shape_1"))),
            then_body: Box::new(store(var("output"), var("tid"), var("value"))),
            else_body: None,
        };

        let kernel = AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "tid".to_string(),
                    dtype: DType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::ThreadId(0),
                },
                VarDecl {
                    name: "input_0".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
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

        let suggestions = suggester.suggest(&kernel);

        // 複数の候補が生成されるはず（parallel_dims=2 x thread_group_sizes）
        assert!(!suggestions.is_empty());

        // 最初の候補をチェック
        if let AstNode::Kernel { params, .. } = &suggestions[0] {
            // tid_0, tid_1 が追加されているはず
            let has_tid_0 = params.iter().any(|p| p.name == "tid_0");
            let has_tid_1 = params.iter().any(|p| p.name == "tid_1");
            assert!(has_tid_0, "Should have tid_0 parameter");
            assert!(has_tid_1, "Should have tid_1 parameter");
        }
    }
}

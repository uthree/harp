//! ループ並列化を提案するSuggester
//!
//! 2つのSuggesterを提供:
//! - GroupParallelizationSuggester: GroupId使用、グループ単位の並列化、動的分岐チェックあり
//! - LocalParallelizationSuggester: LocalId使用、ワークグループ内並列化、動的分岐チェックなし
//!
//! 両方のSuggesterがFunction→KernelとKernel内ループ並列化の両方に対応。
//!
//! グローバルインデックスが必要な場合は `group_id * local_size + local_id` で計算できます。
//!
//! ホスト側でスレッド数・グループ数を正確に設定するため、
//! 並列化時に境界チェック（if文）は生成しない。

use crate::ast::{AddressSpace, AstNode, Literal, Scope};
use crate::opt::ast::{AstSuggestResult, AstSuggester};

use super::parallelization_common::{
    collect_free_variables, const_int, find_next_available_axis, group_id_param,
    infer_params_from_placeholders, is_range_parallelizable, is_range_thread_parallelizable,
    local_id_param, substitute_var, var,
};

// ============================================================================
// 共通ヘルパー関数
// ============================================================================

/// ループの総イテレーション数を計算
fn compute_total_iterations(start: &AstNode, stop: &AstNode) -> AstNode {
    if matches!(start, AstNode::Const(Literal::I64(0))) {
        stop.clone()
    } else {
        AstNode::Add(
            Box::new(stop.clone()),
            Box::new(AstNode::Mul(
                Box::new(const_int(-1)),
                Box::new(start.clone()),
            )),
        )
    }
}

/// grid_sizeを更新（正確なイテレーション数を設定）
fn update_grid_size(current: &[Box<AstNode>; 3], axis: usize, size: &AstNode) -> [Box<AstNode>; 3] {
    let mut new_sizes = current.clone();
    if axis < 3 {
        *new_sizes[axis] = size.clone();
    }
    new_sizes
}

/// thread_group_sizeを更新
fn update_thread_group_size(
    current: &[Box<AstNode>; 3],
    axis: usize,
    size: &AstNode,
) -> [Box<AstNode>; 3] {
    let mut new_sizes = current.clone();
    if axis < 3 {
        *new_sizes[axis] = size.clone();
    }
    new_sizes
}

/// ASTノード内のRangeを置換
fn replace_range_with(node: &AstNode, target_range: &AstNode, replacement: AstNode) -> AstNode {
    if std::ptr::eq(node, target_range) {
        return replacement;
    }

    match node {
        AstNode::Block { statements, scope } => {
            let new_statements: Vec<_> = statements
                .iter()
                .map(|s| replace_range_with(s, target_range, replacement.clone()))
                .collect();
            AstNode::Block {
                statements: new_statements,
                scope: scope.clone(),
            }
        }
        AstNode::If {
            condition,
            then_body,
            else_body,
        } => AstNode::If {
            condition: condition.clone(),
            then_body: Box::new(replace_range_with(then_body, target_range, replacement)),
            else_body: else_body.clone(),
        },
        _ => node.clone(),
    }
}

// ============================================================================
// GroupParallelizationSuggester
// ============================================================================

/// グループ単位の並列化を提案するSuggester
///
/// GroupId（get_group_id）を使用してワークグループ単位の並列化を行います。
///
/// **動的分岐チェック: あり**
/// GPUの分岐ダイバージェンスを避けるため、ループ内にIf文があると並列化しません。
///
/// グローバルインデックスが必要な場合は `group_id * local_size + local_id` で計算します。
///
/// # 対応する変換
///
/// ## Function → Kernel
/// ```text
/// // 変換前
/// Function { body: Range { var: "i", ... } }
/// // 変換後
/// Kernel { params: [gidx0: GroupId(0), ...], grid_size: [N, 1, 1], ... }
/// ```
///
/// ## Kernel内ループ → GroupId追加
/// ```text
/// // 変換前
/// Kernel { params: [gidx0: GroupId(0)], body: Range { var: "j", ... } }
/// // 変換後
/// Kernel { params: [gidx0: GroupId(0), gidx1: GroupId(1)], ... }
/// ```
pub struct GroupParallelizationSuggester;

impl GroupParallelizationSuggester {
    pub fn new() -> Self {
        Self
    }

    /// 最外側の並列化可能なRangeループを見つける（動的分岐チェックあり）
    fn find_parallelizable_range<'a>(&self, node: &'a AstNode) -> Option<&'a AstNode> {
        match node {
            AstNode::Range { .. } => {
                if is_range_thread_parallelizable(node) {
                    Some(node)
                } else {
                    None
                }
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    if let Some(range) = self.find_parallelizable_range(stmt) {
                        return Some(range);
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn try_parallelize_function(&self, func: &AstNode) -> Option<AstNode> {
        let AstNode::Function {
            name,
            params,
            return_type,
            body,
        } = func
        else {
            return None;
        };

        let range_node = self.find_parallelizable_range(body)?;

        let AstNode::Range {
            var: loop_var,
            start,
            stop,
            body: loop_body,
            ..
        } = range_node
        else {
            return None;
        };

        log::debug!(
            "GroupParallelization: Converting Function {:?} to Kernel",
            name
        );

        let gidx_name = "gidx0";
        // ループ変数をGroupIdに置換（境界チェックなし）
        let new_body = substitute_var(loop_body, loop_var, &var(gidx_name));

        let kernel_body = if let AstNode::Block { scope, .. } = loop_body.as_ref() {
            AstNode::Block {
                statements: vec![new_body],
                scope: scope.clone(),
            }
        } else {
            AstNode::Block {
                statements: vec![new_body],
                scope: Box::new(Scope::new()),
            }
        };

        // grid_sizeは正確なイテレーション数を設定
        let total_iterations = compute_total_iterations(start, stop);

        let mut kernel_params = vec![group_id_param(gidx_name, 0)];

        if params.is_empty() {
            let free_vars = collect_free_variables(&kernel_body);
            let free_vars: Vec<_> = free_vars.into_iter().filter(|v| v != gidx_name).collect();
            let inferred_params = infer_params_from_placeholders(&free_vars);
            kernel_params.extend(inferred_params);
        } else {
            kernel_params.extend(params.iter().cloned());
        }

        Some(AstNode::Kernel {
            name: name.clone(),
            params: kernel_params,
            return_type: return_type.clone(),
            body: Box::new(kernel_body),
            default_grid_size: [
                Box::new(total_iterations),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
            // GroupId並列化のみの場合、thread_group_size = [1, 1, 1]
            // LocalId並列化を追加すると、その軸のthread_group_sizeが更新される
            default_thread_group_size: [
                Box::new(const_int(1)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
        })
    }

    fn try_parallelize_kernel(&self, kernel: &AstNode) -> Option<AstNode> {
        let AstNode::Kernel {
            name,
            params,
            return_type,
            body,
            default_grid_size,
            default_thread_group_size,
        } = kernel
        else {
            return None;
        };

        let range_node = self.find_parallelizable_range(body)?;

        let AstNode::Range {
            var: loop_var,
            start,
            stop,
            body: loop_body,
            ..
        } = range_node
        else {
            return None;
        };

        // すべてのID種類（LocalId, GroupId）が使用している軸を避ける
        let next_axis = find_next_available_axis(params)?;
        let gidx_name = format!("gidx{}", next_axis);

        log::debug!(
            "GroupParallelization: Adding GroupId({}) to Kernel {:?}: {} -> {}",
            next_axis,
            name,
            loop_var,
            gidx_name
        );

        // ループ変数をGroupIdに置換（境界チェックなし）
        let new_body = substitute_var(loop_body, loop_var, &var(&gidx_name));
        let new_kernel_body = replace_range_with(body, range_node, new_body);

        let mut new_params = params.clone();
        new_params.push(group_id_param(&gidx_name, next_axis));

        // grid_sizeは正確なイテレーション数を設定
        let total_iterations = compute_total_iterations(start, stop);
        let new_grid_size = update_grid_size(default_grid_size, next_axis, &total_iterations);

        // GroupIdの場合、thread_group_sizeはデフォルトを維持（LocalIdで別途設定可能）
        let new_thread_group_size = default_thread_group_size.clone();

        Some(AstNode::Kernel {
            name: name.clone(),
            params: new_params,
            return_type: return_type.clone(),
            body: Box::new(new_kernel_body),
            default_grid_size: new_grid_size,
            default_thread_group_size: new_thread_group_size,
        })
    }

    fn process_program(&self, program: &AstNode) -> Vec<AstSuggestResult> {
        let mut results = Vec::new();

        let AstNode::Program {
            functions,
            execution_waves,
        } = program
        else {
            return results;
        };

        for (idx, func) in functions.iter().enumerate() {
            match func {
                AstNode::Function { name, .. } => {
                    if let Some(kernel) = self.try_parallelize_function(func) {
                        let mut new_functions = functions.clone();
                        new_functions[idx] = kernel;

                        let func_name = name.clone().unwrap_or_else(|| format!("func_{}", idx));

                        results.push(AstSuggestResult::with_description(
                            AstNode::Program {
                                functions: new_functions,
                                execution_waves: execution_waves.clone(),
                            },
                            self.name(),
                            format!("Parallelize {} (Function→Kernel)", func_name),
                        ));
                    }
                }
                AstNode::Kernel { name, .. } => {
                    if let Some(kernel) = self.try_parallelize_kernel(func) {
                        let mut new_functions = functions.clone();
                        new_functions[idx] = kernel;

                        let kernel_name = name.clone().unwrap_or_else(|| format!("kernel_{}", idx));

                        results.push(AstSuggestResult::with_description(
                            AstNode::Program {
                                functions: new_functions,
                                execution_waves: execution_waves.clone(),
                            },
                            self.name(),
                            format!("Parallelize {} (add GroupId)", kernel_name),
                        ));
                    }
                }
                _ => {}
            }
        }

        results
    }
}

impl Default for GroupParallelizationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for GroupParallelizationSuggester {
    fn name(&self) -> &str {
        "GroupParallelization"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        match ast {
            AstNode::Program { .. } => self.process_program(ast),
            AstNode::Function { .. } => {
                if let Some(kernel) = self.try_parallelize_function(ast) {
                    vec![AstSuggestResult::with_description(
                        kernel,
                        self.name(),
                        "Parallelize (Function→Kernel)".to_string(),
                    )]
                } else {
                    vec![]
                }
            }
            AstNode::Kernel { .. } => {
                if let Some(kernel) = self.try_parallelize_kernel(ast) {
                    vec![AstSuggestResult::with_description(
                        kernel,
                        self.name(),
                        "Parallelize (add GroupId)".to_string(),
                    )]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }
}

// ============================================================================
// LocalParallelizationSuggester
// ============================================================================

/// ローカル並列化を提案するSuggester
///
/// LocalId（get_local_id）を使用してワークグループ内並列化を行います。
///
/// **動的分岐チェック: なし**
/// ワークグループ内は同期的にスケジューリングされるため、分岐ダイバージェンスの
/// 影響は限定的です。If文を含むループも並列化対象となります。
///
/// # 対応する変換
///
/// ## Function → Kernel
/// ```text
/// // 変換前
/// Function { body: Range { var: "i", ... } }
/// // 変換後
/// Kernel { params: [lidx0: LocalId(0), ...], thread_group_size: [N, 1, 1] }
/// ```
///
/// ## Kernel内ループ → LocalId追加
/// ```text
/// // 変換前
/// Kernel { params: [gidx0: GroupId(0)], body: Range { var: "j", ... } }
/// // 変換後（GroupId(0)が軸0を使用しているため、LocalIdは軸1を使用）
/// Kernel { params: [gidx0: GroupId(0), lidx1: LocalId(1)], thread_group_size: [256, M, 1] }
/// ```
pub struct LocalParallelizationSuggester;

impl LocalParallelizationSuggester {
    pub fn new() -> Self {
        Self
    }

    /// 最外側の並列化可能なRangeループを見つける（動的分岐チェックなし）
    fn find_parallelizable_range<'a>(&self, node: &'a AstNode) -> Option<&'a AstNode> {
        match node {
            AstNode::Range { .. } => {
                if is_range_parallelizable(node) {
                    Some(node)
                } else {
                    None
                }
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    if let Some(range) = self.find_parallelizable_range(stmt) {
                        return Some(range);
                    }
                }
                None
            }
            AstNode::If { then_body, .. } => self.find_parallelizable_range(then_body),
            _ => None,
        }
    }

    fn try_parallelize_function(&self, func: &AstNode) -> Option<AstNode> {
        let AstNode::Function {
            name,
            params,
            return_type,
            body,
        } = func
        else {
            return None;
        };

        let range_node = self.find_parallelizable_range(body)?;

        let AstNode::Range {
            var: loop_var,
            start,
            stop,
            body: loop_body,
            ..
        } = range_node
        else {
            return None;
        };

        log::debug!(
            "LocalParallelization: Converting Function {:?} to Kernel with LocalId",
            name
        );

        let lid_name = "lidx0";
        // ループ変数をローカルIDに置換（境界チェックなし）
        let new_body = substitute_var(loop_body, loop_var, &var(lid_name));

        let kernel_body = if let AstNode::Block { scope, .. } = loop_body.as_ref() {
            AstNode::Block {
                statements: vec![new_body],
                scope: scope.clone(),
            }
        } else {
            AstNode::Block {
                statements: vec![new_body],
                scope: Box::new(Scope::new()),
            }
        };

        // thread_group_sizeは正確なイテレーション数を設定
        let total_iterations = compute_total_iterations(start, stop);

        let mut kernel_params = vec![local_id_param(lid_name, 0)];

        if params.is_empty() {
            let free_vars = collect_free_variables(&kernel_body);
            let free_vars: Vec<_> = free_vars.into_iter().filter(|v| v != lid_name).collect();
            let inferred_params = infer_params_from_placeholders(&free_vars);
            kernel_params.extend(inferred_params);
        } else {
            kernel_params.extend(params.iter().cloned());
        }

        Some(AstNode::Kernel {
            name: name.clone(),
            params: kernel_params,
            return_type: return_type.clone(),
            body: Box::new(kernel_body),
            default_grid_size: [
                Box::new(const_int(1)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
            default_thread_group_size: [
                Box::new(total_iterations),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
        })
    }

    fn try_parallelize_kernel(&self, kernel: &AstNode) -> Option<AstNode> {
        let AstNode::Kernel {
            name,
            params,
            return_type,
            body,
            default_grid_size,
            default_thread_group_size,
        } = kernel
        else {
            return None;
        };

        let range_node = self.find_parallelizable_range(body)?;

        let AstNode::Range {
            var: loop_var,
            stop,
            body: loop_body,
            ..
        } = range_node
        else {
            return None;
        };

        // すべてのID種類（LocalId, GroupId）が使用している軸を避ける
        let next_axis = find_next_available_axis(params)?;
        let lid_name = format!("lidx{}", next_axis);

        log::debug!(
            "LocalParallelization: Adding LocalId({}) to Kernel {:?}: {} -> {}",
            next_axis,
            name,
            loop_var,
            lid_name
        );

        // ループ変数をローカルIDに置換（境界チェックなし）
        let new_body = substitute_var(loop_body, loop_var, &var(&lid_name));
        let new_kernel_body = replace_range_with(body, range_node, new_body);

        let mut new_params = params.clone();
        new_params.push(local_id_param(&lid_name, next_axis));

        // thread_group_sizeは正確なイテレーション数を設定
        let new_thread_group_size =
            update_thread_group_size(default_thread_group_size, next_axis, stop.as_ref());

        Some(AstNode::Kernel {
            name: name.clone(),
            params: new_params,
            return_type: return_type.clone(),
            body: Box::new(new_kernel_body),
            default_grid_size: default_grid_size.clone(),
            default_thread_group_size: new_thread_group_size,
        })
    }

    fn process_program(&self, program: &AstNode) -> Vec<AstSuggestResult> {
        let mut results = Vec::new();

        let AstNode::Program {
            functions,
            execution_waves,
        } = program
        else {
            return results;
        };

        for (idx, func) in functions.iter().enumerate() {
            match func {
                AstNode::Function { name, .. } => {
                    if let Some(kernel) = self.try_parallelize_function(func) {
                        let mut new_functions = functions.clone();
                        new_functions[idx] = kernel;

                        let func_name = name.clone().unwrap_or_else(|| format!("func_{}", idx));

                        results.push(AstSuggestResult::with_description(
                            AstNode::Program {
                                functions: new_functions,
                                execution_waves: execution_waves.clone(),
                            },
                            self.name(),
                            format!("Parallelize {} (Function→Kernel, LocalId)", func_name),
                        ));
                    }
                }
                AstNode::Kernel { name, .. } => {
                    if let Some(kernel) = self.try_parallelize_kernel(func) {
                        let mut new_functions = functions.clone();
                        new_functions[idx] = kernel;

                        let kernel_name = name.clone().unwrap_or_else(|| format!("kernel_{}", idx));

                        results.push(AstSuggestResult::with_description(
                            AstNode::Program {
                                functions: new_functions,
                                execution_waves: execution_waves.clone(),
                            },
                            self.name(),
                            format!("Parallelize {} (add LocalId)", kernel_name),
                        ));
                    }
                }
                _ => {}
            }
        }

        results
    }
}

impl Default for LocalParallelizationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for LocalParallelizationSuggester {
    fn name(&self) -> &str {
        "LocalParallelization"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        match ast {
            AstNode::Program { .. } => self.process_program(ast),
            AstNode::Function { .. } => {
                if let Some(kernel) = self.try_parallelize_function(ast) {
                    vec![AstSuggestResult::with_description(
                        kernel,
                        self.name(),
                        "Parallelize (Function→Kernel, LocalId)".to_string(),
                    )]
                } else {
                    vec![]
                }
            }
            AstNode::Kernel { .. } => {
                if let Some(kernel) = self.try_parallelize_kernel(ast) {
                    vec![AstSuggestResult::with_description(
                        kernel,
                        self.name(),
                        "Parallelize (add LocalId)".to_string(),
                    )]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        DType, Mutability, ParallelInfo, VarDecl, VarKind,
        helper::{eq, load, store},
    };

    fn make_simple_function() -> AstNode {
        let body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );

        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(var("N")),
            body: Box::new(body),
            parallel: ParallelInfo::default(),
        };

        AstNode::Function {
            name: Some("kernel_0".to_string()),
            params: vec![
                VarDecl {
                    name: "input".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "N".to_string(),
                    dtype: DType::I64,
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(range),
        }
    }

    fn make_function_with_branch() -> AstNode {
        // for i in 0..N { if (i % 2 == 0) { output[i] = input[i] } }
        let inner_body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );

        let if_node = AstNode::If {
            condition: Box::new(eq(
                AstNode::Rem(Box::new(var("i")), Box::new(const_int(2))),
                const_int(0),
            )),
            then_body: Box::new(inner_body),
            else_body: None,
        };

        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(var("N")),
            body: Box::new(if_node),
            parallel: ParallelInfo::default(),
        };

        AstNode::Function {
            name: Some("branching_kernel".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(range),
        }
    }

    fn make_kernel_with_loop() -> AstNode {
        let body = store(
            var("output"),
            var("j"),
            load(var("input"), var("j"), DType::F32),
        );

        let inner_loop = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(const_int(64)),
            body: Box::new(body),
            parallel: ParallelInfo::default(),
        };

        let one = const_int(1);
        AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "gidx0".to_string(),
                    dtype: DType::I64,
                    mutability: Mutability::Immutable,
                    kind: VarKind::GroupId(0),
                },
                VarDecl {
                    name: "input".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(inner_loop),
            default_grid_size: [
                Box::new(const_int(16)),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(1)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        }
    }

    fn make_kernel_with_branch_loop() -> AstNode {
        // Kernel内に分岐を含むループ
        let inner_body = store(
            var("output"),
            var("j"),
            load(var("input"), var("j"), DType::F32),
        );

        let if_node = AstNode::If {
            condition: Box::new(eq(
                AstNode::Rem(Box::new(var("j")), Box::new(const_int(2))),
                const_int(0),
            )),
            then_body: Box::new(inner_body),
            else_body: None,
        };

        let inner_loop = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(const_int(64)),
            body: Box::new(if_node),
            parallel: ParallelInfo::default(),
        };

        let one = const_int(1);
        AstNode::Kernel {
            name: Some("branching_kernel".to_string()),
            params: vec![VarDecl {
                name: "gidx0".to_string(),
                dtype: DType::I64,
                mutability: Mutability::Immutable,
                kind: VarKind::GroupId(0),
            }],
            return_type: DType::Tuple(vec![]),
            body: Box::new(inner_loop),
            default_grid_size: [
                Box::new(const_int(16)),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(1)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        }
    }

    // GroupParallelizationSuggester tests

    #[test]
    fn test_group_function_to_kernel() {
        let func = make_simple_function();
        let suggester = GroupParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        assert_eq!(results.len(), 1);

        if let AstNode::Kernel { params, .. } = &results[0].ast {
            assert_eq!(params[0].name, "gidx0");
            assert!(matches!(params[0].kind, VarKind::GroupId(0)));
        }
    }

    #[test]
    fn test_group_rejects_branch() {
        let func = make_function_with_branch();
        let suggester = GroupParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        // 動的分岐を含むので並列化しない
        assert!(results.is_empty());
    }

    #[test]
    fn test_group_kernel_inner_loop() {
        let kernel = make_kernel_with_loop();
        let suggester = GroupParallelizationSuggester::new();

        let results = suggester.suggest(&kernel);
        assert_eq!(results.len(), 1);

        if let AstNode::Kernel { params, .. } = &results[0].ast {
            let group_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::GroupId(_)))
                .collect();
            assert_eq!(group_id_params.len(), 2);
            assert_eq!(group_id_params[1].name, "gidx1");
            assert!(matches!(group_id_params[1].kind, VarKind::GroupId(1)));
        }
    }

    #[test]
    fn test_group_kernel_rejects_branch() {
        let kernel = make_kernel_with_branch_loop();
        let suggester = GroupParallelizationSuggester::new();

        // 動的分岐を含むKernel内ループも並列化しない
        let results = suggester.suggest(&kernel);
        assert!(results.is_empty());
    }

    // LocalParallelizationSuggester tests

    #[test]
    fn test_local_function_to_kernel() {
        let func = make_simple_function();
        let suggester = LocalParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        assert_eq!(results.len(), 1);

        if let AstNode::Kernel { params, .. } = &results[0].ast {
            assert_eq!(params[0].name, "lidx0");
            assert!(matches!(params[0].kind, VarKind::LocalId(0)));
        }
    }

    #[test]
    fn test_local_accepts_branch_in_function() {
        let func = make_function_with_branch();
        let suggester = LocalParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        // ローカル並列化は動的分岐を許可
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_local_kernel_parallelization() {
        let kernel = make_kernel_with_loop();
        let suggester = LocalParallelizationSuggester::new();

        let results = suggester.suggest(&kernel);
        assert_eq!(results.len(), 1);

        if let AstNode::Kernel { params, body, .. } = &results[0].ast {
            let local_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::LocalId(_)))
                .collect();
            assert_eq!(local_id_params.len(), 1);
            // make_kernel_with_loopは既にGroupId(0)を持つため、LocalIdはaxis=1を使用
            assert_eq!(local_id_params[0].name, "lidx1");
            assert!(matches!(local_id_params[0].kind, VarKind::LocalId(1)));
            // 境界チェックなしでStoreノードになる
            assert!(matches!(body.as_ref(), AstNode::Store { .. }));
        }
    }

    #[test]
    fn test_local_accepts_branch_in_kernel() {
        let kernel = make_kernel_with_branch_loop();
        let suggester = LocalParallelizationSuggester::new();

        let results = suggester.suggest(&kernel);
        // ローカル並列化は動的分岐を許可
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_program_with_function() {
        let func = make_simple_function();
        let program = AstNode::Program {
            functions: vec![func],
            execution_waves: vec![],
        };

        let suggester = GroupParallelizationSuggester::new();
        let results = suggester.suggest(&program);

        assert_eq!(results.len(), 1);

        if let AstNode::Program { functions, .. } = &results[0].ast {
            assert_eq!(functions.len(), 1);
            assert!(matches!(functions[0], AstNode::Kernel { .. }));
        }
    }

    #[test]
    fn test_non_parallelizable_loop() {
        let body = AstNode::Assign {
            var: "sum".to_string(),
            value: Box::new(AstNode::Add(
                Box::new(var("sum")),
                Box::new(load(var("input"), var("i"), DType::F32)),
            )),
        };

        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(var("N")),
            body: Box::new(body),
            parallel: ParallelInfo::default(),
        };

        let func = AstNode::Function {
            name: Some("reduce".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(range),
        };

        let group_suggester = GroupParallelizationSuggester::new();
        let group_results = group_suggester.suggest(&func);
        assert!(group_results.is_empty());

        let local_suggester = LocalParallelizationSuggester::new();
        let local_results = local_suggester.suggest(&func);
        assert!(local_results.is_empty());
    }

    /// 既にLocalIdで並列化されたKernelに対してGroupParallelizationを適用したとき、
    /// 軸が衝突しないことを確認するテスト（バグ修正確認）
    #[test]
    fn test_group_does_not_overwrite_local_axis() {
        // LocalId(0)を持つKernelを作成
        let body = store(
            var("output"),
            var("j"),
            load(var("input"), var("j"), DType::F32),
        );

        let inner_loop = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(const_int(64)),
            body: Box::new(body),
            parallel: ParallelInfo::default(),
        };

        let one = const_int(1);
        let kernel = AstNode::Kernel {
            name: Some("local_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "lidx0".to_string(),
                    dtype: DType::I64,
                    mutability: Mutability::Immutable,
                    kind: VarKind::LocalId(0), // LocalId(0)がaxis=0を使用
                },
                VarDecl {
                    name: "input".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(inner_loop),
            default_grid_size: [
                Box::new(one.clone()),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(256)), // LocalId(0)用にサイズ設定済み
                Box::new(one.clone()),
                Box::new(one),
            ],
        };

        let suggester = GroupParallelizationSuggester::new();
        let results = suggester.suggest(&kernel);

        assert_eq!(results.len(), 1);

        if let AstNode::Kernel {
            params,
            default_thread_group_size,
            ..
        } = &results[0].ast
        {
            // LocalId(0)が保持されていることを確認
            let local_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::LocalId(_)))
                .collect();
            assert_eq!(local_id_params.len(), 1);
            assert!(matches!(local_id_params[0].kind, VarKind::LocalId(0)));

            // GroupIdはaxis=1を使用すべき（axis=0はLocalIdが使用中）
            let group_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::GroupId(_)))
                .collect();
            assert_eq!(group_id_params.len(), 1);
            assert_eq!(group_id_params[0].name, "gidx1");
            assert!(matches!(group_id_params[0].kind, VarKind::GroupId(1)));

            // thread_group_size[0]がLocalId用の256のまま保持されていることを確認
            assert!(matches!(
                default_thread_group_size[0].as_ref(),
                AstNode::Const(Literal::I64(256))
            ));
        } else {
            panic!("Expected Kernel");
        }
    }

    /// 既にGroupIdで並列化されたKernelに対してLocalParallelizationを適用したとき、
    /// 軸が衝突しないことを確認するテスト
    #[test]
    fn test_local_does_not_overwrite_group_axis() {
        // GroupId(0)を持つKernelを作成
        let body = store(
            var("output"),
            var("j"),
            load(var("input"), var("j"), DType::F32),
        );

        let inner_loop = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(const_int(64)),
            body: Box::new(body),
            parallel: ParallelInfo::default(),
        };

        let one = const_int(1);
        let kernel = AstNode::Kernel {
            name: Some("group_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "gidx0".to_string(),
                    dtype: DType::I64,
                    mutability: Mutability::Immutable,
                    kind: VarKind::GroupId(0), // GroupId(0)がaxis=0を使用
                },
                VarDecl {
                    name: "input".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32), AddressSpace::Global),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(inner_loop),
            default_grid_size: [
                Box::new(const_int(1024)), // GroupId(0)用にサイズ設定済み
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(const_int(256)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        };

        let suggester = LocalParallelizationSuggester::new();
        let results = suggester.suggest(&kernel);

        assert_eq!(results.len(), 1);

        if let AstNode::Kernel {
            params,
            default_grid_size,
            ..
        } = &results[0].ast
        {
            // GroupId(0)が保持されていることを確認
            let group_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::GroupId(_)))
                .collect();
            assert_eq!(group_id_params.len(), 1);
            assert!(matches!(group_id_params[0].kind, VarKind::GroupId(0)));

            // LocalIdはaxis=1を使用すべき（axis=0はGroupIdが使用中）
            let local_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::LocalId(_)))
                .collect();
            assert_eq!(local_id_params.len(), 1);
            assert_eq!(local_id_params[0].name, "lidx1");
            assert!(matches!(local_id_params[0].kind, VarKind::LocalId(1)));

            // grid_size[0]がGroupId用の1024のまま保持されていることを確認
            assert!(matches!(
                default_grid_size[0].as_ref(),
                AstNode::Const(Literal::I64(1024))
            ));
        } else {
            panic!("Expected Kernel");
        }
    }
}

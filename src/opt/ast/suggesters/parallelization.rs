//! ループ並列化を提案するSuggester
//!
//! 2つのSuggesterを提供:
//! - GlobalParallelizationSuggester: ThreadId使用、動的分岐チェックあり
//! - LocalParallelizationSuggester: LocalId使用、動的分岐チェックなし
//!
//! 両方のSuggesterがFunction→KernelとKernel内ループ並列化の両方に対応。

use crate::ast::{AstNode, Literal, Scope, VarDecl, VarKind};
use crate::opt::ast::{AstSuggestResult, AstSuggester};

use super::parallelization_common::{
    ceil_div, collect_free_variables, const_int, infer_params_from_placeholders,
    is_range_parallelizable, is_range_thread_parallelizable, local_id_param, substitute_var,
    thread_id_param, var,
};
use crate::ast::helper::lt;

/// デフォルトのスレッドグループサイズ
const DEFAULT_THREAD_GROUP_SIZE: usize = 256;

// ============================================================================
// 共通ヘルパー関数
// ============================================================================

/// 境界チェック式を生成
fn build_bound_check(var_name: &str, start: &AstNode, stop: &AstNode) -> AstNode {
    if matches!(start, AstNode::Const(Literal::Int(0))) {
        lt(var(var_name), stop.clone())
    } else {
        let range_size = AstNode::Add(
            Box::new(stop.clone()),
            Box::new(AstNode::Mul(
                Box::new(const_int(-1)),
                Box::new(start.clone()),
            )),
        );
        lt(var(var_name), range_size)
    }
}

/// ループの総イテレーション数を計算
fn compute_total_iterations(start: &AstNode, stop: &AstNode) -> AstNode {
    if matches!(start, AstNode::Const(Literal::Int(0))) {
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

/// 次に使用可能な軸を決定
fn find_next_axis(params: &[VarDecl], kind_filter: fn(&VarKind) -> Option<usize>) -> usize {
    let mut max_axis = 0;
    for param in params {
        if let Some(axis) = kind_filter(&param.kind) {
            max_axis = max_axis.max(axis + 1);
        }
    }
    max_axis.min(2) // 最大3次元
}

/// ThreadIdの軸を取得
fn thread_id_axis(kind: &VarKind) -> Option<usize> {
    match kind {
        VarKind::ThreadId(axis) => Some(*axis),
        _ => None,
    }
}

/// LocalIdの軸を取得
fn local_id_axis(kind: &VarKind) -> Option<usize> {
    match kind {
        VarKind::LocalId(axis) => Some(*axis),
        _ => None,
    }
}

/// grid_sizeを更新
fn update_grid_size(
    current: &[Box<AstNode>; 3],
    axis: usize,
    size: &AstNode,
    thread_group_size: usize,
) -> [Box<AstNode>; 3] {
    let mut new_sizes = current.clone();
    if axis < 3 {
        let tg_size = const_int(thread_group_size as isize);
        let num_groups = ceil_div(size.clone(), tg_size.clone());
        let grid_size = AstNode::Mul(Box::new(num_groups), Box::new(tg_size));
        *new_sizes[axis] = grid_size;
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
// GlobalParallelizationSuggester
// ============================================================================

/// グローバル並列化を提案するSuggester
///
/// ThreadId（get_global_id）を使用してグローバルスレッド並列化を行います。
///
/// **動的分岐チェック: あり**
/// GPUの分岐ダイバージェンスを避けるため、ループ内にIf文があると並列化しません。
///
/// # 対応する変換
///
/// ## Function → Kernel
/// ```text
/// // 変換前
/// Function { body: Range { var: "i", ... } }
/// // 変換後
/// Kernel { params: [gidx0: ThreadId(0), ...], ... }
/// ```
///
/// ## Kernel内ループ → ThreadId追加
/// ```text
/// // 変換前
/// Kernel { params: [gidx0: ThreadId(0)], body: Range { var: "j", ... } }
/// // 変換後
/// Kernel { params: [gidx0: ThreadId(0), gidx1: ThreadId(1)], ... }
/// ```
pub struct GlobalParallelizationSuggester {
    thread_group_size: usize,
}

impl GlobalParallelizationSuggester {
    pub fn new() -> Self {
        Self {
            thread_group_size: DEFAULT_THREAD_GROUP_SIZE,
        }
    }

    pub fn with_thread_group_size(thread_group_size: usize) -> Self {
        Self { thread_group_size }
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
            "GlobalParallelization: Converting Function {:?} to Kernel",
            name
        );

        let gid_name = "gidx0";
        let new_body = substitute_var(loop_body, loop_var, &var(gid_name));
        let bound_check = build_bound_check(gid_name, start, stop);

        let guarded_body = AstNode::If {
            condition: Box::new(bound_check),
            then_body: Box::new(new_body),
            else_body: None,
        };

        let kernel_body = if let AstNode::Block { scope, .. } = loop_body.as_ref() {
            AstNode::Block {
                statements: vec![guarded_body],
                scope: scope.clone(),
            }
        } else {
            AstNode::Block {
                statements: vec![guarded_body],
                scope: Box::new(Scope::new()),
            }
        };

        let total_iterations = compute_total_iterations(start, stop);
        let tg_size = const_int(self.thread_group_size as isize);
        let num_groups = ceil_div(total_iterations, tg_size.clone());
        let grid_size_x = AstNode::Mul(Box::new(num_groups), Box::new(tg_size.clone()));

        let mut kernel_params = vec![thread_id_param(gid_name, 0)];

        if params.is_empty() {
            let free_vars = collect_free_variables(&kernel_body);
            let free_vars: Vec<_> = free_vars.into_iter().filter(|v| v != gid_name).collect();
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
                Box::new(grid_size_x),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
            default_thread_group_size: [
                Box::new(const_int(self.thread_group_size as isize)),
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

        let next_axis = find_next_axis(params, thread_id_axis);
        let gid_name = format!("gidx{}", next_axis);

        log::debug!(
            "GlobalParallelization: Adding ThreadId({}) to Kernel {:?}: {} -> {}",
            next_axis,
            name,
            loop_var,
            gid_name
        );

        let new_body = substitute_var(loop_body, loop_var, &var(&gid_name));
        let bound_check = build_bound_check(&gid_name, start, stop);

        let guarded_body = AstNode::If {
            condition: Box::new(bound_check),
            then_body: Box::new(new_body),
            else_body: None,
        };

        let new_kernel_body = replace_range_with(body, range_node, guarded_body);

        let mut new_params = params.clone();
        new_params.push(thread_id_param(&gid_name, next_axis));

        let total_iterations = compute_total_iterations(start, stop);
        let new_grid_size = update_grid_size(
            default_grid_size,
            next_axis,
            &total_iterations,
            self.thread_group_size,
        );

        // ThreadIdの場合、thread_group_sizeも更新
        let new_thread_group_size = update_thread_group_size(
            default_thread_group_size,
            next_axis,
            &const_int(self.thread_group_size as isize),
        );

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

        let AstNode::Program { functions } = program else {
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
                            },
                            self.name(),
                            format!("Parallelize {} (add ThreadId)", kernel_name),
                        ));
                    }
                }
                _ => {}
            }
        }

        results
    }
}

impl Default for GlobalParallelizationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for GlobalParallelizationSuggester {
    fn name(&self) -> &str {
        "GlobalParallelization"
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
                        "Parallelize (add ThreadId)".to_string(),
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
/// Kernel { params: [gidx0: ThreadId(0)], body: Range { var: "j", ... } }
/// // 変換後
/// Kernel { params: [gidx0: ThreadId(0), lidx0: LocalId(0)], thread_group_size: [256, M, 1] }
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
        let new_body = substitute_var(loop_body, loop_var, &var(lid_name));
        let bound_check = build_bound_check(lid_name, start, stop);

        let guarded_body = AstNode::If {
            condition: Box::new(bound_check),
            then_body: Box::new(new_body),
            else_body: None,
        };

        let kernel_body = if let AstNode::Block { scope, .. } = loop_body.as_ref() {
            AstNode::Block {
                statements: vec![guarded_body],
                scope: scope.clone(),
            }
        } else {
            AstNode::Block {
                statements: vec![guarded_body],
                scope: Box::new(Scope::new()),
            }
        };

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
            start,
            stop,
            body: loop_body,
            ..
        } = range_node
        else {
            return None;
        };

        let next_axis = find_next_axis(params, local_id_axis);
        let lid_name = format!("lidx{}", next_axis);

        log::debug!(
            "LocalParallelization: Adding LocalId({}) to Kernel {:?}: {} -> {}",
            next_axis,
            name,
            loop_var,
            lid_name
        );

        let new_body = substitute_var(loop_body, loop_var, &var(&lid_name));
        let bound_check = build_bound_check(&lid_name, start, stop);

        let guarded_body = AstNode::If {
            condition: Box::new(bound_check),
            then_body: Box::new(new_body),
            else_body: None,
        };

        let new_kernel_body = replace_range_with(body, range_node, guarded_body);

        let mut new_params = params.clone();
        new_params.push(local_id_param(&lid_name, next_axis));

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

        let AstNode::Program { functions } = program else {
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
        DType, Mutability, VarDecl, VarKind,
        helper::{load, store},
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
        };

        AstNode::Function {
            name: Some("kernel_0".to_string()),
            params: vec![
                VarDecl {
                    name: "input".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::F32)),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "N".to_string(),
                    dtype: DType::Int,
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
            condition: Box::new(AstNode::Eq(
                Box::new(AstNode::Rem(Box::new(var("i")), Box::new(const_int(2)))),
                Box::new(const_int(0)),
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
        };

        let one = const_int(1);
        AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "gidx0".to_string(),
                    dtype: DType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::ThreadId(0),
                },
                VarDecl {
                    name: "input".to_string(),
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
            condition: Box::new(AstNode::Eq(
                Box::new(AstNode::Rem(Box::new(var("j")), Box::new(const_int(2)))),
                Box::new(const_int(0)),
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
        };

        let one = const_int(1);
        AstNode::Kernel {
            name: Some("branching_kernel".to_string()),
            params: vec![VarDecl {
                name: "gidx0".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::ThreadId(0),
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

    // GlobalParallelizationSuggester tests

    #[test]
    fn test_global_function_to_kernel() {
        let func = make_simple_function();
        let suggester = GlobalParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        assert_eq!(results.len(), 1);

        if let AstNode::Kernel { params, .. } = &results[0].ast {
            assert_eq!(params[0].name, "gidx0");
            assert!(matches!(params[0].kind, VarKind::ThreadId(0)));
        }
    }

    #[test]
    fn test_global_rejects_branch() {
        let func = make_function_with_branch();
        let suggester = GlobalParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        // 動的分岐を含むので並列化しない
        assert!(results.is_empty());
    }

    #[test]
    fn test_global_kernel_inner_loop() {
        let kernel = make_kernel_with_loop();
        let suggester = GlobalParallelizationSuggester::new();

        let results = suggester.suggest(&kernel);
        assert_eq!(results.len(), 1);

        if let AstNode::Kernel { params, .. } = &results[0].ast {
            let thread_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::ThreadId(_)))
                .collect();
            assert_eq!(thread_id_params.len(), 2);
            assert_eq!(thread_id_params[1].name, "gidx1");
            assert!(matches!(thread_id_params[1].kind, VarKind::ThreadId(1)));
        }
    }

    #[test]
    fn test_global_kernel_rejects_branch() {
        let kernel = make_kernel_with_branch_loop();
        let suggester = GlobalParallelizationSuggester::new();

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
            assert_eq!(local_id_params[0].name, "lidx0");
            assert!(matches!(body.as_ref(), AstNode::If { .. }));
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
        };

        let suggester = GlobalParallelizationSuggester::new();
        let results = suggester.suggest(&program);

        assert_eq!(results.len(), 1);

        if let AstNode::Program { functions } = &results[0].ast {
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
        };

        let func = AstNode::Function {
            name: Some("reduce".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(range),
        };

        let global_suggester = GlobalParallelizationSuggester::new();
        let global_results = global_suggester.suggest(&func);
        assert!(global_results.is_empty());

        let local_suggester = LocalParallelizationSuggester::new();
        let local_results = local_suggester.suggest(&func);
        assert!(local_results.is_empty());
    }
}

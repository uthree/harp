//! Kernel内部ループ並列化を提案するSuggester
//!
//! すでに並列化済みのKernel内にあるRangeループをさらに並列化し、
//! 複数次元の並列化を実現します。LocalId（get_local_id）を使用して
//! 各イテレーションを異なるスレッドに割り当てます。

use crate::ast::{AstNode, Literal};
use crate::opt::ast::{AstSuggestResult, AstSuggester};

use super::parallelization_common::{
    is_range_thread_parallelizable, local_id_param, substitute_var, var,
};
use crate::ast::helper::lt;

/// デフォルトのローカルスレッド数
const DEFAULT_MAX_LOCAL_THREADS: usize = 256;

/// Kernel内部ループ並列化を提案するSuggester
///
/// すでに並列化済みのKernel内にあるRangeループを検出し、
/// LocalIdで並列化する候補を生成します。
///
/// # 変換例
///
/// ```text
/// // 変換前
/// Kernel {
///     params: [gidx0: GroupId(0), lidx0: ThreadId(0), ...],
///     body: Range { var: "ridx2", start: 0, stop: M, step: 1,
///         body: Store(output, ridx2, ...)
///     },
///     thread_group_size: [64, 1, 1],
/// }
///
/// // 変換後
/// Kernel {
///     params: [gidx0: GroupId(0), lidx0: ThreadId(0), lidx2: LocalId(1), ...],
///     body: If {
///         condition: lidx2 < M,
///         then: Store(output, lidx2, ...)
///     },
///     thread_group_size: [64, M, 1],
/// }
/// ```
pub struct InnerLoopParallelizationSuggester {
    /// 最大ローカルスレッド数（将来の機能拡張用）
    #[allow(dead_code)]
    max_local_threads: usize,
}

impl InnerLoopParallelizationSuggester {
    /// 新しいSuggesterを作成（デフォルト設定）
    pub fn new() -> Self {
        Self {
            max_local_threads: DEFAULT_MAX_LOCAL_THREADS,
        }
    }

    /// 最大ローカルスレッド数を指定してSuggesterを作成
    pub fn with_max_local_threads(max_local_threads: usize) -> Self {
        Self { max_local_threads }
    }

    /// Kernelノード内のRangeループを並列化する
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

        // 本体からRangeループを探す
        let (range_node, path) = self.find_parallelizable_range(body)?;

        // Range情報を取得
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

        // ループ変数をLocalId変数名に変換（ridx{n} -> lidx{n}）
        let lid_name = loop_var.replace("ridx", "lidx");
        let lid_name = if lid_name == *loop_var {
            format!("lid_{}", loop_var)
        } else {
            lid_name
        };

        log::debug!(
            "Parallelizing inner loop '{}' -> '{}' in Kernel {:?}",
            loop_var,
            lid_name,
            name
        );

        // ループ変数をLocalIdで置換した本体を作成
        let new_body = substitute_var(loop_body, loop_var, &var(&lid_name));

        // 境界チェック: if lidx < stop { new_body }
        let bound_check = if matches!(start.as_ref(), AstNode::Const(Literal::Int(0))) {
            lt(var(&lid_name), stop.as_ref().clone())
        } else {
            // start != 0 の場合: lidx + start < stop => lidx < stop - start
            let range_size = AstNode::Add(
                Box::new(stop.as_ref().clone()),
                Box::new(AstNode::Mul(
                    Box::new(AstNode::Const(Literal::Int(-1))),
                    Box::new(start.as_ref().clone()),
                )),
            );
            lt(var(&lid_name), range_size)
        };

        let guarded_body = AstNode::If {
            condition: Box::new(bound_check),
            then_body: Box::new(new_body),
            else_body: None,
        };

        // 新しいKernel本体を構築（パスに従って置換）
        let new_kernel_body = self.replace_at_path(body, &path, guarded_body);

        // 次に使用可能なLocalId軸を決定
        let next_local_axis = self.find_next_local_axis(params);

        // Kernel paramsにLocalIdパラメータを追加
        let mut new_params = params.clone();
        new_params.push(local_id_param(&lid_name, next_local_axis));

        // thread_group_sizeを更新（該当する軸のサイズを更新）
        let new_thread_group_size = self.update_thread_group_size(
            default_thread_group_size,
            next_local_axis,
            stop.as_ref(),
        );

        Some(AstNode::Kernel {
            name: name.clone(),
            params: new_params,
            return_type: return_type.clone(),
            body: Box::new(new_kernel_body),
            default_grid_size: default_grid_size.clone(),
            default_thread_group_size: new_thread_group_size,
        })
    }

    /// 並列化可能なRangeループを探す
    ///
    /// 戻り値: (Rangeノード, パスのインデックスリスト)
    fn find_parallelizable_range<'a>(
        &self,
        node: &'a AstNode,
    ) -> Option<(&'a AstNode, Vec<usize>)> {
        self.find_range_recursive(node, vec![])
    }

    fn find_range_recursive<'a>(
        &self,
        node: &'a AstNode,
        path: Vec<usize>,
    ) -> Option<(&'a AstNode, Vec<usize>)> {
        match node {
            AstNode::Range { .. } => {
                // スレッド並列化可能かチェック
                if is_range_thread_parallelizable(node) {
                    Some((node, path))
                } else {
                    // このRangeは並列化不可だが、その内側を探索
                    if let AstNode::Range { body, .. } = node {
                        let mut new_path = path;
                        new_path.push(0); // Range内のbodyへ
                        self.find_range_recursive(body, new_path)
                    } else {
                        None
                    }
                }
            }
            AstNode::Block { statements, .. } => {
                for (idx, stmt) in statements.iter().enumerate() {
                    let mut new_path = path.clone();
                    new_path.push(idx);
                    if let Some(result) = self.find_range_recursive(stmt, new_path) {
                        return Some(result);
                    }
                }
                None
            }
            AstNode::If {
                then_body,
                else_body,
                ..
            } => {
                // thenブランチを探索
                let mut then_path = path.clone();
                then_path.push(0); // then branch
                if let Some(result) = self.find_range_recursive(then_body, then_path) {
                    return Some(result);
                }
                // elseブランチがあれば探索
                if let Some(else_b) = else_body {
                    let mut else_path = path;
                    else_path.push(1); // else branch
                    if let Some(result) = self.find_range_recursive(else_b, else_path) {
                        return Some(result);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// パスに従ってノードを置換
    fn replace_at_path(&self, node: &AstNode, path: &[usize], replacement: AstNode) -> AstNode {
        if path.is_empty() {
            return replacement;
        }

        match node {
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                // Rangeの内側のbodyに対して再帰
                let new_body = self.replace_at_path(body, &path[1..], replacement);
                AstNode::Range {
                    var: var.clone(),
                    start: start.clone(),
                    step: step.clone(),
                    stop: stop.clone(),
                    body: Box::new(new_body),
                }
            }
            AstNode::Block { statements, scope } => {
                let idx = path[0];
                let mut new_statements = statements.clone();
                if idx < new_statements.len() {
                    new_statements[idx] =
                        self.replace_at_path(&statements[idx], &path[1..], replacement);
                }
                AstNode::Block {
                    statements: new_statements,
                    scope: scope.clone(),
                }
            }
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                let branch_idx = path[0];
                if branch_idx == 0 {
                    // then branch
                    let new_then = self.replace_at_path(then_body, &path[1..], replacement);
                    AstNode::If {
                        condition: condition.clone(),
                        then_body: Box::new(new_then),
                        else_body: else_body.clone(),
                    }
                } else if let Some(else_b) = else_body {
                    // else branch
                    let new_else = self.replace_at_path(else_b, &path[1..], replacement);
                    AstNode::If {
                        condition: condition.clone(),
                        then_body: then_body.clone(),
                        else_body: Some(Box::new(new_else)),
                    }
                } else {
                    node.clone()
                }
            }
            _ => node.clone(),
        }
    }

    /// 次に使用可能なLocalId軸を決定
    fn find_next_local_axis(&self, params: &[crate::ast::VarDecl]) -> usize {
        use crate::ast::VarKind;

        let mut max_axis = 0;
        for param in params {
            if let VarKind::LocalId(axis) = &param.kind {
                max_axis = max_axis.max(axis + 1);
            }
            // ThreadIdも考慮（ThreadIdとLocalIdは同じ軸を使用する可能性がある）
            if let VarKind::ThreadId(axis) = &param.kind {
                max_axis = max_axis.max(axis + 1);
            }
        }
        max_axis.min(2) // 最大3次元（0, 1, 2）
    }

    /// thread_group_sizeを更新
    fn update_thread_group_size(
        &self,
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

    /// Program内のKernelを処理
    fn process_program(&self, program: &AstNode) -> Vec<AstSuggestResult> {
        let mut results = Vec::new();

        let AstNode::Program { functions } = program else {
            return results;
        };

        for (idx, func) in functions.iter().enumerate() {
            if let AstNode::Kernel { name, .. } = func
                && let Some(parallelized) = self.try_parallelize_kernel(func)
            {
                let mut new_functions = functions.clone();
                new_functions[idx] = parallelized;

                let kernel_name = name.clone().unwrap_or_else(|| format!("kernel_{}", idx));

                results.push(AstSuggestResult::with_description(
                    AstNode::Program {
                        functions: new_functions,
                    },
                    self.name(),
                    format!("Parallelize inner loop in {} with LocalId", kernel_name),
                ));
            }
        }

        results
    }
}

impl Default for InnerLoopParallelizationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for InnerLoopParallelizationSuggester {
    fn name(&self) -> &str {
        "InnerLoopParallelization"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        match ast {
            AstNode::Program { .. } => self.process_program(ast),
            AstNode::Kernel { .. } => {
                if let Some(parallelized) = self.try_parallelize_kernel(ast) {
                    vec![AstSuggestResult::with_description(
                        parallelized,
                        self.name(),
                        "Parallelize inner loop with LocalId".to_string(),
                    )]
                } else {
                    vec![]
                }
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{const_int, load, store};
    use crate::ast::{DType, Mutability, VarDecl, VarKind};
    // var is provided by parallelization_common via super::*

    fn make_kernel_with_inner_loop() -> AstNode {
        // Kernel with an inner Range loop
        // kernel void test_kernel(gidx0, lidx0, input, output) {
        //     for ridx2 in 0..64 {
        //         output[ridx2] = input[ridx2]
        //     }
        // }
        let body = store(
            var("output"),
            var("ridx2"),
            load(var("input"), var("ridx2"), DType::F32),
        );

        let inner_loop = AstNode::Range {
            var: "ridx2".to_string(),
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
                    kind: VarKind::GroupId(0),
                },
                VarDecl {
                    name: "lidx0".to_string(),
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
                Box::new(const_int(64)),
                Box::new(one.clone()),
                Box::new(one),
            ],
        }
    }

    #[test]
    fn test_inner_loop_parallelization() {
        let kernel = make_kernel_with_inner_loop();
        let suggester = InnerLoopParallelizationSuggester::new();

        let results = suggester.suggest(&kernel);
        assert_eq!(results.len(), 1);

        let result = &results[0];
        if let AstNode::Kernel { params, body, .. } = &result.ast {
            // LocalIdパラメータが追加されていること
            let local_id_params: Vec<_> = params
                .iter()
                .filter(|p| matches!(p.kind, VarKind::LocalId(_)))
                .collect();
            assert_eq!(local_id_params.len(), 1);
            assert_eq!(local_id_params[0].name, "lidx2");

            // 本体がIfになっていること（境界チェック）
            assert!(matches!(body.as_ref(), AstNode::If { .. }));
        } else {
            panic!("Expected Kernel");
        }
    }

    #[test]
    fn test_kernel_without_parallelizable_loop() {
        // Kernel with a loop that has external writes (not parallelizable)
        let body = AstNode::Assign {
            var: "sum".to_string(),
            value: Box::new(AstNode::Add(
                Box::new(var("sum")),
                Box::new(load(var("input"), var("i"), DType::F32)),
            )),
        };

        let inner_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(const_int(64)),
            body: Box::new(body),
        };

        let one = const_int(1);
        let kernel = AstNode::Kernel {
            name: Some("reduce_kernel".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(inner_loop),
            default_grid_size: [
                Box::new(one.clone()),
                Box::new(one.clone()),
                Box::new(one.clone()),
            ],
            default_thread_group_size: [
                Box::new(one.clone()),
                Box::new(one.clone()),
                Box::new(one),
            ],
        };

        let suggester = InnerLoopParallelizationSuggester::new();
        let results = suggester.suggest(&kernel);

        // 並列化不可能なので結果は空
        assert!(results.is_empty());
    }
}

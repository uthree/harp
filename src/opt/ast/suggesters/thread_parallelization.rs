//! スレッド単位の並列化を提案するSuggester
//!
//! Function内の最外側Rangeループを解析し、並列化可能な場合は
//! Kernelへの変換を提案します。各イテレーションが1スレッドで実行されます。

use crate::ast::{AstNode, Literal, Scope, helper::lt};
use crate::opt::ast::{AstSuggestResult, AstSuggester};

use super::parallelization_common::{
    ceil_div, collect_free_variables, const_int, infer_params_from_placeholders,
    is_range_thread_parallelizable, substitute_var, thread_id_param, var,
};

/// デフォルトのスレッドグループサイズ
const DEFAULT_THREAD_GROUP_SIZE: usize = 256;

/// スレッド単位の並列化を提案するSuggester
///
/// Function内の最外側Rangeループが並列化可能な場合、
/// そのFunctionをKernelに変換する候補を生成します。
///
/// # 変換例
///
/// ```text
/// // 変換前
/// Function {
///     name: "kernel_0",
///     params: [input: Ptr<F32>, output: Ptr<F32>, N: Int],
///     body: Range { var: "i", start: 0, stop: N, step: 1,
///         body: Store(output, i, Load(input, i))
///     }
/// }
///
/// // 変換後
/// Kernel {
///     name: "kernel_0",
///     params: [gidx0: ThreadId(0), input: Ptr<F32>, output: Ptr<F32>, N: Int],
///     body: If {
///         condition: gidx0 < N,
///         then: Store(output, gidx0, Load(input, gidx0))
///     },
///     grid_size: [ceil_div(N, 256) * 256, 1, 1],
///     thread_group_size: [256, 1, 1],
/// }
/// ```
pub struct ThreadParallelizationSuggester {
    /// スレッドグループサイズ
    thread_group_size: usize,
}

impl ThreadParallelizationSuggester {
    /// 新しいSuggesterを作成（デフォルト設定）
    pub fn new() -> Self {
        Self {
            thread_group_size: DEFAULT_THREAD_GROUP_SIZE,
        }
    }

    /// スレッドグループサイズを指定してSuggesterを作成
    pub fn with_thread_group_size(thread_group_size: usize) -> Self {
        Self { thread_group_size }
    }

    /// FunctionノードをKernelに変換可能か判定し、変換する
    fn try_convert_function(&self, func: &AstNode) -> Option<AstNode> {
        let AstNode::Function {
            name,
            params,
            return_type,
            body,
        } = func
        else {
            return None;
        };

        // bodyがRangeループか、またはBlockの中にRangeがあるか確認
        let range_node = self.find_outermost_range(body)?;

        // スレッド並列化可能かチェック（動的分岐を含む場合は不可）
        if !is_range_thread_parallelizable(range_node) {
            log::trace!(
                "Function {:?}: Range loop is not thread-parallelizable",
                name
            );
            return None;
        }

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

        log::debug!(
            "Converting Function {:?} to Kernel with thread parallelization",
            name
        );

        // グローバルスレッドID変数名（gidx0を使用）
        let gid_name = "gidx0";

        // ループ変数をgidx0で置換した本体を作成
        let new_body = substitute_var(loop_body, loop_var, &var(gid_name));

        // 境界チェック: if gidx0 < stop { new_body }
        // startが0でない場合は gidx0 + start として扱う必要があるが、
        // 簡略化のためstart=0を前提とする
        let bound_check = if matches!(start.as_ref(), AstNode::Const(Literal::Int(0))) {
            lt(var(gid_name), stop.as_ref().clone())
        } else {
            // start != 0 の場合: gidx0 + start < stop => gidx0 < stop - start
            let range_size = AstNode::Add(
                Box::new(stop.as_ref().clone()),
                Box::new(AstNode::Mul(
                    Box::new(const_int(-1)),
                    Box::new(start.as_ref().clone()),
                )),
            );
            lt(var(gid_name), range_size)
        };

        let guarded_body = AstNode::If {
            condition: Box::new(bound_check),
            then_body: Box::new(new_body),
            else_body: None,
        };

        // スコープを作成（元のbodyがBlockならそのスコープを使用）
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

        // グリッドサイズを計算
        // startが0の場合: ceil_div(stop, tg_size) * tg_size
        // それ以外: ceil_div(stop - start, tg_size) * tg_size
        let total_iterations = if matches!(start.as_ref(), AstNode::Const(Literal::Int(0))) {
            stop.as_ref().clone()
        } else {
            AstNode::Add(
                Box::new(stop.as_ref().clone()),
                Box::new(AstNode::Mul(
                    Box::new(const_int(-1)),
                    Box::new(start.as_ref().clone()),
                )),
            )
        };

        let tg_size = const_int(self.thread_group_size as isize);
        let num_groups = ceil_div(total_iterations, tg_size.clone());
        let grid_size_x = AstNode::Mul(Box::new(num_groups), Box::new(tg_size.clone()));

        // Kernel paramsを作成（gidx0を先頭に追加）
        let mut kernel_params = vec![thread_id_param(gid_name, 0)];

        // 元のFunctionのparamsが空の場合、本体から自由変数を収集してパラメータを生成
        if params.is_empty() {
            // kernel_bodyから自由変数を収集
            let free_vars = collect_free_variables(&kernel_body);
            // gidx0は除外（既にパラメータとして追加済み）
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

    /// 最外側のRangeループを見つける
    fn find_outermost_range<'a>(&self, node: &'a AstNode) -> Option<&'a AstNode> {
        match node {
            AstNode::Range { .. } => Some(node),
            AstNode::Block { statements, .. } => {
                // Blockの中で最初のRangeを探す
                for stmt in statements {
                    if let Some(range) = self.find_outermost_range(stmt) {
                        return Some(range);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Program内のFunctionを処理
    fn process_program(&self, program: &AstNode) -> Vec<AstSuggestResult> {
        let mut results = Vec::new();

        let AstNode::Program { functions } = program else {
            return results;
        };

        // 各Functionに対して並列化を試みる
        for (idx, func) in functions.iter().enumerate() {
            if let Some(kernel) = self.try_convert_function(func) {
                // 変換成功: 新しいProgramを作成
                let mut new_functions = functions.clone();
                new_functions[idx] = kernel;

                let func_name = if let AstNode::Function { name, .. } = func {
                    name.clone().unwrap_or_else(|| format!("function_{}", idx))
                } else {
                    format!("function_{}", idx)
                };

                results.push(AstSuggestResult::with_description(
                    AstNode::Program {
                        functions: new_functions,
                    },
                    self.name(),
                    format!(
                        "Convert {} to thread-parallel Kernel (tg_size={})",
                        func_name, self.thread_group_size
                    ),
                ));
            }
        }

        results
    }
}

impl Default for ThreadParallelizationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for ThreadParallelizationSuggester {
    fn name(&self) -> &str {
        "ThreadParallelization"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        match ast {
            AstNode::Program { .. } => self.process_program(ast),
            AstNode::Function { .. } => {
                // 単独のFunctionの場合、Programでラップして処理
                if let Some(kernel) = self.try_convert_function(ast) {
                    vec![AstSuggestResult::with_description(
                        kernel,
                        self.name(),
                        format!(
                            "Convert to thread-parallel Kernel (tg_size={})",
                            self.thread_group_size
                        ),
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
    use crate::ast::{
        DType, Mutability, VarDecl, VarKind,
        helper::{load, store},
    };

    fn make_simple_function() -> AstNode {
        // Function that loads from input and stores to output
        // for i in 0..N { output[i] = input[i] }
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

    #[test]
    fn test_thread_parallelization() {
        let func = make_simple_function();
        let suggester = ThreadParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        assert_eq!(results.len(), 1);

        let result = &results[0];
        assert!(matches!(result.ast, AstNode::Kernel { .. }));

        if let AstNode::Kernel { params, .. } = &result.ast {
            // 最初のパラメータがThreadId
            assert_eq!(params[0].name, "gidx0");
            assert!(matches!(params[0].kind, VarKind::ThreadId(0)));
        }
    }

    #[test]
    fn test_program_with_function() {
        let func = make_simple_function();
        let program = AstNode::Program {
            functions: vec![func],
        };

        let suggester = ThreadParallelizationSuggester::new();
        let results = suggester.suggest(&program);

        assert_eq!(results.len(), 1);

        if let AstNode::Program { functions } = &results[0].ast {
            assert_eq!(functions.len(), 1);
            assert!(matches!(functions[0], AstNode::Kernel { .. }));
        }
    }
}

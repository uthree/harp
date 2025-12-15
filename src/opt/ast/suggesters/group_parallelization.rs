//! グループ単位の並列化を提案するSuggester
//!
//! タイル化されたループ（外側ループ + 内側ループ）を解析し、
//! 外側ループをグループID、内側ループをスレッドIDとして
//! Kernelへの変換を提案します。
//!
//! LoopTilingSuggesterとの組み合わせで使用することを想定しています。

use crate::ast::{AstNode, Literal, Scope};
use crate::opt::ast::{AstSuggestResult, AstSuggester};

use super::parallelization_common::{
    collect_free_variables, const_int, group_id_param, infer_params_from_placeholders,
    substitute_var, thread_id_param, var,
};

/// グループ単位の並列化を提案するSuggester
///
/// タイル化された2重ループ（外側ループと内側ループ）を検出し、
/// - 外側ループ → GroupId（スレッドグループのインデックス）
/// - 内側ループ → ThreadId（スレッドグループ内のスレッドインデックス）
///
/// として並列化します。
///
/// # 変換例
///
/// ```text
/// // 変換前（タイル化後のループ）
/// Function {
///     body: Range { var: "ridx0_outer", start: 0, stop: N/tile, step: 1,
///         body: Range { var: "ridx0_inner", start: 0, stop: tile, step: 1,
///             body: {
///                 ridx0 = ridx0_outer * tile + ridx0_inner
///                 Store(output, ridx0, Load(input, ridx0))
///             }
///         }
///     }
/// }
///
/// // 変換後
/// Kernel {
///     params: [gidx0: GroupId(0), lidx0: ThreadId(0), ...],
///     body: {
///         ridx0 = gidx0 * tile + lidx0
///         Store(output, ridx0, Load(input, ridx0))
///     },
///     grid_size: [N/tile, 1, 1],
///     thread_group_size: [tile, 1, 1],
/// }
/// ```
///
/// 境界チェックは不要です（グリッドサイズとスレッドグループサイズで正確に制御）。
pub struct GroupParallelizationSuggester;

impl GroupParallelizationSuggester {
    /// 新しいSuggesterを作成
    pub fn new() -> Self {
        Self
    }

    /// タイル化された2重ループを検出
    ///
    /// 外側ループの本体が直接Rangeノード、または最初の要素がRangeのBlockである場合を検出
    fn find_tiled_loop<'a>(&self, node: &'a AstNode) -> Option<TiledLoopInfo<'a>> {
        match node {
            AstNode::Range {
                var: outer_var,
                start: outer_start,
                stop: outer_stop,
                body: outer_body,
                ..
            } => {
                // 外側ループの本体を解析
                let inner_range = match outer_body.as_ref() {
                    AstNode::Range { .. } => Some(outer_body.as_ref()),
                    AstNode::Block { statements, .. } => {
                        // Blockの最初または2番目の要素がRangeの場合
                        // (最初がAssignの場合もある)
                        statements
                            .iter()
                            .find(|s| matches!(s, AstNode::Range { .. }))
                    }
                    _ => None,
                }?;

                if let AstNode::Range {
                    var: inner_var,
                    start: inner_start,
                    stop: inner_stop,
                    body: inner_body,
                    ..
                } = inner_range
                {
                    // タイル化パターンかどうかチェック
                    // - 変数名が _outer/_inner のサフィックスを持つ
                    // - または ridx で始まり、異なる番号を持つ
                    let is_tiled_pattern = self.is_tiled_variable_pair(outer_var, inner_var);

                    if is_tiled_pattern {
                        return Some(TiledLoopInfo {
                            outer_var: outer_var.as_str(),
                            outer_start: outer_start.as_ref(),
                            outer_stop: outer_stop.as_ref(),
                            inner_var: inner_var.as_str(),
                            inner_start: inner_start.as_ref(),
                            inner_stop: inner_stop.as_ref(),
                            inner_body: inner_body.as_ref(),
                        });
                    }
                }
                None
            }
            AstNode::Block { statements, .. } => {
                // Blockの中で最初のタイル化ループを探す
                for stmt in statements {
                    if let Some(info) = self.find_tiled_loop(stmt) {
                        return Some(info);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// 変数名がタイル化パターン（outer/inner）かどうかを判定
    fn is_tiled_variable_pair(&self, outer_var: &str, inner_var: &str) -> bool {
        // パターン1: ridxN_outer / ridxN_inner
        if (outer_var.contains("_outer") || outer_var.contains("outer"))
            && (inner_var.contains("_inner") || inner_var.contains("inner"))
        {
            return true;
        }

        // パターン2: 連続するridx番号（ridx0とridx1など）
        if outer_var.starts_with("ridx") && inner_var.starts_with("ridx") {
            let outer_num: Option<usize> = outer_var[4..].parse().ok();
            let inner_num: Option<usize> = inner_var[4..].parse().ok();
            if let (Some(o), Some(i)) = (outer_num, inner_num) {
                // 連続する番号または同じ番号（_outer/_innerなしの場合）
                return (i == o + 1) || (outer_var != inner_var);
            }
        }

        false
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

        // タイル化された2重ループを検出
        let tiled_loop = self.find_tiled_loop(body)?;

        // タイル化されたループは構造的に並列化可能と仮定
        // (LoopTilingSuggesterによって生成されたループは並列化安全)
        //
        // より厳密なチェックが必要な場合は、派生変数の依存関係を追跡する
        // 必要があるが、現時点ではタイル化パターンを信頼する

        log::debug!(
            "Converting Function {:?} to Kernel with group parallelization",
            name
        );

        // グループID、ローカルスレッドID変数名（gidx0, lidx0）
        let group_id_name = "gidx0";
        let local_id_name = "lidx0";

        // 外側ループ変数をgidx0で置換
        let body_with_group = substitute_var(
            tiled_loop.inner_body,
            tiled_loop.outer_var,
            &var(group_id_name),
        );
        // 内側ループ変数をlidx0で置換
        let new_body = substitute_var(&body_with_group, tiled_loop.inner_var, &var(local_id_name));

        // グリッドサイズ = 外側ループの反復回数
        let grid_size_x = if matches!(tiled_loop.outer_start, AstNode::Const(Literal::Int(0))) {
            tiled_loop.outer_stop.clone()
        } else {
            AstNode::Add(
                Box::new(tiled_loop.outer_stop.clone()),
                Box::new(AstNode::Mul(
                    Box::new(const_int(-1)),
                    Box::new(tiled_loop.outer_start.clone()),
                )),
            )
        };

        // スレッドグループサイズ = 内側ループの反復回数（タイルサイズ）
        let thread_group_size_x =
            if matches!(tiled_loop.inner_start, AstNode::Const(Literal::Int(0))) {
                tiled_loop.inner_stop.clone()
            } else {
                AstNode::Add(
                    Box::new(tiled_loop.inner_stop.clone()),
                    Box::new(AstNode::Mul(
                        Box::new(const_int(-1)),
                        Box::new(tiled_loop.inner_start.clone()),
                    )),
                )
            };

        // スコープを作成（境界チェック不要：グリッドサイズとスレッドグループサイズで正確に制御）
        let kernel_body = if let AstNode::Block { scope, .. } = tiled_loop.inner_body {
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

        // Kernel paramsを作成（gidx0, lidx0を先頭に追加）
        let mut kernel_params = vec![
            group_id_param(group_id_name, 0),
            thread_id_param(local_id_name, 0),
        ];

        // 元のFunctionのparamsが空の場合、本体から自由変数を収集してパラメータを生成
        if params.is_empty() {
            // kernel_bodyから自由変数を収集
            let free_vars = collect_free_variables(&kernel_body);
            // gidx0, lidx0は除外（既にパラメータとして追加済み）
            let free_vars: Vec<_> = free_vars
                .into_iter()
                .filter(|v| v != group_id_name && v != local_id_name)
                .collect();
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
                Box::new(thread_group_size_x),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
        })
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
                    format!("Convert {} to group-parallel Kernel", func_name),
                ));
            }
        }

        results
    }
}

/// タイル化されたループの情報
struct TiledLoopInfo<'a> {
    outer_var: &'a str,
    outer_start: &'a AstNode,
    outer_stop: &'a AstNode,
    inner_var: &'a str,
    inner_start: &'a AstNode,
    inner_stop: &'a AstNode,
    inner_body: &'a AstNode,
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
                if let Some(kernel) = self.try_convert_function(ast) {
                    vec![AstSuggestResult::with_description(
                        kernel,
                        self.name(),
                        "Convert to group-parallel Kernel".to_string(),
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
        DType, Mutability, Scope, VarDecl, VarKind,
        helper::{assign, load, store},
    };

    fn make_tiled_function() -> AstNode {
        // タイル化されたループを持つFunction
        // for ridx1 in 0..N/64 { for ridx2 in 0..64 { ridx0 = ridx1*64 + ridx2; output[ridx0] = input[ridx0] } }
        let tile_size = 64;
        let n_tiles = AstNode::Idiv(Box::new(var("N")), Box::new(const_int(tile_size)));

        // ridx0 = ridx1 * 64 + ridx2
        let ridx0_expr = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(var("ridx1")),
                Box::new(const_int(tile_size)),
            )),
            Box::new(var("ridx2")),
        );

        let mut inner_scope = Scope::new();
        inner_scope
            .declare("ridx0".to_string(), DType::Int, Mutability::Mutable)
            .unwrap();

        let inner_body = AstNode::Block {
            statements: vec![
                assign("ridx0", ridx0_expr),
                store(
                    var("output"),
                    var("ridx0"),
                    load(var("input"), var("ridx0"), DType::F32),
                ),
            ],
            scope: Box::new(inner_scope),
        };

        let inner_loop = AstNode::Range {
            var: "ridx2".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(const_int(tile_size)),
            body: Box::new(inner_body),
        };

        let outer_loop = AstNode::Range {
            var: "ridx1".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(n_tiles),
            body: Box::new(inner_loop),
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
            body: Box::new(outer_loop),
        }
    }

    #[test]
    fn test_group_parallelization() {
        let func = make_tiled_function();
        let suggester = GroupParallelizationSuggester::new();

        let results = suggester.suggest(&func);
        assert_eq!(results.len(), 1);

        let result = &results[0];
        assert!(matches!(result.ast, AstNode::Kernel { .. }));

        if let AstNode::Kernel { params, .. } = &result.ast {
            // 最初の2つのパラメータがGroupIdとThreadId
            assert_eq!(params[0].name, "gidx0");
            assert!(matches!(params[0].kind, VarKind::GroupId(0)));
            assert_eq!(params[1].name, "lidx0");
            assert!(matches!(params[1].kind, VarKind::ThreadId(0)));
        }
    }

    #[test]
    fn test_is_tiled_variable_pair() {
        let suggester = GroupParallelizationSuggester::new();

        // ridx pattern
        assert!(suggester.is_tiled_variable_pair("ridx1", "ridx2"));
        assert!(suggester.is_tiled_variable_pair("ridx0", "ridx1"));

        // _outer/_inner pattern
        assert!(suggester.is_tiled_variable_pair("i_outer", "i_inner"));
        assert!(suggester.is_tiled_variable_pair("outer_i", "inner_i"));

        // Not tiled
        assert!(!suggester.is_tiled_variable_pair("i", "j")); // different vars, no pattern
    }
}

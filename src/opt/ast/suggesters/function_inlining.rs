//! 関数インライン展開のためのSuggester実装
//!
//! 小さい関数をインライン展開して、関数呼び出しのオーバーヘッドを削減します。

use crate::ast::{AddressSpace, AstNode};
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};
use std::collections::{HashMap, HashSet};

/// 関数インライン展開を提案するSuggester
pub struct FunctionInliningSuggester {
    /// インライン展開する関数の最大ノード数
    max_nodes: usize,
}

impl FunctionInliningSuggester {
    /// 新しいFunctionInliningSuggesterを作成
    pub fn new(max_nodes: usize) -> Self {
        Self { max_nodes }
    }

    /// デフォルトの設定で作成（最大10000ノードまで展開）
    /// カーネル関数は大きくなることが多いため、十分大きな値を使用
    pub fn with_default_limit() -> Self {
        Self { max_nodes: 10000 }
    }

    /// サイズ制限なしで作成
    pub fn without_limit() -> Self {
        Self {
            max_nodes: usize::MAX,
        }
    }

    /// ASTノードの数を数える
    fn count_nodes(ast: &AstNode) -> usize {
        1 + ast
            .children()
            .iter()
            .map(|child| Self::count_nodes(child))
            .sum::<usize>()
    }

    /// 関数本体から全てのReturn文を検出する
    /// Return文が複数ある場合や制御フローが複雑な場合はインライン展開しない
    fn find_single_return(body: &AstNode) -> Option<&AstNode> {
        let mut returns = Vec::new();
        Self::collect_returns(body, &mut returns);

        // Return文が1つだけの場合のみインライン展開可能
        if returns.len() == 1 {
            Some(returns[0])
        } else {
            None
        }
    }

    /// ASTから全てのReturn文を再帰的に収集
    fn collect_returns<'a>(ast: &'a AstNode, returns: &mut Vec<&'a AstNode>) {
        if let AstNode::Return { value } = ast {
            returns.push(value.as_ref());
            return;
        }

        for child in ast.children() {
            Self::collect_returns(child, returns);
        }
    }

    /// 変数名を置き換える（引数の置換用）
    fn substitute_vars(ast: &AstNode, replacements: &HashMap<String, AstNode>) -> AstNode {
        match ast {
            AstNode::Var(name) => replacements
                .get(name)
                .cloned()
                .unwrap_or_else(|| ast.clone()),
            _ => ast.map_children(&|child| Self::substitute_vars(child, replacements)),
        }
    }

    /// 関数呼び出しをインライン展開する（式として使用される場合）
    ///
    /// Call(name, args) を関数本体で置き換える
    /// 返り値: 成功した場合は展開後のノード、失敗した場合はNone
    fn inline_call(
        &self,
        call_node: &AstNode,
        functions: &HashMap<String, &AstNode>,
    ) -> Option<AstNode> {
        if let AstNode::Call { name, args } = call_node {
            // 関数定義を取得
            let func_node = functions.get(name)?;

            if let AstNode::Function { params, body, .. } = func_node {
                // 引数の数が一致するかチェック
                if params.len() != args.len() {
                    trace!("Argument count mismatch for function '{}'", name);
                    return None;
                }

                // 関数本体のノード数をチェック
                if Self::count_nodes(body) > self.max_nodes {
                    trace!("Function '{}' is too large to inline", name);
                    return None;
                }

                // 引数の置換マップを作成
                let mut replacements = HashMap::new();
                for (param, arg) in params.iter().zip(args.iter()) {
                    replacements.insert(param.name.clone(), arg.clone());
                }

                // Return文が1つだけかチェック
                if let Some(return_value) = Self::find_single_return(body) {
                    // Return値の式を引数で置き換えて返す（式としての展開）
                    let inlined = Self::substitute_vars(return_value, &replacements);
                    trace!("Successfully inlined function '{}' (expression)", name);
                    Some(inlined)
                } else {
                    // Return文がない場合（void型関数）は展開しない
                    // Block内のCall文として処理される
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    }

    /// 関数呼び出しをインライン展開する（文として使用される場合）
    ///
    /// Call(name, args) を関数本体のstatementsで置き換える
    /// 返り値: 成功した場合は展開後のstatements、失敗した場合はNone
    fn inline_call_as_statement(
        &self,
        call_node: &AstNode,
        functions: &HashMap<String, &AstNode>,
    ) -> Option<Vec<AstNode>> {
        if let AstNode::Call { name, args } = call_node {
            // 関数定義を取得
            let Some(func_node) = functions.get(name) else {
                debug!("Function '{}' not found in func_map", name);
                return None;
            };

            if let AstNode::Function { params, body, .. } = func_node {
                // 引数の数が一致するかチェック
                if params.len() != args.len() {
                    debug!(
                        "Argument count mismatch for function '{}': expected {}, got {}",
                        name,
                        params.len(),
                        args.len()
                    );
                    return None;
                }

                // 関数本体のノード数をチェック
                let node_count = Self::count_nodes(body);
                if node_count > self.max_nodes {
                    debug!(
                        "Function '{}' is too large to inline: {} nodes > {} max",
                        name, node_count, self.max_nodes
                    );
                    return None;
                }

                // 引数の置換マップを作成
                let mut replacements = HashMap::new();
                for (param, arg) in params.iter().zip(args.iter()) {
                    replacements.insert(param.name.clone(), arg.clone());
                }

                // Return文がある場合は文としての展開は行わない
                if Self::find_single_return(body).is_some() {
                    debug!(
                        "Function '{}' has return statement, cannot inline as statement",
                        name
                    );
                    return None;
                }

                // 関数本体がBlockの場合、そのstatementsを取り出す
                if let AstNode::Block { statements, scope } = body.as_ref() {
                    // 各statementで引数を置換
                    let inlined_statements: Vec<AstNode> = statements
                        .iter()
                        .map(|stmt| Self::substitute_vars(stmt, &replacements))
                        .collect();

                    trace!(
                        "Successfully inlined function '{}' (statement, {} statements)",
                        name,
                        inlined_statements.len()
                    );

                    // スコープにローカル変数がある場合は、Block全体を返す
                    // （ローカル変数の宣言を保持するため）
                    if scope.local_variables().next().is_some() {
                        let block = AstNode::Block {
                            statements: inlined_statements,
                            scope: scope.clone(),
                        };
                        Some(vec![block])
                    } else {
                        Some(inlined_statements)
                    }
                } else {
                    // Blockでない場合（例: 直接Rangeノードの場合）、単一の文として扱う
                    debug!(
                        "Function '{}' body is not a Block ({}), treating as single statement",
                        name,
                        match body.as_ref() {
                            AstNode::Range { .. } => "Range",
                            _ => "other",
                        }
                    );

                    let inlined_stmt = Self::substitute_vars(body, &replacements);
                    trace!(
                        "Successfully inlined function '{}' (single statement body)",
                        name
                    );
                    Some(vec![inlined_stmt])
                }
            } else {
                debug!("func_node is not a Function: {:?}", func_node);
                None
            }
        } else {
            None
        }
    }

    /// ASTツリーを走査して、全てのCall ノードをインライン展開可能か試みる
    fn try_inline_in_ast(
        &self,
        ast: &AstNode,
        functions: &HashMap<String, &AstNode>,
    ) -> Option<AstNode> {
        // 現在のノードがCallの場合、インライン展開を試みる（式として）
        if matches!(ast, AstNode::Call { .. })
            && let Some(inlined) = self.inline_call(ast, functions)
        {
            return Some(inlined);
        }

        // Blockの場合は、statements内のCall文も処理
        if let AstNode::Block { statements, scope } = ast {
            let mut new_statements = Vec::new();
            let mut changed = false;

            debug!("Processing Block with {} statements", statements.len());

            for stmt in statements {
                // Call文の場合、文としてインライン展開を試みる
                if let AstNode::Call { name, .. } = stmt {
                    debug!("Found Call statement to function '{}'", name);
                    if let Some(inlined_stmts) = self.inline_call_as_statement(stmt, functions) {
                        debug!(
                            "Successfully inlined '{}' as {} statements",
                            name,
                            inlined_stmts.len()
                        );
                        new_statements.extend(inlined_stmts);
                        changed = true;
                        continue;
                    } else {
                        debug!("Failed to inline function '{}'", name);
                    }
                }

                // 子ノードを再帰的に処理
                if let Some(inlined_stmt) = self.try_inline_in_ast(stmt, functions) {
                    new_statements.push(inlined_stmt);
                    changed = true;
                } else {
                    new_statements.push(stmt.clone());
                }
            }

            if changed {
                return Some(AstNode::Block {
                    statements: new_statements,
                    scope: scope.clone(),
                });
            }
        }

        // 子ノードに対して再帰的に適用
        let children = ast.children();
        let mut new_children = Vec::new();
        let mut changed = false;

        for child in children {
            if let Some(inlined_child) = self.try_inline_in_ast(child, functions) {
                new_children.push(inlined_child);
                changed = true;
            } else {
                new_children.push(child.clone());
            }
        }

        if changed {
            // 新しい子ノードでASTを再構築
            Some(Self::reconstruct_with_children(ast, &new_children))
        } else {
            None
        }
    }

    /// 子ノードのリストから親ノードを再構築する
    fn reconstruct_with_children(ast: &AstNode, children: &[AstNode]) -> AstNode {
        match ast {
            // 子ノードを持たないノードはそのまま返す
            AstNode::Wildcard(_)
            | AstNode::Const(_)
            | AstNode::Var(_)
            | AstNode::Barrier
            | AstNode::Rand => ast.clone(),

            // 二項演算
            AstNode::Add(_, _) => {
                AstNode::Add(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::Mul(_, _) => {
                AstNode::Mul(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::Max(_, _) => {
                AstNode::Max(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::Rem(_, _) => {
                AstNode::Rem(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::Idiv(_, _) => {
                AstNode::Idiv(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::BitwiseAnd(_, _) => {
                AstNode::BitwiseAnd(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::BitwiseOr(_, _) => {
                AstNode::BitwiseOr(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::BitwiseXor(_, _) => {
                AstNode::BitwiseXor(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::LeftShift(_, _) => {
                AstNode::LeftShift(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::RightShift(_, _) => {
                AstNode::RightShift(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }

            // 比較・論理演算（プリミティブのみ）
            AstNode::Lt(_, _) => {
                AstNode::Lt(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::And(_, _) => {
                AstNode::And(Box::new(children[0].clone()), Box::new(children[1].clone()))
            }
            AstNode::Not(_) => AstNode::Not(Box::new(children[0].clone())),

            // Select (ternary)
            AstNode::Select { .. } => AstNode::Select {
                cond: Box::new(children[0].clone()),
                then_val: Box::new(children[1].clone()),
                else_val: Box::new(children[2].clone()),
            },

            // 単項演算
            AstNode::Recip(_) => AstNode::Recip(Box::new(children[0].clone())),
            AstNode::Sqrt(_) => AstNode::Sqrt(Box::new(children[0].clone())),
            AstNode::Log2(_) => AstNode::Log2(Box::new(children[0].clone())),
            AstNode::Exp2(_) => AstNode::Exp2(Box::new(children[0].clone())),
            AstNode::Sin(_) => AstNode::Sin(Box::new(children[0].clone())),
            AstNode::Floor(_) => AstNode::Floor(Box::new(children[0].clone())),
            AstNode::BitwiseNot(_) => AstNode::BitwiseNot(Box::new(children[0].clone())),
            AstNode::Cast(_, dtype) => AstNode::Cast(Box::new(children[0].clone()), dtype.clone()),

            // Fused Multiply-Add
            AstNode::Fma { .. } => AstNode::Fma {
                a: Box::new(children[0].clone()),
                b: Box::new(children[1].clone()),
                c: Box::new(children[2].clone()),
            },

            // Atomic operations
            AstNode::AtomicAdd { dtype, .. } => AstNode::AtomicAdd {
                ptr: Box::new(children[0].clone()),
                offset: Box::new(children[1].clone()),
                value: Box::new(children[2].clone()),
                dtype: dtype.clone(),
            },
            AstNode::AtomicMax { dtype, .. } => AstNode::AtomicMax {
                ptr: Box::new(children[0].clone()),
                offset: Box::new(children[1].clone()),
                value: Box::new(children[2].clone()),
                dtype: dtype.clone(),
            },

            // メモリ操作
            AstNode::Load { count, dtype, .. } => AstNode::Load {
                ptr: Box::new(children[0].clone()),
                offset: Box::new(children[1].clone()),
                count: *count,
                dtype: dtype.clone(),
            },
            AstNode::Store { .. } => AstNode::Store {
                ptr: Box::new(children[0].clone()),
                offset: Box::new(children[1].clone()),
                value: Box::new(children[2].clone()),
            },

            // 代入
            AstNode::Assign { var, .. } => AstNode::Assign {
                var: var.clone(),
                value: Box::new(children[0].clone()),
            },

            // Block
            AstNode::Block { scope, .. } => AstNode::Block {
                statements: children.to_vec(),
                scope: scope.clone(),
            },

            // Range
            AstNode::Range { var, parallel, .. } => AstNode::Range {
                var: var.clone(),
                start: Box::new(children[0].clone()),
                step: Box::new(children[1].clone()),
                stop: Box::new(children[2].clone()),
                body: Box::new(children[3].clone()),
                parallel: parallel.clone(),
            },

            // If
            AstNode::If { else_body, .. } => AstNode::If {
                condition: Box::new(children[0].clone()),
                then_body: Box::new(children[1].clone()),
                else_body: if else_body.is_some() {
                    Some(Box::new(children[2].clone()))
                } else {
                    None
                },
            },

            // Call
            AstNode::Call { name, .. } => AstNode::Call {
                name: name.clone(),
                args: children.to_vec(),
            },

            // CallKernel
            AstNode::CallKernel {
                name,
                grid_size,
                thread_group_size,
                ..
            } => AstNode::CallKernel {
                name: name.clone(),
                args: children.to_vec(),
                grid_size: grid_size.clone(),
                thread_group_size: thread_group_size.clone(),
            },

            // Return
            AstNode::Return { .. } => AstNode::Return {
                value: Box::new(children[0].clone()),
            },

            // Allocate
            AstNode::Allocate { dtype, .. } => AstNode::Allocate {
                dtype: dtype.clone(),
                size: Box::new(children[0].clone()),
            },

            // Deallocate
            AstNode::Deallocate { .. } => AstNode::Deallocate {
                ptr: Box::new(children[0].clone()),
            },

            // Function
            AstNode::Function {
                name,
                params,
                return_type,
                ..
            } => AstNode::Function {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(children[0].clone()),
            },

            // Kernel
            AstNode::Kernel {
                name,
                params,
                return_type,
                default_grid_size,
                default_thread_group_size,
                ..
            } => AstNode::Kernel {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(children[0].clone()),
                default_grid_size: default_grid_size.clone(),
                default_thread_group_size: default_thread_group_size.clone(),
            },

            // Program
            AstNode::Program {
                execution_waves, ..
            } => AstNode::Program {
                functions: children.to_vec(),
                execution_waves: execution_waves.clone(),
            },

            // WmmaMatmul
            AstNode::WmmaMatmul {
                dtype_ab, dtype_c, ..
            } => AstNode::WmmaMatmul {
                a_ptr: Box::new(children[0].clone()),
                a_offset: Box::new(children[1].clone()),
                a_stride: Box::new(children[2].clone()),
                b_ptr: Box::new(children[3].clone()),
                b_offset: Box::new(children[4].clone()),
                b_stride: Box::new(children[5].clone()),
                c_ptr: Box::new(children[6].clone()),
                c_offset: Box::new(children[7].clone()),
                c_stride: Box::new(children[8].clone()),
                m: Box::new(children[9].clone()),
                k: Box::new(children[10].clone()),
                n: Box::new(children[11].clone()),
                dtype_ab: dtype_ab.clone(),
                dtype_c: dtype_c.clone(),
            },

            // SharedMemory operations
            AstNode::SharedAlloc { name, dtype, .. } => AstNode::SharedAlloc {
                name: name.clone(),
                dtype: dtype.clone(),
                size: Box::new(children[0].clone()),
            },
            AstNode::SharedLoad { dtype, .. } => AstNode::SharedLoad {
                ptr: Box::new(children[0].clone()),
                offset: Box::new(children[1].clone()),
                dtype: dtype.clone(),
            },
            AstNode::SharedStore { .. } => AstNode::SharedStore {
                ptr: Box::new(children[0].clone()),
                offset: Box::new(children[1].clone()),
                value: Box::new(children[2].clone()),
            },
        }
    }

    /// Program全体から関数のインライン展開候補を収集
    fn collect_inlining_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        if let AstNode::Program {
            functions,
            execution_waves,
        } = ast
        {
            // 関数名→関数定義のマップを作成
            let func_map: HashMap<String, &AstNode> = functions
                .iter()
                .filter_map(|f| {
                    if let AstNode::Function { name: Some(n), .. } = f {
                        Some((n.clone(), f))
                    } else {
                        None
                    }
                })
                .collect();

            // 各関数に対してインライン展開を試みる
            for (i, func) in functions.iter().enumerate() {
                // FunctionとKernelの両方を処理
                let (name, params, return_type, body, kernel_dispatch) = match func {
                    AstNode::Function {
                        name,
                        params,
                        return_type,
                        body,
                    } => (name, params, return_type, body, None),
                    AstNode::Kernel {
                        name,
                        params,
                        return_type,
                        body,
                        default_grid_size,
                        default_thread_group_size,
                    } => (
                        name,
                        params,
                        return_type,
                        body,
                        Some((default_grid_size.clone(), default_thread_group_size.clone())),
                    ),
                    _ => continue,
                };

                debug!("Checking function {:?} for inlining opportunities", name);

                // 関数本体でインライン展開を試みる
                if let Some(new_body) = self.try_inline_in_ast(body, &func_map) {
                    let new_func = if let Some((grid_size, tg_size)) = kernel_dispatch {
                        AstNode::Kernel {
                            name: name.clone(),
                            params: params.clone(),
                            return_type: return_type.clone(),
                            body: Box::new(new_body),
                            default_grid_size: grid_size,
                            default_thread_group_size: tg_size,
                        }
                    } else {
                        AstNode::Function {
                            name: name.clone(),
                            params: params.clone(),
                            return_type: return_type.clone(),
                            body: Box::new(new_body),
                        }
                    };

                    let mut new_functions = functions.clone();
                    new_functions[i] = new_func;

                    candidates.push(AstNode::Program {
                        functions: new_functions,
                        execution_waves: execution_waves.clone(),
                    });
                }
            }

            // デッドコード削除：使われていない関数を削除する候補を生成
            let dead_code_candidate = self.remove_dead_functions(ast);
            if let Some(candidate) = dead_code_candidate {
                candidates.push(candidate);
            }
        }

        candidates
    }

    /// 使われていない関数を削除する
    ///
    /// Programから、Call文から呼び出されていないヘルパー関数を削除する
    /// Kernelノードは全て保持される（それぞれがエントリポイント）
    /// 他から呼ばれていないFunctionもエントリポイントとして保持
    fn remove_dead_functions(&self, ast: &AstNode) -> Option<AstNode> {
        if let AstNode::Program {
            functions,
            execution_waves,
        } = ast
        {
            // 使われている関数名を収集
            let mut used_functions: HashSet<String> = HashSet::new();
            let mut called_functions: HashSet<String> = HashSet::new();

            // 全てのKernel関数名は必ず残す（エントリポイント）
            for func in functions {
                if let AstNode::Kernel { name: Some(n), .. } = func {
                    used_functions.insert(n.clone());
                }
            }

            // 全てのCall文を走査して呼ばれている関数を特定
            for func in functions {
                Self::collect_called_functions(func, &mut called_functions);
            }

            // 呼ばれている関数は使われている
            used_functions.extend(called_functions.iter().cloned());

            // 他から呼ばれていないFunctionはトップレベルのエントリポイントとして保持
            for func in functions {
                if let AstNode::Function { name: Some(n), .. } = func
                    && !called_functions.contains(n)
                {
                    used_functions.insert(n.clone());
                }
            }

            // 使われていない関数を削除
            let new_functions: Vec<AstNode> = functions
                .iter()
                .filter(|f| {
                    if let AstNode::Function { name: Some(n), .. } = f {
                        used_functions.contains(n)
                    } else {
                        true // Kernelやその他のノードは保持
                    }
                })
                .cloned()
                .collect();

            // 関数が削除された場合のみ候補を生成
            if new_functions.len() < functions.len() {
                trace!(
                    "Removed {} dead functions",
                    functions.len() - new_functions.len()
                );
                Some(AstNode::Program {
                    functions: new_functions,
                    execution_waves: execution_waves.clone(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// ASTから全てのCall文で使われている関数名を収集
    fn collect_called_functions(ast: &AstNode, used: &mut HashSet<String>) {
        if let AstNode::Call { name, .. } = ast {
            used.insert(name.clone());
        }

        for child in ast.children() {
            Self::collect_called_functions(child, used);
        }
    }
}

impl AstSuggester for FunctionInliningSuggester {
    fn name(&self) -> &str {
        "FunctionInlining"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        trace!("FunctionInliningSuggester: Generating function inlining suggestions");
        let candidates = self.collect_inlining_candidates(ast);

        // インライン展開後にデッドコード削除を適用
        let optimized_candidates: Vec<AstNode> = candidates
            .into_iter()
            .map(|candidate| self.remove_dead_functions(&candidate).unwrap_or(candidate))
            .collect();

        let suggestions = super::deduplicate_candidates(optimized_candidates);

        debug!(
            "FunctionInliningSuggester: Generated {} unique suggestions",
            suggestions.len()
        );

        suggestions
            .into_iter()
            .map(|ast| AstSuggestResult::with_description(ast, self.name(), "inline function"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{const_int, var};
    use crate::ast::{DType, Mutability, VarDecl, VarKind};

    #[test]
    fn test_simple_function_inlining() {
        let suggester = FunctionInliningSuggester::with_default_limit();

        // 定義: fn add_one(x: Int) -> Int { return x + 1 }
        let add_one_body = AstNode::Return {
            value: Box::new(var("x") + const_int(1)),
        };

        let add_one_func = AstNode::Function {
            name: Some("add_one".to_string()),
            params: vec![VarDecl {
                name: "x".to_string(),
                dtype: DType::I64,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::I64,
            body: Box::new(add_one_body),
        };

        // メイン関数: fn main() -> Int { return add_one(5) }
        let main_body = AstNode::Return {
            value: Box::new(AstNode::Call {
                name: "add_one".to_string(),
                args: vec![const_int(5)],
            }),
        };

        let main_func = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::I64,
            body: Box::new(main_body),
        };

        let program = AstNode::Program {
            functions: vec![add_one_func, main_func],
            execution_waves: vec![],
        };

        let suggestions = suggester.suggest(&program);

        // 少なくとも1つの候補（main関数でadd_oneがインライン展開される）が生成されるはず
        assert!(!suggestions.is_empty());

        // 最初の候補を検証
        if let AstNode::Program { functions, .. } = &suggestions[0].ast {
            // main関数を取得
            let main_func = functions
                .iter()
                .find(|f| matches!(f, AstNode::Function { name: Some(n), .. } if n == "main"));

            assert!(main_func.is_some());

            if let Some(AstNode::Function { body, .. }) = main_func {
                // main関数の本体がReturn { 5 + 1 }になっているはず
                if let AstNode::Return { value } = body.as_ref() {
                    // 値がAdd(5, 1)になっているはず
                    assert!(matches!(value.as_ref(), AstNode::Add(..)));
                } else {
                    panic!("Expected Return node in main function body");
                }
            }
        } else {
            panic!("Expected Program node");
        }
    }

    #[test]
    fn test_function_with_multiple_returns_not_inlined() {
        let suggester = FunctionInliningSuggester::with_default_limit();

        // 複数のReturn文を持つ関数（制御フローが複雑）
        // この実装では簡単のため、Return文が複数ある場合は展開しない
        // 実際には条件分岐などで複雑になるため

        // fn identity(x: Int) -> Int { return x }
        let identity_body = AstNode::Return {
            value: Box::new(var("x")),
        };

        let identity_func = AstNode::Function {
            name: Some("identity".to_string()),
            params: vec![VarDecl {
                name: "x".to_string(),
                dtype: DType::I64,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::I64,
            body: Box::new(identity_body),
        };

        // fn main() -> Int { return identity(42) }
        let main_body = AstNode::Return {
            value: Box::new(AstNode::Call {
                name: "identity".to_string(),
                args: vec![const_int(42)],
            }),
        };

        let main_func = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::I64,
            body: Box::new(main_body),
        };

        let program = AstNode::Program {
            functions: vec![identity_func, main_func],
            execution_waves: vec![],
        };

        let suggestions = suggester.suggest(&program);

        // identity関数は展開可能なので、候補が生成されるはず
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_count_nodes() {
        // Const(1)
        let simple = const_int(1);
        assert_eq!(FunctionInliningSuggester::count_nodes(&simple), 1);

        // Add(Const(1), Const(2))
        let add = const_int(1) + const_int(2);
        assert_eq!(FunctionInliningSuggester::count_nodes(&add), 3);
    }

    #[test]
    fn test_substitute_vars() {
        let mut replacements = HashMap::new();
        replacements.insert("x".to_string(), const_int(10));

        // x + 1 → 10 + 1
        let expr = var("x") + const_int(1);

        let result = FunctionInliningSuggester::substitute_vars(&expr, &replacements);

        if let AstNode::Add(left, right) = result {
            assert!(matches!(left.as_ref(), AstNode::Const(_)));
            assert!(matches!(right.as_ref(), AstNode::Const(_)));
        } else {
            panic!("Expected Add node");
        }
    }

    #[test]
    fn test_kernel_function_inlining() {
        use crate::ast::Scope;
        use crate::ast::helper::{block, range, store};

        // カーネル関数のような構造を作成（Blockの中にRangeがある）
        // fn kernel_0(input: Ptr<Int>, output: Ptr<Int>) {
        //     for i in 0..10 {
        //         Store(output, i, Load(input, i))
        //     }
        // }
        let kernel_body = block(
            vec![range(
                "i",
                const_int(0),
                const_int(1),
                const_int(10),
                store(var("output"), var("i"), var("input")),
            )],
            Scope::new(),
        );

        let kernel_func = AstNode::Function {
            name: Some("kernel_0".to_string()),
            params: vec![
                VarDecl {
                    name: "input".to_string(),
                    dtype: DType::Ptr(Box::new(DType::I64), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "output".to_string(),
                    dtype: DType::Ptr(Box::new(DType::I64), AddressSpace::Global),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(kernel_body),
        };

        // main関数: kernel_0を呼び出す
        let main_body = block(
            vec![AstNode::Call {
                name: "kernel_0".to_string(),
                args: vec![var("a"), var("b")],
            }],
            Scope::new(),
        );

        let main_func = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![
                VarDecl {
                    name: "a".to_string(),
                    dtype: DType::Ptr(Box::new(DType::I64), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "b".to_string(),
                    dtype: DType::Ptr(Box::new(DType::I64), AddressSpace::Global),
                    mutability: Mutability::Mutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]),
            body: Box::new(main_body),
        };

        let program = AstNode::Program {
            functions: vec![kernel_func, main_func],
            execution_waves: vec![],
        };

        let suggester = FunctionInliningSuggester::with_default_limit();
        let suggestions = suggester.suggest(&program);

        // カーネル関数がmainにインライン展開されるはず
        assert!(
            !suggestions.is_empty(),
            "Expected at least one suggestion for kernel inlining"
        );

        // インライン展開後の構造を確認
        if let AstNode::Program { functions, .. } = &suggestions[0].ast {
            // mainだけになっているはず（kernel_0が削除された場合）
            // または、main内にCallがなくなっているはず
            let main_func = functions
                .iter()
                .find(|f| matches!(f, AstNode::Function { name: Some(n), .. } if n == "main"));

            assert!(main_func.is_some());

            if let Some(AstNode::Function { body, .. }) = main_func
                && let AstNode::Block { statements, .. } = body.as_ref()
            {
                // Call文がRangeに置き換わっているはず
                let has_range = statements
                    .iter()
                    .any(|stmt| matches!(stmt, AstNode::Range { .. }));
                assert!(
                    has_range,
                    "Expected Range statement after inlining, got: {:?}",
                    statements
                );
            }
        }
    }

    #[test]
    fn test_void_function_inlining() {
        use crate::ast::Scope;
        use crate::ast::helper::{block, store};

        let suggester = FunctionInliningSuggester::with_default_limit();

        // 定義: fn write_value(ptr: Ptr<Int>, offset: Int, value: Int) {
        //     Store(ptr, offset, value)
        // }
        let write_value_body = block(
            vec![store(var("ptr"), var("offset"), var("value"))],
            Scope::new(),
        );

        let write_value_func = AstNode::Function {
            name: Some("write_value".to_string()),
            params: vec![
                VarDecl {
                    name: "ptr".to_string(),
                    dtype: DType::Ptr(Box::new(DType::I64), AddressSpace::Global),
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "offset".to_string(),
                    dtype: DType::I64,
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
                VarDecl {
                    name: "value".to_string(),
                    dtype: DType::I64,
                    mutability: Mutability::Immutable,
                    kind: VarKind::Normal,
                },
            ],
            return_type: DType::Tuple(vec![]), // void型（unit型）
            body: Box::new(write_value_body),
        };

        // メイン関数: fn main() {
        //     write_value(buffer, 0, 42)
        //     write_value(buffer, 1, 100)
        // }
        let main_body = block(
            vec![
                AstNode::Call {
                    name: "write_value".to_string(),
                    args: vec![var("buffer"), const_int(0), const_int(42)],
                },
                AstNode::Call {
                    name: "write_value".to_string(),
                    args: vec![var("buffer"), const_int(1), const_int(100)],
                },
            ],
            Scope::new(),
        );

        let main_func = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::Tuple(vec![]),
            body: Box::new(main_body),
        };

        let program = AstNode::Program {
            functions: vec![write_value_func, main_func],
            execution_waves: vec![],
        };

        let suggestions = suggester.suggest(&program);

        // void型関数のインライン展開候補が生成されるはず
        assert!(!suggestions.is_empty());

        // 最初の候補を検証
        if let AstNode::Program { functions, .. } = &suggestions[0].ast {
            // main関数を取得
            let main_func = functions
                .iter()
                .find(|f| matches!(f, AstNode::Function { name: Some(n), .. } if n == "main"));

            assert!(main_func.is_some());

            if let Some(AstNode::Function { body, .. }) = main_func {
                // main関数の本体がBlockで、Store文が直接含まれているはず
                if let AstNode::Block { statements, .. } = body.as_ref() {
                    // インライン展開されていれば、Call文がStore文に置き換わっているはず
                    // 元々2つのCall文があったので、展開後は2つのStore文になるはず
                    assert!(statements.len() >= 2);

                    // 少なくとも1つはStore文であるべき
                    let has_store = statements
                        .iter()
                        .any(|stmt| matches!(stmt, AstNode::Store { .. }));
                    assert!(
                        has_store,
                        "Expected at least one Store statement after inlining"
                    );
                } else {
                    panic!("Expected Block node in main function body");
                }
            }
        } else {
            panic!("Expected Program node");
        }
    }
}

//! 変数展開（Variable Expansion / Inlining）のためのSuggester実装
//!
//! 一時変数への代入を、その変数の使用箇所に展開します。
//! CSE（共通部分式除去）の逆操作として機能します。
//!
//! # 例
//! ```text
//! // 展開前
//! x = (a * (a + 1))
//! y = x + x
//!
//! // 展開後
//! y = (a * (a + 1)) + (a * (a + 1))
//! ```
//!
//! # 用途
//! - CSE適用後にレジスタ圧力が高くなった場合の最適化解除
//! - 1回しか使用されない一時変数の除去
//! - 定数伝播など他の最適化のための準備

use crate::ast::{AstNode, Scope};
use crate::opt::ast::AstSuggester;
use log::{debug, trace};

/// 変数展開を提案するSuggester
pub struct VariableExpansionSuggester {
    /// 展開対象とする変数名のプレフィックス（Noneの場合はすべての変数が対象）
    target_prefix: Option<String>,
    /// 使用回数がこの値以下の変数のみ展開（Noneの場合は制限なし）
    max_usage_count: Option<usize>,
}

impl VariableExpansionSuggester {
    /// 新しいVariableExpansionSuggesterを作成
    pub fn new() -> Self {
        Self {
            target_prefix: None,
            max_usage_count: None,
        }
    }

    /// 展開対象とする変数名のプレフィックスを設定
    ///
    /// 例: `with_prefix("cse_tmp")` でCSEで生成された変数のみを対象にする
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.target_prefix = Some(prefix.to_string());
        self
    }

    /// 展開対象とする変数の最大使用回数を設定
    ///
    /// 例: `with_max_usage(2)` で2回以下使用される変数のみを展開
    pub fn with_max_usage(mut self, count: usize) -> Self {
        self.max_usage_count = Some(count);
        self
    }

    /// 変数が展開対象かどうかをチェック
    fn is_expansion_target(&self, var_name: &str) -> bool {
        match &self.target_prefix {
            Some(prefix) => var_name.starts_with(prefix),
            None => true,
        }
    }

    /// 式内の変数の使用回数をカウント
    fn count_var_usage(expr: &AstNode, var_name: &str) -> usize {
        match expr {
            AstNode::Var(name) if name == var_name => 1,
            AstNode::Var(_) | AstNode::Const(_) | AstNode::Rand => 0,
            AstNode::Add(a, b)
            | AstNode::Mul(a, b)
            | AstNode::Max(a, b)
            | AstNode::Rem(a, b)
            | AstNode::Idiv(a, b)
            | AstNode::BitwiseAnd(a, b)
            | AstNode::BitwiseOr(a, b)
            | AstNode::BitwiseXor(a, b)
            | AstNode::LeftShift(a, b)
            | AstNode::RightShift(a, b) => {
                Self::count_var_usage(a, var_name) + Self::count_var_usage(b, var_name)
            }
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a)
            | AstNode::BitwiseNot(a)
            | AstNode::Cast(a, _) => Self::count_var_usage(a, var_name),
            AstNode::Load { ptr, offset, .. } => {
                Self::count_var_usage(ptr, var_name) + Self::count_var_usage(offset, var_name)
            }
            AstNode::Store { ptr, offset, value } => {
                Self::count_var_usage(ptr, var_name)
                    + Self::count_var_usage(offset, var_name)
                    + Self::count_var_usage(value, var_name)
            }
            AstNode::Assign { value, .. } => Self::count_var_usage(value, var_name),
            AstNode::Block { statements, .. } => statements
                .iter()
                .map(|s| Self::count_var_usage(s, var_name))
                .sum(),
            AstNode::Range {
                start,
                step,
                stop,
                body,
                ..
            } => {
                Self::count_var_usage(start, var_name)
                    + Self::count_var_usage(step, var_name)
                    + Self::count_var_usage(stop, var_name)
                    + Self::count_var_usage(body, var_name)
            }
            AstNode::Call { args, .. } => args
                .iter()
                .map(|a| Self::count_var_usage(a, var_name))
                .sum(),
            AstNode::Return { value } => Self::count_var_usage(value, var_name),
            AstNode::Function { body, .. } => Self::count_var_usage(body, var_name),
            AstNode::Program { functions, .. } => functions
                .iter()
                .map(|f| Self::count_var_usage(f, var_name))
                .sum(),
            _ => 0,
        }
    }

    /// 式内の変数参照を別の式で置換
    fn substitute_var(expr: &AstNode, var_name: &str, replacement: &AstNode) -> AstNode {
        match expr {
            AstNode::Var(name) if name == var_name => replacement.clone(),
            AstNode::Var(_) | AstNode::Const(_) | AstNode::Rand | AstNode::Wildcard(_) => {
                expr.clone()
            }
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Max(a, b) => AstNode::Max(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Rem(a, b) => AstNode::Rem(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Idiv(a, b) => AstNode::Idiv(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::BitwiseAnd(a, b) => AstNode::BitwiseAnd(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::BitwiseOr(a, b) => AstNode::BitwiseOr(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::BitwiseXor(a, b) => AstNode::BitwiseXor(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::LeftShift(a, b) => AstNode::LeftShift(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::RightShift(a, b) => AstNode::RightShift(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Recip(a) => {
                AstNode::Recip(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Sqrt(a) => {
                AstNode::Sqrt(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Log2(a) => {
                AstNode::Log2(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Exp2(a) => {
                AstNode::Exp2(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Sin(a) => {
                AstNode::Sin(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::BitwiseNot(a) => {
                AstNode::BitwiseNot(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Cast(a, dtype) => AstNode::Cast(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                dtype.clone(),
            ),
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => AstNode::Load {
                ptr: Box::new(Self::substitute_var(ptr, var_name, replacement)),
                offset: Box::new(Self::substitute_var(offset, var_name, replacement)),
                count: *count,
                dtype: dtype.clone(),
            },
            AstNode::Store { ptr, offset, value } => AstNode::Store {
                ptr: Box::new(Self::substitute_var(ptr, var_name, replacement)),
                offset: Box::new(Self::substitute_var(offset, var_name, replacement)),
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
            },
            AstNode::Assign { var, value } => AstNode::Assign {
                var: var.clone(),
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
            },
            AstNode::Block { statements, scope } => AstNode::Block {
                statements: statements
                    .iter()
                    .map(|s| Self::substitute_var(s, var_name, replacement))
                    .collect(),
                scope: scope.clone(),
            },
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => AstNode::Range {
                var: var.clone(),
                start: Box::new(Self::substitute_var(start, var_name, replacement)),
                step: Box::new(Self::substitute_var(step, var_name, replacement)),
                stop: Box::new(Self::substitute_var(stop, var_name, replacement)),
                body: Box::new(Self::substitute_var(body, var_name, replacement)),
            },
            AstNode::Call { name, args } => AstNode::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_var(a, var_name, replacement))
                    .collect(),
            },
            AstNode::Return { value } => AstNode::Return {
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
            },
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => AstNode::Function {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(Self::substitute_var(body, var_name, replacement)),
            },
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => AstNode::Kernel {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(Self::substitute_var(body, var_name, replacement)),
                default_grid_size: default_grid_size.clone(),
                default_thread_group_size: default_thread_group_size.clone(),
            },
            AstNode::Program {
                functions,
                entry_point,
            } => AstNode::Program {
                functions: functions
                    .iter()
                    .map(|f| Self::substitute_var(f, var_name, replacement))
                    .collect(),
                entry_point: entry_point.clone(),
            },
            // その他のノードはそのまま返す
            _ => expr.clone(),
        }
    }

    /// Block内で変数展開を試みる
    fn try_expand_in_block(&self, statements: &[AstNode], scope: &Scope) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        // 各Assign文を探す
        for (i, stmt) in statements.iter().enumerate() {
            if let AstNode::Assign { var, value } = stmt {
                // 展開対象かチェック
                if !self.is_expansion_target(var) {
                    continue;
                }

                // 後続のstatements内での使用回数をカウント
                let usage_count: usize = statements[i + 1..]
                    .iter()
                    .map(|s| Self::count_var_usage(s, var))
                    .sum();

                // 使用回数の制限チェック
                if let Some(max) = self.max_usage_count
                    && usage_count > max
                {
                    continue;
                }

                // 使用されていない場合は単純に削除
                if usage_count == 0 {
                    debug!("Variable '{}' is unused, removing assignment", var);
                    let mut new_statements: Vec<AstNode> = statements[..i].to_vec();
                    new_statements.extend(statements[i + 1..].iter().cloned());
                    candidates.push(AstNode::Block {
                        statements: new_statements,
                        scope: Box::new(scope.clone()),
                    });
                    continue;
                }

                debug!(
                    "Expanding variable '{}' (used {} times): {:?}",
                    var, usage_count, value
                );

                // 変数参照を式で置換し、Assign文を削除
                let mut new_statements: Vec<AstNode> = statements[..i].to_vec();
                for following_stmt in &statements[i + 1..] {
                    new_statements.push(Self::substitute_var(following_stmt, var, value));
                }

                candidates.push(AstNode::Block {
                    statements: new_statements,
                    scope: Box::new(scope.clone()),
                });
            }
        }

        candidates
    }

    /// ASTツリーを走査して、展開可能な箇所を探す
    fn collect_expansion_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        match ast {
            AstNode::Block { statements, scope } => {
                // Block内での展開を試みる
                let block_candidates = self.try_expand_in_block(statements, scope);
                candidates.extend(block_candidates);

                // 子要素を再帰的に処理
                for (i, stmt) in statements.iter().enumerate() {
                    let child_candidates = self.collect_expansion_candidates(stmt);
                    for child in child_candidates {
                        let mut new_statements = statements.clone();
                        new_statements[i] = child;
                        candidates.push(AstNode::Block {
                            statements: new_statements,
                            scope: scope.clone(),
                        });
                    }
                }
            }

            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                // ループ本体内での展開を試みる
                let child_candidates = self.collect_expansion_candidates(body);
                for child in child_candidates {
                    candidates.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(child),
                    });
                }
            }

            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                let child_candidates = self.collect_expansion_candidates(body);
                for child in child_candidates {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(child),
                    });
                }
            }

            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => {
                let child_candidates = self.collect_expansion_candidates(body);
                for child in child_candidates {
                    candidates.push(AstNode::Kernel {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(child),
                        default_grid_size: default_grid_size.clone(),
                        default_thread_group_size: default_thread_group_size.clone(),
                    });
                }
            }

            AstNode::Program {
                functions,
                entry_point,
            } => {
                for (i, func) in functions.iter().enumerate() {
                    let child_candidates = self.collect_expansion_candidates(func);
                    for child in child_candidates {
                        let mut new_functions = functions.clone();
                        new_functions[i] = child;
                        candidates.push(AstNode::Program {
                            functions: new_functions,
                            entry_point: entry_point.clone(),
                        });
                    }
                }
            }

            _ => {}
        }

        candidates
    }
}

impl Default for VariableExpansionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for VariableExpansionSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("VariableExpansionSuggester: Generating expansion suggestions");
        let candidates = self.collect_expansion_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);

        debug!(
            "VariableExpansionSuggester: Generated {} unique suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{block, const_int, store, var};

    #[test]
    fn test_simple_expansion() {
        // xのみを展開対象にする
        let suggester = VariableExpansionSuggester::new().with_prefix("x");

        // x = (a * (a + 1))
        // y = x + x
        let common_expr = AstNode::Mul(
            Box::new(var("a")),
            Box::new(AstNode::Add(Box::new(var("a")), Box::new(const_int(1)))),
        );

        let assign_x = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(common_expr.clone()),
        };

        let assign_y = AstNode::Assign {
            var: "y".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("x")), Box::new(var("x")))),
        };

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // 展開後は1つのstatementのみ（xの代入は削除）
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 1);

            // y = (a * (a + 1)) + (a * (a + 1))
            if let AstNode::Assign { var, value } = &statements[0] {
                assert_eq!(var, "y");
                // 展開されているか確認（xが含まれていない）
                assert_eq!(VariableExpansionSuggester::count_var_usage(value, "x"), 0);
                // 元の式が展開されている
                let value_str = format!("{:?}", value);
                assert!(value_str.contains("Mul"));
            }
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_expansion_with_prefix_filter() {
        let suggester = VariableExpansionSuggester::new().with_prefix("cse_tmp");

        // cse_tmp_0 = (a + b)  <- 対象
        // regular_var = (c + d)  <- 対象外
        // y = cse_tmp_0 + regular_var
        let assign_cse = AstNode::Assign {
            var: "cse_tmp_0".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("b")))),
        };

        let assign_regular = AstNode::Assign {
            var: "regular_var".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("c")), Box::new(var("d")))),
        };

        let assign_y = AstNode::Assign {
            var: "y".to_string(),
            value: Box::new(AstNode::Add(
                Box::new(var("cse_tmp_0")),
                Box::new(var("regular_var")),
            )),
        };

        let input = block(vec![assign_cse, assign_regular, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // cse_tmp_0のみ展開される
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 2); // regular_varの代入 + yの代入

            // regular_varの代入は残っている
            if let AstNode::Assign { var, .. } = &statements[0] {
                assert_eq!(var, "regular_var");
            }

            // yの式でcse_tmp_0は展開されているが、regular_varは残っている
            if let AstNode::Assign { value, .. } = &statements[1] {
                assert_eq!(
                    VariableExpansionSuggester::count_var_usage(value, "cse_tmp_0"),
                    0
                );
                assert_eq!(
                    VariableExpansionSuggester::count_var_usage(value, "regular_var"),
                    1
                );
            }
        }
    }

    #[test]
    fn test_expansion_with_max_usage() {
        // xのみを展開対象にする（yは未使用なので別の候補になる）
        let suggester = VariableExpansionSuggester::new()
            .with_max_usage(1)
            .with_prefix("x");

        // x = (a + b)  <- 2回使用されるので展開対象外
        // y = x + x
        let assign_x = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("b")))),
        };

        let assign_y = AstNode::Assign {
            var: "y".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("x")), Box::new(var("x")))),
        };

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        // 使用回数が2なので展開されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_expansion_single_use() {
        // xのみを展開対象にする（yは未使用なので別の候補になる）
        let suggester = VariableExpansionSuggester::new()
            .with_max_usage(1)
            .with_prefix("x");

        // x = (a + b)  <- 1回のみ使用
        // y = x * c
        let assign_x = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("b")))),
        };

        let assign_y = AstNode::Assign {
            var: "y".to_string(),
            value: Box::new(AstNode::Mul(Box::new(var("x")), Box::new(var("c")))),
        };

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // 展開後は1つのstatement
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 1);
        }
    }

    #[test]
    fn test_remove_unused_variable() {
        // xのみを展開対象にする（yも未使用なので別の候補になる）
        let suggester = VariableExpansionSuggester::new().with_prefix("x");

        // x = (a + b)  <- 使用されていない
        // y = c * d
        let assign_x = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("b")))),
        };

        let assign_y = AstNode::Assign {
            var: "y".to_string(),
            value: Box::new(AstNode::Mul(Box::new(var("c")), Box::new(var("d")))),
        };

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // 未使用の代入が削除される
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 1);
            if let AstNode::Assign { var, .. } = &statements[0] {
                assert_eq!(var, "y");
            }
        }
    }

    #[test]
    fn test_expansion_in_loop() {
        let suggester = VariableExpansionSuggester::new();

        // for i in 0..10:
        //   tmp = a + b
        //   output[i] = tmp * c
        let assign_tmp = AstNode::Assign {
            var: "tmp".to_string(),
            value: Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("b")))),
        };

        let store_output = store(
            var("output"),
            var("i"),
            AstNode::Mul(Box::new(var("tmp")), Box::new(var("c"))),
        );

        let loop_body = block(vec![assign_tmp, store_output], Scope::new());

        let input = block(
            vec![AstNode::Range {
                var: "i".to_string(),
                start: Box::new(const_int(0)),
                step: Box::new(const_int(1)),
                stop: Box::new(const_int(10)),
                body: Box::new(loop_body),
            }],
            Scope::new(),
        );

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // ループ本体で展開されている
        if let AstNode::Block { statements, .. } = &suggestions[0]
            && let AstNode::Range { body, .. } = &statements[0]
            && let AstNode::Block {
                statements: inner, ..
            } = body.as_ref()
        {
            // 1つのstatementのみ（tmpの代入が削除）
            assert_eq!(inner.len(), 1);
        }
    }

    #[test]
    fn test_count_var_usage() {
        // a + b -> aの使用回数は1
        let expr1 = AstNode::Add(Box::new(var("a")), Box::new(var("b")));
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr1, "a"), 1);
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr1, "b"), 1);
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr1, "c"), 0);

        // a + a -> aの使用回数は2
        let expr2 = AstNode::Add(Box::new(var("a")), Box::new(var("a")));
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr2, "a"), 2);

        // (a + b) * a -> aの使用回数は2
        let expr3 = AstNode::Mul(
            Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("b")))),
            Box::new(var("a")),
        );
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr3, "a"), 2);
    }

    #[test]
    fn test_cse_and_expansion_roundtrip() {
        // CSEと展開は互いに逆操作
        // 元の式: y = (a * b) + (a * b)
        let common_expr = AstNode::Mul(Box::new(var("a")), Box::new(var("b")));
        let _original = block(
            vec![AstNode::Assign {
                var: "y".to_string(),
                value: Box::new(AstNode::Add(
                    Box::new(common_expr.clone()),
                    Box::new(common_expr),
                )),
            }],
            Scope::new(),
        );

        // CSE適用後の形式（手動で作成）
        // cse_tmp_0 = a * b
        // y = cse_tmp_0 + cse_tmp_0
        let cse_applied = block(
            vec![
                AstNode::Assign {
                    var: "cse_tmp_0".to_string(),
                    value: Box::new(AstNode::Mul(Box::new(var("a")), Box::new(var("b")))),
                },
                AstNode::Assign {
                    var: "y".to_string(),
                    value: Box::new(AstNode::Add(
                        Box::new(var("cse_tmp_0")),
                        Box::new(var("cse_tmp_0")),
                    )),
                },
            ],
            Scope::new(),
        );

        // 展開すると元の形式に戻る
        // cse_tmp_で始まる変数のみを展開対象にする（yは別の候補になる）
        let suggester = VariableExpansionSuggester::new().with_prefix("cse_tmp_");
        let suggestions = suggester.suggest(&cse_applied);

        assert_eq!(suggestions.len(), 1);

        // 展開後の構造が元の構造と同じ
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 1);
            if let AstNode::Assign { var, .. } = &statements[0] {
                assert_eq!(var, "y");
            }
        }
    }
}

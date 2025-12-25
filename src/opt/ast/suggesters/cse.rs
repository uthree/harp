//! 共通部分式除去（Common Subexpression Elimination）のためのSuggester実装
//!
//! 同一の部分式が複数回出現する場合、その式を一時変数に抽出して
//! 計算量を削減します。
//!
//! # 例
//! ```text
//! // 最適化前
//! y = (a * (a + 1)) + (a * (a + 1))
//!
//! // 最適化後
//! cse_tmp0 = (a * (a + 1))
//! y = cse_tmp0 + cse_tmp0
//! ```

use crate::ast::{AstNode, DType, Mutability, Scope};
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};
use std::collections::HashMap;

/// 共通部分式除去を提案するSuggester
pub struct CseSuggester {
    /// CSE適用の最小コスト閾値（これより小さい式はCSE対象外）
    min_expr_cost: usize,
    /// 生成する一時変数のプレフィックス
    temp_var_prefix: String,
}

impl CseSuggester {
    /// 新しいCseSuggesterを作成
    pub fn new() -> Self {
        Self {
            min_expr_cost: 2, // 加算1つ程度以上の複雑さが必要
            temp_var_prefix: "cse_tmp".to_string(),
        }
    }

    /// 最小コスト閾値を設定
    pub fn with_min_cost(mut self, cost: usize) -> Self {
        self.min_expr_cost = cost;
        self
    }

    /// 一時変数のプレフィックスを設定
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.temp_var_prefix = prefix.to_string();
        self
    }

    /// 式のコスト（複雑さ）を計算
    /// 変数や定数は0、演算子ごとに1を加算
    fn expr_cost(expr: &AstNode) -> usize {
        match expr {
            AstNode::Const(_) | AstNode::Var(_) | AstNode::Rand => 0,
            AstNode::Add(a, b)
            | AstNode::Mul(a, b)
            | AstNode::Max(a, b)
            | AstNode::Rem(a, b)
            | AstNode::Idiv(a, b)
            | AstNode::BitwiseAnd(a, b)
            | AstNode::BitwiseOr(a, b)
            | AstNode::BitwiseXor(a, b)
            | AstNode::LeftShift(a, b)
            | AstNode::RightShift(a, b) => 1 + Self::expr_cost(a) + Self::expr_cost(b),
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a)
            | AstNode::BitwiseNot(a)
            | AstNode::Cast(a, _) => 1 + Self::expr_cost(a),
            AstNode::Load { ptr, offset, .. } => 1 + Self::expr_cost(ptr) + Self::expr_cost(offset),
            // その他のノード（Store, Block, Range等）はCSE対象外
            _ => 0,
        }
    }

    /// 式がCSE対象として適切かチェック
    /// - 副作用がない純粋な式であること
    /// - 一定以上の複雑さを持つこと
    fn is_cse_candidate(&self, expr: &AstNode) -> bool {
        // Loadは副作用がないが、メモリ読み込みなので注意が必要
        // ここでは純粋な算術式のみをCSE対象とする
        let is_pure_expr = matches!(
            expr,
            AstNode::Add(_, _)
                | AstNode::Mul(_, _)
                | AstNode::Max(_, _)
                | AstNode::Rem(_, _)
                | AstNode::Idiv(_, _)
                | AstNode::Recip(_)
                | AstNode::Sqrt(_)
                | AstNode::Log2(_)
                | AstNode::Exp2(_)
                | AstNode::Sin(_)
                | AstNode::Cast(_, _)
                | AstNode::BitwiseAnd(_, _)
                | AstNode::BitwiseOr(_, _)
                | AstNode::BitwiseXor(_, _)
                | AstNode::BitwiseNot(_)
                | AstNode::LeftShift(_, _)
                | AstNode::RightShift(_, _)
        );

        is_pure_expr && Self::expr_cost(expr) >= self.min_expr_cost
    }

    /// AST内のすべての部分式をカウント
    fn count_subexpressions(&self, expr: &AstNode, counts: &mut HashMap<String, (AstNode, usize)>) {
        // 現在の式をカウント
        if self.is_cse_candidate(expr) {
            let key = format!("{:?}", expr);
            counts
                .entry(key)
                .and_modify(|(_, count)| *count += 1)
                .or_insert((expr.clone(), 1));
        }

        // 子ノードを再帰的に処理
        match expr {
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
                self.count_subexpressions(a, counts);
                self.count_subexpressions(b, counts);
            }
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a)
            | AstNode::BitwiseNot(a)
            | AstNode::Cast(a, _) => {
                self.count_subexpressions(a, counts);
            }
            AstNode::Load { ptr, offset, .. } => {
                self.count_subexpressions(ptr, counts);
                self.count_subexpressions(offset, counts);
            }
            AstNode::Store { ptr, offset, value } => {
                self.count_subexpressions(ptr, counts);
                self.count_subexpressions(offset, counts);
                self.count_subexpressions(value, counts);
            }
            AstNode::Assign { value, .. } => {
                self.count_subexpressions(value, counts);
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    self.count_subexpressions(stmt, counts);
                }
            }
            AstNode::Range { body, .. } => {
                self.count_subexpressions(body, counts);
            }
            AstNode::Call { args, .. } => {
                for arg in args {
                    self.count_subexpressions(arg, counts);
                }
            }
            AstNode::Return { value } => {
                self.count_subexpressions(value, counts);
            }
            AstNode::Function { body, .. } => {
                self.count_subexpressions(body, counts);
            }
            AstNode::Program { functions, .. } => {
                for func in functions {
                    self.count_subexpressions(func, counts);
                }
            }
            _ => {}
        }
    }

    /// 式内の特定の部分式を変数参照に置換
    fn substitute_expr(expr: &AstNode, target: &AstNode, var_name: &str) -> AstNode {
        // 対象の式と一致したら変数参照に置換
        if expr == target {
            return AstNode::Var(var_name.to_string());
        }

        // 子ノードを再帰的に処理
        match expr {
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::Max(a, b) => AstNode::Max(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::Rem(a, b) => AstNode::Rem(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::Idiv(a, b) => AstNode::Idiv(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::BitwiseAnd(a, b) => AstNode::BitwiseAnd(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::BitwiseOr(a, b) => AstNode::BitwiseOr(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::BitwiseXor(a, b) => AstNode::BitwiseXor(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::LeftShift(a, b) => AstNode::LeftShift(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::RightShift(a, b) => AstNode::RightShift(
                Box::new(Self::substitute_expr(a, target, var_name)),
                Box::new(Self::substitute_expr(b, target, var_name)),
            ),
            AstNode::Recip(a) => {
                AstNode::Recip(Box::new(Self::substitute_expr(a, target, var_name)))
            }
            AstNode::Sqrt(a) => AstNode::Sqrt(Box::new(Self::substitute_expr(a, target, var_name))),
            AstNode::Log2(a) => AstNode::Log2(Box::new(Self::substitute_expr(a, target, var_name))),
            AstNode::Exp2(a) => AstNode::Exp2(Box::new(Self::substitute_expr(a, target, var_name))),
            AstNode::Sin(a) => AstNode::Sin(Box::new(Self::substitute_expr(a, target, var_name))),
            AstNode::BitwiseNot(a) => {
                AstNode::BitwiseNot(Box::new(Self::substitute_expr(a, target, var_name)))
            }
            AstNode::Cast(a, dtype) => AstNode::Cast(
                Box::new(Self::substitute_expr(a, target, var_name)),
                dtype.clone(),
            ),
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => AstNode::Load {
                ptr: Box::new(Self::substitute_expr(ptr, target, var_name)),
                offset: Box::new(Self::substitute_expr(offset, target, var_name)),
                count: *count,
                dtype: dtype.clone(),
            },
            AstNode::Store { ptr, offset, value } => AstNode::Store {
                ptr: Box::new(Self::substitute_expr(ptr, target, var_name)),
                offset: Box::new(Self::substitute_expr(offset, target, var_name)),
                value: Box::new(Self::substitute_expr(value, target, var_name)),
            },
            AstNode::Assign { var, value } => AstNode::Assign {
                var: var.clone(),
                value: Box::new(Self::substitute_expr(value, target, var_name)),
            },
            AstNode::Block { statements, scope } => AstNode::Block {
                statements: statements
                    .iter()
                    .map(|s| Self::substitute_expr(s, target, var_name))
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
                start: Box::new(Self::substitute_expr(start, target, var_name)),
                step: Box::new(Self::substitute_expr(step, target, var_name)),
                stop: Box::new(Self::substitute_expr(stop, target, var_name)),
                body: Box::new(Self::substitute_expr(body, target, var_name)),
            },
            AstNode::Call { name, args } => AstNode::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_expr(a, target, var_name))
                    .collect(),
            },
            AstNode::Return { value } => AstNode::Return {
                value: Box::new(Self::substitute_expr(value, target, var_name)),
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
                body: Box::new(Self::substitute_expr(body, target, var_name)),
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
                body: Box::new(Self::substitute_expr(body, target, var_name)),
                default_grid_size: default_grid_size.clone(),
                default_thread_group_size: default_thread_group_size.clone(),
            },
            AstNode::Program {
                functions,
                execution_waves,
            } => AstNode::Program {
                functions: functions
                    .iter()
                    .map(|f| Self::substitute_expr(f, target, var_name))
                    .collect(),
                execution_waves: execution_waves.clone(),
            },
            // その他のノードはそのまま返す
            _ => expr.clone(),
        }
    }

    /// Block内で共通部分式を抽出する
    fn try_cse_in_block(
        &self,
        statements: &[AstNode],
        scope: &Scope,
        temp_counter: &mut usize,
    ) -> Option<(Vec<AstNode>, Scope)> {
        // すべてのstatementsから部分式をカウント
        let mut counts: HashMap<String, (AstNode, usize)> = HashMap::new();
        for stmt in statements {
            self.count_subexpressions(stmt, &mut counts);
        }

        // 2回以上出現する最も複雑な部分式を見つける
        let mut best_candidate: Option<(AstNode, usize)> = None;
        for (_key, (expr, count)) in counts.iter() {
            if *count >= 2 {
                let cost = Self::expr_cost(expr);
                if best_candidate
                    .as_ref()
                    .map(|(_, best_cost)| cost > *best_cost)
                    .unwrap_or(true)
                {
                    best_candidate = Some((expr.clone(), cost));
                }
            }
        }

        // 候補がなければNone
        let (target_expr, _) = best_candidate?;

        debug!(
            "Found common subexpression to extract: cost={}, expr={:?}",
            Self::expr_cost(&target_expr),
            target_expr
        );

        // 一時変数名を生成
        let var_name = format!("{}_{}", self.temp_var_prefix, *temp_counter);
        *temp_counter += 1;

        // 新しいスコープを作成して一時変数を宣言
        let mut new_scope = scope.clone();
        // 型はF32と仮定（実際には型推論が必要だが、簡略化のため）
        let _ = new_scope.declare(var_name.clone(), DType::F32, Mutability::Immutable);

        // 一時変数への代入文を作成
        let assign_stmt = AstNode::Assign {
            var: var_name.clone(),
            value: Box::new(target_expr.clone()),
        };

        // すべてのstatementsで共通部分式を置換
        let mut new_statements = vec![assign_stmt];
        for stmt in statements {
            new_statements.push(Self::substitute_expr(stmt, &target_expr, &var_name));
        }

        trace!(
            "CSE: Extracted '{}' = {:?}, reduced {} statements to {}",
            var_name,
            target_expr,
            statements.len(),
            new_statements.len()
        );

        Some((new_statements, new_scope))
    }

    /// ASTツリーを走査して、CSE可能な箇所を探す
    fn try_cse_in_ast(&self, ast: &AstNode, temp_counter: &mut usize) -> Option<AstNode> {
        match ast {
            AstNode::Block { statements, scope } => {
                // まずBlock内でCSEを試みる
                if let Some((new_statements, new_scope)) =
                    self.try_cse_in_block(statements, scope, temp_counter)
                {
                    return Some(AstNode::Block {
                        statements: new_statements,
                        scope: Box::new(new_scope),
                    });
                }

                // 子要素を再帰的に処理
                let mut new_statements = Vec::new();
                let mut changed = false;

                for stmt in statements {
                    if let Some(new_stmt) = self.try_cse_in_ast(stmt, temp_counter) {
                        new_statements.push(new_stmt);
                        changed = true;
                    } else {
                        new_statements.push(stmt.clone());
                    }
                }

                if changed {
                    Some(AstNode::Block {
                        statements: new_statements,
                        scope: scope.clone(),
                    })
                } else {
                    None
                }
            }

            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                // ループ本体内でCSEを試みる
                self.try_cse_in_ast(body, temp_counter)
                    .map(|new_body| AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(new_body),
                    })
            }

            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => self
                .try_cse_in_ast(body, temp_counter)
                .map(|new_body| AstNode::Function {
                    name: name.clone(),
                    params: params.clone(),
                    return_type: return_type.clone(),
                    body: Box::new(new_body),
                }),
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => self
                .try_cse_in_ast(body, temp_counter)
                .map(|new_body| AstNode::Kernel {
                    name: name.clone(),
                    params: params.clone(),
                    return_type: return_type.clone(),
                    body: Box::new(new_body),
                    default_grid_size: default_grid_size.clone(),
                    default_thread_group_size: default_thread_group_size.clone(),
                }),

            AstNode::Program {
                functions,
                execution_waves,
            } => {
                let mut new_functions = Vec::new();
                let mut changed = false;

                for func in functions {
                    if let Some(new_func) = self.try_cse_in_ast(func, temp_counter) {
                        new_functions.push(new_func);
                        changed = true;
                    } else {
                        new_functions.push(func.clone());
                    }
                }

                if changed {
                    Some(AstNode::Program {
                        functions: new_functions,
                        execution_waves: execution_waves.clone(),
                    })
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Program全体からCSE候補を収集
    fn collect_cse_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();
        let mut temp_counter = 0;

        if let Some(optimized) = self.try_cse_in_ast(ast, &mut temp_counter) {
            candidates.push(optimized);
        }

        candidates
    }
}

impl Default for CseSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for CseSuggester {
    fn name(&self) -> &str {
        "CSE"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        trace!("CseSuggester: Generating CSE suggestions");
        let candidates = self.collect_cse_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);

        debug!(
            "CseSuggester: Generated {} unique suggestions",
            suggestions.len()
        );

        suggestions
            .into_iter()
            .map(|ast| {
                AstSuggestResult::with_description(ast, self.name(), "extract common subexpression")
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{block, const_int, store, var};

    #[test]
    fn test_simple_cse() {
        let suggester = CseSuggester::new();

        // y = (a * (a + 1)) + (a * (a + 1))
        let common_expr = AstNode::Mul(
            Box::new(var("a")),
            Box::new(AstNode::Add(Box::new(var("a")), Box::new(const_int(1)))),
        );
        let expr = AstNode::Add(Box::new(common_expr.clone()), Box::new(common_expr));

        let input = block(
            vec![AstNode::Assign {
                var: "y".to_string(),
                value: Box::new(expr),
            }],
            Scope::new(),
        );

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // CSE後は2つのstatementがあるはず（一時変数の代入 + 元の式）
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 2);

            // 最初のstatementは一時変数への代入
            if let AstNode::Assign { var: v, .. } = &statements[0] {
                assert!(v.starts_with("cse_tmp"));
            } else {
                panic!("Expected Assign for temp variable");
            }

            // 2番目のstatementで一時変数が使用されている
            if let AstNode::Assign { value, .. } = &statements[1] {
                // value内に一時変数への参照がある
                let value_str = format!("{:?}", value);
                assert!(
                    value_str.contains("cse_tmp"),
                    "Expected cse_tmp in value: {}",
                    value_str
                );
            }
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_no_cse_for_simple_expr() {
        let suggester = CseSuggester::new();

        // y = a + a は単純すぎてCSE対象外
        let input = block(
            vec![AstNode::Assign {
                var: "y".to_string(),
                value: Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("a")))),
            }],
            Scope::new(),
        );

        let suggestions = suggester.suggest(&input);

        // 単純な変数の重複はCSE対象外
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_cse_across_statements() {
        let suggester = CseSuggester::new();

        // 複数のstatementにまたがる共通部分式
        // x = ((a + b) * c) + e
        // y = ((a + b) * c) + f
        // 共通部分式: (a + b) * c (コスト2)
        let common_expr = AstNode::Mul(
            Box::new(AstNode::Add(Box::new(var("a")), Box::new(var("b")))),
            Box::new(var("c")),
        );

        let stmt1 = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Add(
                Box::new(common_expr.clone()),
                Box::new(var("e")),
            )),
        };

        let stmt2 = AstNode::Assign {
            var: "y".to_string(),
            value: Box::new(AstNode::Add(Box::new(common_expr), Box::new(var("f")))),
        };

        let input = block(vec![stmt1, stmt2], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // CSE後は3つのstatementがあるはず
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 3);
        }
    }

    #[test]
    fn test_cse_in_loop() {
        let suggester = CseSuggester::new();

        // ループ内での共通部分式
        // for i in 0..10:
        //   output[i] = (a * (b + c)) + (a * (b + c))
        // 共通部分式: a * (b + c) (コスト2)
        let common_expr = AstNode::Mul(
            Box::new(var("a")),
            Box::new(AstNode::Add(Box::new(var("b")), Box::new(var("c")))),
        );
        let expr = AstNode::Add(Box::new(common_expr.clone()), Box::new(common_expr));

        let loop_body = block(vec![store(var("output"), var("i"), expr)], Scope::new());

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

        // ループ本体のBlock内でCSEが適用されている
        if let AstNode::Block { statements, .. } = &suggestions[0].ast
            && let AstNode::Range { body, .. } = &statements[0]
            && let AstNode::Block {
                statements: inner, ..
            } = body.as_ref()
        {
            // 2つのstatement（一時変数の代入 + store）
            assert_eq!(inner.len(), 2);
        }
    }

    #[test]
    fn test_expr_cost() {
        // 定数はコスト0
        assert_eq!(CseSuggester::expr_cost(&const_int(42)), 0);

        // 変数はコスト0
        assert_eq!(CseSuggester::expr_cost(&var("x")), 0);

        // 単純な加算はコスト1
        let add = AstNode::Add(Box::new(var("a")), Box::new(var("b")));
        assert_eq!(CseSuggester::expr_cost(&add), 1);

        // ネストした式はコストが加算される
        // (a + b) * c -> cost = 1 (mul) + 1 (add) = 2
        let nested = AstNode::Mul(Box::new(add), Box::new(var("c")));
        assert_eq!(CseSuggester::expr_cost(&nested), 2);
    }

    #[test]
    fn test_cse_with_custom_prefix() {
        let suggester = CseSuggester::new().with_prefix("temp");

        let common_expr = AstNode::Mul(
            Box::new(var("a")),
            Box::new(AstNode::Add(Box::new(var("a")), Box::new(const_int(1)))),
        );
        let expr = AstNode::Add(Box::new(common_expr.clone()), Box::new(common_expr));

        let input = block(
            vec![AstNode::Assign {
                var: "y".to_string(),
                value: Box::new(expr),
            }],
            Scope::new(),
        );

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // カスタムプレフィックスが使用されている
        if let AstNode::Block { statements, .. } = &suggestions[0].ast
            && let AstNode::Assign { var, .. } = &statements[0]
        {
            assert!(
                var.starts_with("temp_"),
                "Expected temp_ prefix, got {}",
                var
            );
        }
    }

    #[test]
    fn test_no_cse_single_occurrence() {
        let suggester = CseSuggester::new();

        // 1回しか出現しない式はCSE対象外
        let expr = AstNode::Mul(
            Box::new(var("a")),
            Box::new(AstNode::Add(Box::new(var("b")), Box::new(const_int(1)))),
        );

        let input = block(
            vec![AstNode::Assign {
                var: "y".to_string(),
                value: Box::new(expr),
            }],
            Scope::new(),
        );

        let suggestions = suggester.suggest(&input);

        // 単一の出現ではCSEなし
        assert_eq!(suggestions.len(), 0);
    }
}

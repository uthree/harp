use crate::ast::AstNode;
use crate::ast::pat::AstRewriteRule;
use crate::opt::ast::AstSuggester;
use log::{debug, trace};
use std::collections::HashSet;
use std::rc::Rc;

/// ルールベースの候補提案器
pub struct RuleBaseSuggester {
    rules: Vec<Rc<AstRewriteRule>>,
    /// 深さ優先探索の最大深さ
    max_depth: usize,
}

impl RuleBaseSuggester {
    /// 新しい候補提案器を作成
    pub fn new(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self {
            rules,
            max_depth: 10,
        }
    }

    /// 最大深さを設定
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// 指定されたASTノードにルールを適用して新しいノードを生成
    /// ルールが適用できる全ての位置で書き換えを試みる
    fn apply_rule_at_all_positions(
        &self,
        ast: &AstNode,
        rule: &Rc<AstRewriteRule>,
    ) -> Vec<AstNode> {
        let mut results = Vec::new();

        // 現在のノードに対してルールを試す
        if rule.try_match(ast).is_some() {
            // 条件チェック（AstRewriteRuleの内部でも行われるが、ここでも明示的に）
            let rewritten = rule.apply(ast);
            if rewritten != *ast {
                results.push(rewritten);
            }
        }

        // 子ノードに対して再帰的に適用
        self.apply_to_children(ast, rule, &mut results);

        results
    }

    /// 子ノードに対してルールを適用
    fn apply_to_children(
        &self,
        ast: &AstNode,
        rule: &Rc<AstRewriteRule>,
        results: &mut Vec<AstNode>,
    ) {
        match ast {
            AstNode::Add(l, r)
            | AstNode::Mul(l, r)
            | AstNode::Max(l, r)
            | AstNode::Rem(l, r)
            | AstNode::Idiv(l, r)
            | AstNode::BitwiseAnd(l, r)
            | AstNode::BitwiseOr(l, r)
            | AstNode::BitwiseXor(l, r)
            | AstNode::LeftShift(l, r)
            | AstNode::RightShift(l, r) => {
                // 左側の子に適用
                for new_left in self.apply_rule_at_all_positions(l, rule) {
                    results.push(ast.map_children(&|child| {
                        if child == l.as_ref() {
                            new_left.clone()
                        } else {
                            child.clone()
                        }
                    }));
                }

                // 右側の子に適用
                for new_right in self.apply_rule_at_all_positions(r, rule) {
                    results.push(ast.map_children(&|child| {
                        if child == r.as_ref() {
                            new_right.clone()
                        } else {
                            child.clone()
                        }
                    }));
                }
            }
            AstNode::Recip(n)
            | AstNode::Sqrt(n)
            | AstNode::Log2(n)
            | AstNode::Exp2(n)
            | AstNode::Sin(n)
            | AstNode::BitwiseNot(n)
            | AstNode::Cast(n, _) => {
                for new_child in self.apply_rule_at_all_positions(n, rule) {
                    results.push(ast.map_children(&|_| new_child.clone()));
                }
            }
            AstNode::Load { ptr, offset, .. } => {
                for new_ptr in self.apply_rule_at_all_positions(ptr, rule) {
                    results.push(ast.map_children(&|child| {
                        if child == ptr.as_ref() {
                            new_ptr.clone()
                        } else {
                            child.clone()
                        }
                    }));
                }
                for new_offset in self.apply_rule_at_all_positions(offset, rule) {
                    results.push(ast.map_children(&|child| {
                        if child == offset.as_ref() {
                            new_offset.clone()
                        } else {
                            child.clone()
                        }
                    }));
                }
            }
            AstNode::Store { ptr, offset, value } => {
                for new_ptr in self.apply_rule_at_all_positions(ptr, rule) {
                    results.push(ast.map_children(&|child| {
                        if child == ptr.as_ref() {
                            new_ptr.clone()
                        } else {
                            child.clone()
                        }
                    }));
                }
                for new_offset in self.apply_rule_at_all_positions(offset, rule) {
                    results.push(ast.map_children(&|child| {
                        if child == offset.as_ref() {
                            new_offset.clone()
                        } else {
                            child.clone()
                        }
                    }));
                }
                for new_value in self.apply_rule_at_all_positions(value, rule) {
                    results.push(ast.map_children(&|child| {
                        if child == value.as_ref() {
                            new_value.clone()
                        } else {
                            child.clone()
                        }
                    }));
                }
            }
            AstNode::Assign { value, .. } => {
                for new_value in self.apply_rule_at_all_positions(value, rule) {
                    results.push(ast.map_children(&|_| new_value.clone()));
                }
            }
            AstNode::Block { statements, scope } => {
                for (i, stmt) in statements.iter().enumerate() {
                    for new_stmt in self.apply_rule_at_all_positions(stmt, rule) {
                        let mut new_stmts = statements.clone();
                        new_stmts[i] = new_stmt;
                        results.push(AstNode::Block {
                            statements: new_stmts,
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
                for new_start in self.apply_rule_at_all_positions(start, rule) {
                    results.push(AstNode::Range {
                        var: var.clone(),
                        start: Box::new(new_start),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: body.clone(),
                    });
                }
                for new_step in self.apply_rule_at_all_positions(step, rule) {
                    results.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: Box::new(new_step),
                        stop: stop.clone(),
                        body: body.clone(),
                    });
                }
                for new_stop in self.apply_rule_at_all_positions(stop, rule) {
                    results.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: Box::new(new_stop),
                        body: body.clone(),
                    });
                }
                for new_body in self.apply_rule_at_all_positions(body, rule) {
                    results.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(new_body),
                    });
                }
            }
            AstNode::Return { value } => {
                for new_value in self.apply_rule_at_all_positions(value, rule) {
                    results.push(AstNode::Return {
                        value: Box::new(new_value),
                    });
                }
            }
            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                // 関数本体に対してルールを適用
                for new_body in self.apply_rule_at_all_positions(body, rule) {
                    results.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(new_body),
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
                // カーネル本体に対してルールを適用
                for new_body in self.apply_rule_at_all_positions(body, rule) {
                    results.push(AstNode::Kernel {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(new_body),
                        default_grid_size: default_grid_size.clone(),
                        default_thread_group_size: default_thread_group_size.clone(),
                    });
                }
            }
            AstNode::Program {
                functions,
                entry_point,
                execution_order,
            } => {
                // 各関数に対してルールを適用
                for (i, func) in functions.iter().enumerate() {
                    for new_func in self.apply_rule_at_all_positions(func, rule) {
                        let mut new_functions = functions.clone();
                        new_functions[i] = new_func;
                        results.push(AstNode::Program {
                            functions: new_functions,
                            entry_point: entry_point.clone(),
                            execution_order: execution_order.clone(),
                        });
                    }
                }
            }
            AstNode::Call { name, args } => {
                // 引数に対してルールを適用
                for (i, arg) in args.iter().enumerate() {
                    for new_arg in self.apply_rule_at_all_positions(arg, rule) {
                        let mut new_args = args.clone();
                        new_args[i] = new_arg;
                        results.push(AstNode::Call {
                            name: name.clone(),
                            args: new_args,
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

impl AstSuggester for RuleBaseSuggester {
    fn name(&self) -> &str {
        "RuleBased"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("RuleBaseSuggester: Generating suggestions for AST");
        let mut suggestions = Vec::new();
        let mut seen = HashSet::new();

        // 各ルールを適用して候補を生成
        for rule in &self.rules {
            let candidates = self.apply_rule_at_all_positions(ast, rule);
            for candidate in candidates {
                let candidate_str = format!("{:?}", candidate);
                if !seen.contains(&candidate_str) {
                    seen.insert(candidate_str);
                    suggestions.push(candidate);
                }
            }
        }

        debug!(
            "RuleBaseSuggester: Generated {} unique suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;
    use crate::astpat;

    #[test]
    fn test_rule_base_suggester() {
        // a + b => b + a という交換則
        let rule = astpat!(|a, b| {
            AstNode::Add(Box::new(a), Box::new(b))
        } => {
            AstNode::Add(Box::new(b), Box::new(a))
        });

        let suggester = RuleBaseSuggester::new(vec![rule]);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Int(1))),
            Box::new(AstNode::Const(Literal::Int(2))),
        );

        let suggestions = suggester.suggest(&input);

        // 交換則により1つの候補が生成されるはず
        assert!(!suggestions.is_empty());
        assert_eq!(
            suggestions[0],
            AstNode::Add(
                Box::new(AstNode::Const(Literal::Int(2))),
                Box::new(AstNode::Const(Literal::Int(1))),
            )
        );
    }

    #[test]
    fn test_suggester_multiple_rules() {
        // ルール1: Add(a, 0) -> a
        let rule1 = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Int(0))))
        } => {
            a
        });

        // ルール2: Mul(a, 1) -> a
        let rule2 = astpat!(|a| {
            AstNode::Mul(Box::new(a), Box::new(AstNode::Const(Literal::Int(1))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule1, rule2]);

        // Mul(Add(x, 0), 1)
        let input = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::Int(42))),
                Box::new(AstNode::Const(Literal::Int(0))),
            )),
            Box::new(AstNode::Const(Literal::Int(1))),
        );

        let suggestions = suggester.suggest(&input);

        // 複数の候補が生成されるはず
        // - ルール1で内側のAddを簡約: Mul(42, 1)
        // - ルール2で外側のMulを簡約: Add(42, 0)
        assert!(suggestions.len() >= 2);
    }
}

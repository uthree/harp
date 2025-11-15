//! ループ融合のためのSuggester実装
//!
//! 連続するループで境界が同じ場合に、ループ本体をマージして
//! ループのオーバーヘッドを削減します。

use crate::ast::AstNode;
use crate::opt::ast::Suggester;
use log::{debug, trace};
use std::collections::HashSet;

/// ループ融合を提案するSuggester
pub struct LoopFusionSuggester;

impl LoopFusionSuggester {
    /// 新しいLoopFusionSuggesterを作成
    pub fn new() -> Self {
        Self
    }

    /// 2つのASTノードが構造的に等しいかチェック
    fn ast_equal(a: &AstNode, b: &AstNode) -> bool {
        match (a, b) {
            (AstNode::Const(l1), AstNode::Const(l2)) => l1 == l2,
            (AstNode::Var(n1), AstNode::Var(n2)) => n1 == n2,
            (AstNode::Add(a1, b1), AstNode::Add(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::Mul(a1, b1), AstNode::Mul(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::Max(a1, b1), AstNode::Max(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::Rem(a1, b1), AstNode::Rem(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::Idiv(a1, b1), AstNode::Idiv(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::Recip(a1), AstNode::Recip(a2)) => Self::ast_equal(a1, a2),
            (AstNode::Sqrt(a1), AstNode::Sqrt(a2)) => Self::ast_equal(a1, a2),
            (AstNode::Log2(a1), AstNode::Log2(a2)) => Self::ast_equal(a1, a2),
            (AstNode::Exp2(a1), AstNode::Exp2(a2)) => Self::ast_equal(a1, a2),
            (AstNode::Sin(a1), AstNode::Sin(a2)) => Self::ast_equal(a1, a2),
            (AstNode::Cast(a1, t1), AstNode::Cast(a2, t2)) => t1 == t2 && Self::ast_equal(a1, a2),
            (AstNode::BitwiseAnd(a1, b1), AstNode::BitwiseAnd(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::BitwiseOr(a1, b1), AstNode::BitwiseOr(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::BitwiseXor(a1, b1), AstNode::BitwiseXor(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::BitwiseNot(a1), AstNode::BitwiseNot(a2)) => Self::ast_equal(a1, a2),
            (AstNode::LeftShift(a1, b1), AstNode::LeftShift(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            (AstNode::RightShift(a1, b1), AstNode::RightShift(a2, b2)) => {
                Self::ast_equal(a1, a2) && Self::ast_equal(b1, b2)
            }
            // その他のノードは一致しないとみなす
            _ => false,
        }
    }

    /// ループ変数を置換する
    fn substitute_loop_var(ast: &AstNode, old_var: &str, new_var: &str) -> AstNode {
        match ast {
            AstNode::Var(name) if name == old_var => AstNode::Var(new_var.to_string()),

            // 再帰的に子ノードを処理
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::Max(a, b) => AstNode::Max(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::Rem(a, b) => AstNode::Rem(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::Idiv(a, b) => AstNode::Idiv(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::Recip(a) => {
                AstNode::Recip(Box::new(Self::substitute_loop_var(a, old_var, new_var)))
            }
            AstNode::Sqrt(a) => {
                AstNode::Sqrt(Box::new(Self::substitute_loop_var(a, old_var, new_var)))
            }
            AstNode::Log2(a) => {
                AstNode::Log2(Box::new(Self::substitute_loop_var(a, old_var, new_var)))
            }
            AstNode::Exp2(a) => {
                AstNode::Exp2(Box::new(Self::substitute_loop_var(a, old_var, new_var)))
            }
            AstNode::Sin(a) => {
                AstNode::Sin(Box::new(Self::substitute_loop_var(a, old_var, new_var)))
            }
            AstNode::Cast(a, dtype) => AstNode::Cast(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                dtype.clone(),
            ),
            AstNode::BitwiseAnd(a, b) => AstNode::BitwiseAnd(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::BitwiseOr(a, b) => AstNode::BitwiseOr(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::BitwiseXor(a, b) => AstNode::BitwiseXor(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::BitwiseNot(a) => {
                AstNode::BitwiseNot(Box::new(Self::substitute_loop_var(a, old_var, new_var)))
            }
            AstNode::LeftShift(a, b) => AstNode::LeftShift(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::RightShift(a, b) => AstNode::RightShift(
                Box::new(Self::substitute_loop_var(a, old_var, new_var)),
                Box::new(Self::substitute_loop_var(b, old_var, new_var)),
            ),
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => AstNode::Load {
                ptr: Box::new(Self::substitute_loop_var(ptr, old_var, new_var)),
                offset: Box::new(Self::substitute_loop_var(offset, old_var, new_var)),
                count: *count,
                dtype: dtype.clone(),
            },
            AstNode::Store { ptr, offset, value } => AstNode::Store {
                ptr: Box::new(Self::substitute_loop_var(ptr, old_var, new_var)),
                offset: Box::new(Self::substitute_loop_var(offset, old_var, new_var)),
                value: Box::new(Self::substitute_loop_var(value, old_var, new_var)),
            },
            AstNode::Assign { var, value } => AstNode::Assign {
                var: if var == old_var {
                    new_var.to_string()
                } else {
                    var.clone()
                },
                value: Box::new(Self::substitute_loop_var(value, old_var, new_var)),
            },
            AstNode::Block { statements, scope } => AstNode::Block {
                statements: statements
                    .iter()
                    .map(|s| Self::substitute_loop_var(s, old_var, new_var))
                    .collect(),
                scope: scope.clone(),
            },
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                // ネストしたRangeのループ変数と衝突する場合はスキップ
                if var == old_var {
                    ast.clone()
                } else {
                    AstNode::Range {
                        var: var.clone(),
                        start: Box::new(Self::substitute_loop_var(start, old_var, new_var)),
                        step: Box::new(Self::substitute_loop_var(step, old_var, new_var)),
                        stop: Box::new(Self::substitute_loop_var(stop, old_var, new_var)),
                        body: Box::new(Self::substitute_loop_var(body, old_var, new_var)),
                    }
                }
            }
            AstNode::Call { name, args } => AstNode::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_loop_var(a, old_var, new_var))
                    .collect(),
            },
            AstNode::Return { value } => AstNode::Return {
                value: Box::new(Self::substitute_loop_var(value, old_var, new_var)),
            },
            // その他のノードはそのまま返す
            _ => ast.clone(),
        }
    }

    /// ループのbodyをマージする
    /// 両方のbodyがBlockなら、statementsを連結する
    /// そうでない場合は、新しいBlockを作成する
    fn merge_loop_bodies(body1: &AstNode, body2: &AstNode) -> AstNode {
        use crate::ast::Scope;

        match (body1, body2) {
            (
                AstNode::Block {
                    statements: s1,
                    scope: _,
                },
                AstNode::Block {
                    statements: s2,
                    scope: _,
                },
            ) => {
                let mut merged = s1.clone();
                merged.extend(s2.clone());
                AstNode::Block {
                    statements: merged,
                    scope: Box::new(Scope::new()),
                }
            }
            (
                AstNode::Block {
                    statements: s1,
                    scope: _,
                },
                other,
            ) => {
                let mut merged = s1.clone();
                merged.push(other.clone());
                AstNode::Block {
                    statements: merged,
                    scope: Box::new(Scope::new()),
                }
            }
            (
                other,
                AstNode::Block {
                    statements: s2,
                    scope: _,
                },
            ) => {
                let mut merged = vec![other.clone()];
                merged.extend(s2.clone());
                AstNode::Block {
                    statements: merged,
                    scope: Box::new(Scope::new()),
                }
            }
            (b1, b2) => AstNode::Block {
                statements: vec![b1.clone(), b2.clone()],
                scope: Box::new(Scope::new()),
            },
        }
    }

    /// Block内で連続するRangeをfusionできるか試みる
    /// Barrierノードを挟む場合は融合しない（依存関係の保証）
    fn try_fuse_in_block(&self, statements: &[AstNode]) -> Option<Vec<AstNode>> {
        if statements.len() < 2 {
            return None;
        }

        let mut new_statements = Vec::new();
        let mut i = 0;
        let mut fused = false;

        while i < statements.len() {
            if i + 1 < statements.len() {
                // Barrierノードは融合の境界として扱う
                if matches!(statements[i], AstNode::Barrier) {
                    new_statements.push(statements[i].clone());
                    i += 1;
                    continue;
                }

                // 連続する2つのRangeをチェック
                if let (
                    AstNode::Range {
                        var: var1,
                        start: start1,
                        step: step1,
                        stop: stop1,
                        body: body1,
                    },
                    AstNode::Range {
                        var: var2,
                        start: start2,
                        step: step2,
                        stop: stop2,
                        body: body2,
                    },
                ) = (&statements[i], &statements[i + 1])
                {
                    // 境界が同じかチェック
                    if Self::ast_equal(start1, start2)
                        && Self::ast_equal(step1, step2)
                        && Self::ast_equal(stop1, stop2)
                    {
                        debug!("Found fusable loops: var1='{}', var2='{}'", var1, var2);

                        // ループ変数が異なる場合、2番目のbodyの変数を1番目に合わせる
                        let adjusted_body2 = if var1 != var2 {
                            Self::substitute_loop_var(body2, var2, var1)
                        } else {
                            body2.as_ref().clone()
                        };

                        // bodyをマージ
                        let merged_body = Self::merge_loop_bodies(body1, &adjusted_body2);

                        let fused_range = AstNode::Range {
                            var: var1.clone(),
                            start: start1.clone(),
                            step: step1.clone(),
                            stop: stop1.clone(),
                            body: Box::new(merged_body),
                        };

                        new_statements.push(fused_range);
                        fused = true;
                        i += 2; // 2つのRangeをスキップ
                        continue;
                    }
                }
            }

            new_statements.push(statements[i].clone());
            i += 1;
        }

        if fused {
            trace!(
                "Fused loops, reduced {} to {} statements",
                statements.len(),
                new_statements.len()
            );
            Some(new_statements)
        } else {
            None
        }
    }

    /// ASTツリーを走査して、融合可能なループを探す
    fn try_fuse_in_ast(&self, ast: &AstNode) -> Option<AstNode> {
        match ast {
            AstNode::Block { statements, scope } => {
                // まずBlock内の連続するRangeの融合を試みる
                if let Some(fused_statements) = self.try_fuse_in_block(statements) {
                    return Some(AstNode::Block {
                        statements: fused_statements,
                        scope: scope.clone(),
                    });
                }

                // 子要素を再帰的に処理
                let mut new_statements = Vec::new();
                let mut changed = false;

                for stmt in statements {
                    if let Some(new_stmt) = self.try_fuse_in_ast(stmt) {
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
                // ループ本体内で融合を試みる
                self.try_fuse_in_ast(body).map(|new_body| AstNode::Range {
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
                kind,
            } => self
                .try_fuse_in_ast(body)
                .map(|new_body| AstNode::Function {
                    name: name.clone(),
                    params: params.clone(),
                    return_type: return_type.clone(),
                    body: Box::new(new_body),
                    kind: kind.clone(),
                }),

            AstNode::Program {
                functions,
                entry_point,
            } => {
                let mut new_functions = Vec::new();
                let mut changed = false;

                for func in functions {
                    if let Some(new_func) = self.try_fuse_in_ast(func) {
                        new_functions.push(new_func);
                        changed = true;
                    } else {
                        new_functions.push(func.clone());
                    }
                }

                if changed {
                    Some(AstNode::Program {
                        functions: new_functions,
                        entry_point: entry_point.clone(),
                    })
                } else {
                    None
                }
            }

            _ => None,
        }
    }

    /// Program全体から融合候補を収集
    fn collect_fusion_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        if let Some(fused) = self.try_fuse_in_ast(ast) {
            candidates.push(fused);
        }

        candidates
    }
}

impl Default for LoopFusionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl Suggester for LoopFusionSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("LoopFusionSuggester: Generating loop fusion suggestions");
        let mut suggestions = Vec::new();
        let mut seen = HashSet::new();

        let candidates = self.collect_fusion_candidates(ast);

        for candidate in candidates {
            let candidate_str = format!("{:?}", candidate);
            if !seen.contains(&candidate_str) {
                seen.insert(candidate_str);
                suggestions.push(candidate);
            }
        }

        debug!(
            "LoopFusionSuggester: Generated {} unique suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Scope;
    use crate::ast::helper::{block, const_int, range, store, var};

    #[test]
    fn test_simple_loop_fusion() {
        let suggester = LoopFusionSuggester::new();

        // 2つの同じ境界を持つループ
        let loop1 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("a"), var("i"), const_int(1)),
        );

        let loop2 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("b"), var("i"), const_int(2)),
        );

        let program_body = block(vec![loop1, loop2], Scope::new());

        let suggestions = suggester.suggest(&program_body);

        assert_eq!(suggestions.len(), 1);

        // 融合後は1つのRangeになるはず
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 1);

            if let AstNode::Range { body, .. } = &statements[0] {
                // bodyは2つのstoreを含むBlock
                if let AstNode::Block {
                    statements: inner, ..
                } = body.as_ref()
                {
                    assert_eq!(inner.len(), 2);
                } else {
                    panic!("Expected Block body after fusion");
                }
            } else {
                panic!("Expected Range after fusion");
            }
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_loop_fusion_different_var_names() {
        let suggester = LoopFusionSuggester::new();

        // ループ変数名が異なるが境界は同じ
        let loop1 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(50),
            store(var("a"), var("i"), const_int(1)),
        );

        let loop2 = range(
            "j",
            const_int(0),
            const_int(1),
            const_int(50),
            store(var("b"), var("j"), const_int(2)),
        );

        let program_body = block(vec![loop1, loop2], Scope::new());

        let suggestions = suggester.suggest(&program_body);

        assert_eq!(suggestions.len(), 1);

        // 融合後、jはiに置換されるはず
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            if let AstNode::Range { var, body, .. } = &statements[0] {
                assert_eq!(var, "i");

                // 2番目のstoreでvar("j")がvar("i")に置換されているか確認
                if let AstNode::Block {
                    statements: inner, ..
                } = body.as_ref()
                {
                    if let AstNode::Store { offset, .. } = &inner[1] {
                        assert_eq!(offset.as_ref(), &AstNode::Var("i".to_string()));
                    }
                }
            }
        }
    }

    #[test]
    fn test_no_fusion_different_bounds() {
        let suggester = LoopFusionSuggester::new();

        // 境界が異なるループは融合しない
        let loop1 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("a"), var("i"), const_int(1)),
        );

        let loop2 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(200), // 異なる境界
            store(var("b"), var("i"), const_int(2)),
        );

        let program_body = block(vec![loop1, loop2], Scope::new());

        let suggestions = suggester.suggest(&program_body);

        // 融合できないので提案は0
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_nested_loop_fusion() {
        let suggester = LoopFusionSuggester::new();

        // ネストしたループ内での融合
        let inner_loop1 = range(
            "j",
            const_int(0),
            const_int(1),
            const_int(10),
            store(var("a"), var("j"), const_int(1)),
        );

        let inner_loop2 = range(
            "j",
            const_int(0),
            const_int(1),
            const_int(10),
            store(var("b"), var("j"), const_int(2)),
        );

        let outer_loop = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            block(vec![inner_loop1, inner_loop2], Scope::new()),
        );

        let program_body = block(vec![outer_loop], Scope::new());

        let suggestions = suggester.suggest(&program_body);

        assert_eq!(suggestions.len(), 1);

        // 外側のループ内で内側のループが融合されているはず
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            if let AstNode::Range { body, .. } = &statements[0] {
                if let AstNode::Block {
                    statements: inner, ..
                } = body.as_ref()
                {
                    // 内側は1つのRangeに融合されているはず
                    assert_eq!(inner.len(), 1);
                }
            }
        }
    }

    #[test]
    fn test_no_fusion_across_barrier() {
        let suggester = LoopFusionSuggester::new();

        // Barrierを挟むループは融合しない
        let loop1 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("a"), var("i"), const_int(1)),
        );

        let barrier = AstNode::Barrier;

        let loop2 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("b"), var("i"), const_int(2)),
        );

        let program_body = block(vec![loop1, barrier, loop2], Scope::new());

        let suggestions = suggester.suggest(&program_body);

        // Barrierを挟むので融合されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_fusion_between_barriers() {
        let suggester = LoopFusionSuggester::new();

        // Barrierの間のループは融合可能
        let loop1 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("a"), var("i"), const_int(1)),
        );

        let loop2 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("b"), var("i"), const_int(2)),
        );

        let barrier = AstNode::Barrier;

        let loop3 = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var("c"), var("i"), const_int(3)),
        );

        let program_body = block(vec![loop1, loop2, barrier, loop3], Scope::new());

        let suggestions = suggester.suggest(&program_body);

        // loop1とloop2が融合される
        assert_eq!(suggestions.len(), 1);

        if let AstNode::Block { statements, .. } = &suggestions[0] {
            // Barrierの前に1つのRangeになっているはず
            assert_eq!(statements.len(), 3); // fused_loop, barrier, loop3
            assert!(matches!(statements[0], AstNode::Range { .. }));
            assert!(matches!(statements[1], AstNode::Barrier));
            assert!(matches!(statements[2], AstNode::Range { .. }));
        } else {
            panic!("Expected Block");
        }
    }
}

//! ループ最適化のためのSuggester実装
//!
//! タイル化とインライン展開をビームサーチで使えるようにします。

use crate::ast::AstNode;
use crate::opt::ast::Suggester;
use crate::opt::ast::transforms::{inline_small_loop, tile_loop};
use log::{debug, trace};
use std::collections::HashSet;

/// ループタイル化を提案するSuggester
pub struct LoopTilingSuggester {
    /// 試行するタイルサイズのリスト
    tile_sizes: Vec<usize>,
}

impl LoopTilingSuggester {
    /// 新しいLoopTilingSuggesterを作成
    pub fn new(tile_sizes: Vec<usize>) -> Self {
        Self { tile_sizes }
    }

    /// デフォルトのタイルサイズで作成
    pub fn with_default_sizes() -> Self {
        Self {
            tile_sizes: vec![2, 4, 8, 16],
        }
    }

    /// AST内の全てのRangeノードを探索してタイル化を試みる
    fn collect_tiling_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        // 現在のノードがRangeの場合、各タイルサイズで変換を試みる
        if matches!(ast, AstNode::Range { .. }) {
            for &tile_size in &self.tile_sizes {
                if let Some(tiled) = tile_loop(ast, tile_size) {
                    candidates.push(tiled);
                }
            }
        }

        // 子ノードを再帰的に探索
        match ast {
            AstNode::Block { statements, scope } => {
                for (i, stmt) in statements.iter().enumerate() {
                    for tiled_stmt in self.collect_tiling_candidates(stmt) {
                        let mut new_stmts = statements.clone();
                        new_stmts[i] = tiled_stmt;
                        candidates.push(AstNode::Block {
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
                // ループ本体を再帰的に探索
                for tiled_body in self.collect_tiling_candidates(body) {
                    candidates.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(tiled_body),
                    });
                }
            }
            AstNode::Function {
                name,
                params,
                return_type,
                body,
                kind,
            } => {
                // 関数本体を再帰的に探索
                for tiled_body in self.collect_tiling_candidates(body) {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(tiled_body),
                        kind: kind.clone(),
                    });
                }
            }
            AstNode::Program {
                functions,
                entry_point,
            } => {
                // 各関数を再帰的に探索
                for (i, func) in functions.iter().enumerate() {
                    for tiled_func in self.collect_tiling_candidates(func) {
                        let mut new_functions = functions.clone();
                        new_functions[i] = tiled_func;
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

impl Suggester for LoopTilingSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("LoopTilingSuggester: Generating tiling suggestions");
        let mut suggestions = Vec::new();
        let mut seen = HashSet::new();

        let candidates = self.collect_tiling_candidates(ast);

        for candidate in candidates {
            let candidate_str = format!("{:?}", candidate);
            if !seen.contains(&candidate_str) {
                seen.insert(candidate_str);
                suggestions.push(candidate);
            }
        }

        debug!(
            "LoopTilingSuggester: Generated {} unique suggestions",
            suggestions.len()
        );
        suggestions
    }
}

/// ループインライン展開を提案するSuggester
pub struct LoopInliningSuggester {
    /// 展開する最大反復回数
    max_iterations: usize,
}

impl LoopInliningSuggester {
    /// 新しいLoopInliningSuggesterを作成
    pub fn new(max_iterations: usize) -> Self {
        Self { max_iterations }
    }

    /// デフォルトの設定で作成（最大8回まで展開）
    pub fn with_default_limit() -> Self {
        Self { max_iterations: 8 }
    }

    /// AST内の全てのRangeノードを探索してインライン展開を試みる
    fn collect_inlining_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        // 現在のノードがRangeの場合、インライン展開を試みる
        if matches!(ast, AstNode::Range { .. })
            && let Some(inlined) = inline_small_loop(ast, self.max_iterations)
        {
            candidates.push(inlined);
        }

        // 子ノードを再帰的に探索
        match ast {
            AstNode::Block { statements, scope } => {
                for (i, stmt) in statements.iter().enumerate() {
                    for inlined_stmt in self.collect_inlining_candidates(stmt) {
                        let mut new_stmts = statements.clone();
                        new_stmts[i] = inlined_stmt;
                        candidates.push(AstNode::Block {
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
                // ループ本体を再帰的に探索
                for inlined_body in self.collect_inlining_candidates(body) {
                    candidates.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(inlined_body),
                    });
                }
            }
            AstNode::Function {
                name,
                params,
                return_type,
                body,
                kind,
            } => {
                // 関数本体を再帰的に探索
                for inlined_body in self.collect_inlining_candidates(body) {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(inlined_body),
                        kind: kind.clone(),
                    });
                }
            }
            AstNode::Program {
                functions,
                entry_point,
            } => {
                // 各関数を再帰的に探索
                for (i, func) in functions.iter().enumerate() {
                    for inlined_func in self.collect_inlining_candidates(func) {
                        let mut new_functions = functions.clone();
                        new_functions[i] = inlined_func;
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

impl Suggester for LoopInliningSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("LoopInliningSuggester: Generating inlining suggestions");
        let mut suggestions = Vec::new();
        let mut seen = HashSet::new();

        let candidates = self.collect_inlining_candidates(ast);

        for candidate in candidates {
            let candidate_str = format!("{:?}", candidate);
            if !seen.contains(&candidate_str) {
                seen.insert(candidate_str);
                suggestions.push(candidate);
            }
        }

        debug!(
            "LoopInliningSuggester: Generated {} unique suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_loop_tiling_suggester() {
        let suggester = LoopTilingSuggester::with_default_sizes();

        // for i in 0..16 step 1 { Store(ptr, i, i) }
        let body = Box::new(AstNode::Store {
            ptr: Box::new(AstNode::Var("ptr".to_string())),
            offset: Box::new(AstNode::Var("i".to_string())),
            value: Box::new(AstNode::Var("i".to_string())),
        });

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            stop: Box::new(AstNode::Const(Literal::Isize(16))),
            body,
        };

        let suggestions = suggester.suggest(&loop_node);

        // デフォルトで4つのタイルサイズ（2, 4, 8, 16）を試すので、
        // 4つの候補が生成されるはず
        assert_eq!(suggestions.len(), 4);
    }

    #[test]
    fn test_loop_inlining_suggester() {
        let suggester = LoopInliningSuggester::with_default_limit();

        // for i in 0..4 step 1 { Store(ptr, i, i) }
        let body = Box::new(AstNode::Store {
            ptr: Box::new(AstNode::Var("ptr".to_string())),
            offset: Box::new(AstNode::Var("i".to_string())),
            value: Box::new(AstNode::Var("i".to_string())),
        });

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            stop: Box::new(AstNode::Const(Literal::Isize(4))),
            body,
        };

        let suggestions = suggester.suggest(&loop_node);

        // 4回のループなので展開可能、1つの候補が生成されるはず
        assert_eq!(suggestions.len(), 1);

        // 展開結果がBlockノードになっているか確認
        assert!(matches!(suggestions[0], AstNode::Block { .. }));
    }

    #[test]
    fn test_loop_inlining_suggester_too_large() {
        let suggester = LoopInliningSuggester::new(4);

        // for i in 0..10 step 1 { body }
        let body = Box::new(AstNode::Var("x".to_string()));

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            stop: Box::new(AstNode::Const(Literal::Isize(10))),
            body,
        };

        let suggestions = suggester.suggest(&loop_node);

        // max_iterations=4なので、10回のループは展開されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_nested_loop_suggestions() {
        let suggester = LoopTilingSuggester::new(vec![4]);

        // 外側ループ: for i in 0..16 step 1 { 内側ループ }
        // 内側ループ: for j in 0..16 step 1 { body }
        let inner_body = Box::new(AstNode::Var("x".to_string()));

        let inner_loop = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            stop: Box::new(AstNode::Const(Literal::Isize(16))),
            body: inner_body,
        };

        let outer_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Isize(0))),
            step: Box::new(AstNode::Const(Literal::Isize(1))),
            stop: Box::new(AstNode::Const(Literal::Isize(16))),
            body: Box::new(inner_loop),
        };

        let suggestions = suggester.suggest(&outer_loop);

        // 外側ループと内側ループのタイル化候補が生成される
        // 少なくとも2つの候補があるはず
        assert!(suggestions.len() >= 2);
    }
}

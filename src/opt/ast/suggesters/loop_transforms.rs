//! ループ最適化のためのSuggester実装
//!
//! タイル化とインライン展開をビームサーチで使えるようにします。

use crate::ast::AstNode;
use crate::opt::ast::Suggester;
use crate::opt::ast::transforms::{inline_small_loop, tile_loop};
use log::{debug, trace};

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
            tile_sizes: vec![2, 3, 4, 5, 7, 8, 16, 32, 64],
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
        let candidates = self.collect_tiling_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);
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

    /// デフォルトの設定で作成（最大4回まで展開）
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
        let candidates = self.collect_inlining_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);
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
    use crate::ast::helper::{const_int, range, store, var};

    #[test]
    fn test_loop_tiling_suggester() {
        let suggester = LoopTilingSuggester::with_default_sizes();

        // for i in 0..16 step 1 { Store(ptr, i, i) }
        let loop_node = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(16),
            store(var("ptr"), var("i"), var("i")),
        );

        let suggestions = suggester.suggest(&loop_node);

        // デフォルトで6つのタイルサイズ（2, 4, 8, 16, 32, 64）を試す
        // ループ範囲が16なので、割り切れるのは2, 4, 8, 16の4つ
        // 各タイルサイズで複数のタイリング候補が生成される可能性がある
        // 重複排除後、実際に生成されるユニークな候補数を確認
        assert!(
            suggestions.len() >= 4,
            "Expected at least 4 suggestions, got {}",
            suggestions.len()
        );
    }

    #[test]
    fn test_loop_inlining_suggester() {
        let suggester = LoopInliningSuggester::with_default_limit();

        // for i in 0..4 step 1 { Store(ptr, i, i) }
        let loop_node = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(4),
            store(var("ptr"), var("i"), var("i")),
        );

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
        let loop_node = range("i", const_int(0), const_int(1), const_int(10), var("x"));

        let suggestions = suggester.suggest(&loop_node);

        // max_iterations=4なので、10回のループは展開されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_nested_loop_suggestions() {
        let suggester = LoopTilingSuggester::new(vec![4]);

        // 外側ループ: for i in 0..16 step 1 { 内側ループ }
        // 内側ループ: for j in 0..16 step 1 { body }
        let inner_loop = range("j", const_int(0), const_int(1), const_int(16), var("x"));

        let outer_loop = range("i", const_int(0), const_int(1), const_int(16), inner_loop);

        let suggestions = suggester.suggest(&outer_loop);

        // 外側ループと内側ループのタイル化候補が生成される
        // 少なくとも2つの候補があるはず
        assert!(suggestions.len() >= 2);
    }
}

/// ループ交換（Loop Interchange）を提案するSuggester
///
/// ネストしたループの順序を入れ替えることで、キャッシュ効率を改善します。
/// 例: for i { for j { body } } → for j { for i { body } }
pub struct LoopInterchangeSuggester;

impl LoopInterchangeSuggester {
    /// 新しいLoopInterchangeSuggesterを作成
    pub fn new() -> Self {
        Self
    }

    /// ループ交換を試みる
    ///
    /// 外側ループの直下、またはBlockを挟んで内側ループがある場合に交換可能
    fn try_interchange(ast: &AstNode) -> Option<AstNode> {
        if let AstNode::Range {
            var: outer_var,
            start: outer_start,
            step: outer_step,
            stop: outer_stop,
            body: outer_body,
        } = ast
        {
            // 外側ループの本体が内側ループの場合（直接）
            if let AstNode::Range {
                var: inner_var,
                start: inner_start,
                step: inner_step,
                stop: inner_stop,
                body: inner_body,
            } = outer_body.as_ref()
            {
                // ループを交換: 内側を外側に、外側を内側に
                let new_inner = AstNode::Range {
                    var: outer_var.clone(),
                    start: outer_start.clone(),
                    step: outer_step.clone(),
                    stop: outer_stop.clone(),
                    body: inner_body.clone(),
                };

                let new_outer = AstNode::Range {
                    var: inner_var.clone(),
                    start: inner_start.clone(),
                    step: inner_step.clone(),
                    stop: inner_stop.clone(),
                    body: Box::new(new_inner),
                };

                return Some(new_outer);
            }

            // 外側ループの本体がBlockで、その中に単一のRangeがある場合
            if let AstNode::Block { statements, scope } = outer_body.as_ref()
                && statements.len() == 1
                && let AstNode::Range {
                    var: inner_var,
                    start: inner_start,
                    step: inner_step,
                    stop: inner_stop,
                    body: inner_body,
                } = &statements[0]
            {
                // ループを交換: 内側を外側に、外側を内側に
                let new_inner = AstNode::Range {
                    var: outer_var.clone(),
                    start: outer_start.clone(),
                    step: outer_step.clone(),
                    stop: outer_stop.clone(),
                    body: inner_body.clone(),
                };

                let new_outer_body = AstNode::Block {
                    statements: vec![new_inner],
                    scope: scope.clone(),
                };

                let new_outer = AstNode::Range {
                    var: inner_var.clone(),
                    start: inner_start.clone(),
                    step: inner_step.clone(),
                    stop: inner_stop.clone(),
                    body: Box::new(new_outer_body),
                };

                return Some(new_outer);
            }
        }

        None
    }

    /// AST内の全てのネストループを探索して交換を試みる
    #[allow(clippy::only_used_in_recursion)]
    fn collect_interchange_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        // 現在のノードがネストループの場合、交換を試みる
        if let Some(interchanged) = Self::try_interchange(ast) {
            candidates.push(interchanged);
        }

        // 子ノードを再帰的に探索
        match ast {
            AstNode::Block { statements, scope } => {
                for (i, stmt) in statements.iter().enumerate() {
                    for interchanged_stmt in self.collect_interchange_candidates(stmt) {
                        let mut new_stmts = statements.clone();
                        new_stmts[i] = interchanged_stmt;
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
                // ループ本体を再帰的に探索（ただし直下の場合は上で処理済み）
                for interchanged_body in self.collect_interchange_candidates(body) {
                    candidates.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(interchanged_body),
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
                for interchanged_body in self.collect_interchange_candidates(body) {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(interchanged_body),
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
                    for interchanged_func in self.collect_interchange_candidates(func) {
                        let mut new_functions = functions.clone();
                        new_functions[i] = interchanged_func;
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

impl Default for LoopInterchangeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl Suggester for LoopInterchangeSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("LoopInterchangeSuggester: Generating loop interchange suggestions");
        let candidates = self.collect_interchange_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);
        debug!(
            "LoopInterchangeSuggester: Generated {} unique suggestions",
            suggestions.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod interchange_tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_loop_interchange_basic() {
        let suggester = LoopInterchangeSuggester::new();

        // 外側ループ: for i in 0..M step 1 { 内側ループ }
        // 内側ループ: for j in 0..N step 1 { body }
        let inner_body = Box::new(AstNode::Var("x".to_string()));

        let inner_loop = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body: inner_body.clone(),
        };

        let outer_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(20))),
            body: Box::new(inner_loop),
        };

        let suggestions = suggester.suggest(&outer_loop);

        // 1つの交換候補が生成されるはず
        assert_eq!(suggestions.len(), 1);

        // 交換結果を検証
        if let AstNode::Range {
            var: new_outer_var,
            body: new_outer_body,
            ..
        } = &suggestions[0]
        {
            // 外側がjになっているはず
            assert_eq!(new_outer_var, "j");

            if let AstNode::Range {
                var: new_inner_var,
                body: new_inner_body,
                ..
            } = new_outer_body.as_ref()
            {
                // 内側がiになっているはず
                assert_eq!(new_inner_var, "i");
                // 最も内側のbodyは変わらないはず
                assert_eq!(new_inner_body.as_ref(), inner_body.as_ref());
            } else {
                panic!("Expected inner Range node");
            }
        } else {
            panic!("Expected outer Range node");
        }
    }

    #[test]
    fn test_loop_interchange_not_applicable() {
        let suggester = LoopInterchangeSuggester::new();

        // 単一のループ（ネストしていない）
        let single_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body: Box::new(AstNode::Var("x".to_string())),
        };

        let suggestions = suggester.suggest(&single_loop);

        // ネストしていないので交換候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_triple_nested_loops() {
        let suggester = LoopInterchangeSuggester::new();

        // 3重ネストループ: for i { for j { for k { body } } }
        let innermost_body = Box::new(AstNode::Var("x".to_string()));

        let k_loop = AstNode::Range {
            var: "k".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(5))),
            body: innermost_body,
        };

        let j_loop = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(10))),
            body: Box::new(k_loop),
        };

        let i_loop = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(20))),
            body: Box::new(j_loop),
        };

        let suggestions = suggester.suggest(&i_loop);

        // 3重ネストの場合、2つのペアを交換できる：
        // 1. i-j を交換 (外側2つ)
        // 2. j-k を交換 (内側2つ)
        assert_eq!(suggestions.len(), 2);
    }
}

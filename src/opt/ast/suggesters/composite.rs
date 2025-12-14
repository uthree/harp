use crate::ast::AstNode;
use crate::opt::ast::AstSuggester;
use log::{debug, trace};
use std::collections::HashSet;

/// 複数のSuggesterを組み合わせるSuggester
pub struct CompositeSuggester {
    suggesters: Vec<Box<dyn AstSuggester>>,
}

impl CompositeSuggester {
    /// 新しいCompositeSuggesterを作成
    pub fn new(suggesters: Vec<Box<dyn AstSuggester>>) -> Self {
        Self { suggesters }
    }
}

impl AstSuggester for CompositeSuggester {
    fn suggest(&self, ast: &AstNode) -> Vec<AstNode> {
        trace!("CompositeSuggester: Generating suggestions from multiple suggesters");
        let mut suggestions = Vec::new();
        let mut seen = HashSet::new();

        // 各Suggesterから候補を収集
        for suggester in &self.suggesters {
            let candidates = suggester.suggest(ast);
            for candidate in candidates {
                let candidate_str = format!("{:?}", candidate);
                if !seen.contains(&candidate_str) {
                    seen.insert(candidate_str);
                    suggestions.push(candidate);
                }
            }
        }

        debug!(
            "CompositeSuggester: Generated {} unique suggestions from {} suggesters",
            suggestions.len(),
            self.suggesters.len()
        );
        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, Literal, Mutability, VarDecl, VarKind};
    use crate::opt::ast::rules::all_rules_with_search;
    use crate::opt::ast::suggesters::{
        FunctionInliningSuggester, LoopInliningSuggester, LoopTilingSuggester, RuleBaseSuggester,
        ThreadPartitionSuggester,
    };

    #[test]
    fn test_composite_suggester_with_all_optimizations() {
        // 全ての最適化を含むSuggester
        let suggester = CompositeSuggester::new(vec![
            Box::new(RuleBaseSuggester::new(all_rules_with_search())),
            Box::new(LoopTilingSuggester::with_default_sizes()),
            Box::new(LoopInliningSuggester::with_default_limit()),
        ]);

        // for i in 0..4 step 1 { body }
        let body = Box::new(AstNode::Add(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(0))),
        ));

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(4))),
            body,
        };

        let suggestions = suggester.suggest(&loop_node);

        // ルールベース（Add(x, 0) -> x）、タイル化、インライン展開の
        // 候補が含まれるはず
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_composite_suggester_rules_only() {
        // ルールベース最適化のみ
        let suggester = CompositeSuggester::new(vec![Box::new(RuleBaseSuggester::new(
            all_rules_with_search(),
        ))]);

        // x + 0
        let input = AstNode::Add(
            Box::new(AstNode::Var("x".to_string())),
            Box::new(AstNode::Const(Literal::Int(0))),
        );

        let suggestions = suggester.suggest(&input);

        // 交換則などの候補が生成されるはず
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_composite_suggester_loop_only() {
        // ループ最適化のみ
        let suggester = CompositeSuggester::new(vec![
            Box::new(LoopTilingSuggester::with_default_sizes()),
            Box::new(LoopInliningSuggester::with_default_limit()),
        ]);

        // for i in 0..4 step 1 { body }
        let body = Box::new(AstNode::Var("x".to_string()));

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(4))),
            body,
        };

        let suggestions = suggester.suggest(&loop_node);

        // タイル化とインライン展開の候補が生成されるはず
        // タイル化: デフォルトで4つのタイルサイズ（2, 4, 8, 16）
        // インライン展開: 1つ
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_composite_suggester_custom() {
        // カスタム: インライン展開のみ
        let suggester =
            CompositeSuggester::new(vec![Box::new(LoopInliningSuggester::with_default_limit())]);

        // for i in 0..4 step 1 { body }
        let body = Box::new(AstNode::Var("x".to_string()));

        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(Literal::Int(0))),
            step: Box::new(AstNode::Const(Literal::Int(1))),
            stop: Box::new(AstNode::Const(Literal::Int(4))),
            body,
        };

        let suggestions = suggester.suggest(&loop_node);

        // インライン展開の候補が1つ生成されるはず
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_composite_suggester_with_function_inlining() {
        // 関数インライン展開を含む統合テスト
        let suggester = CompositeSuggester::new(vec![
            Box::new(RuleBaseSuggester::new(all_rules_with_search())),
            Box::new(FunctionInliningSuggester::with_default_limit()),
        ]);

        // fn add_one(x: Int) -> Int { return x + 1 }
        let add_one_func = AstNode::Function {
            name: Some("add_one".to_string()),
            params: vec![VarDecl {
                name: "x".to_string(),
                dtype: DType::Int,
                mutability: Mutability::Immutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::Int,
            body: Box::new(AstNode::Return {
                value: Box::new(AstNode::Add(
                    Box::new(AstNode::Var("x".to_string())),
                    Box::new(AstNode::Const(Literal::Int(1))),
                )),
            }),
        };

        // fn main() -> Int { return add_one(5) }
        let main_func = AstNode::Function {
            name: Some("main".to_string()),
            params: vec![],
            return_type: DType::Int,
            body: Box::new(AstNode::Return {
                value: Box::new(AstNode::Call {
                    name: "add_one".to_string(),
                    args: vec![AstNode::Const(Literal::Int(5))],
                }),
            }),
        };

        let program = AstNode::Program {
            functions: vec![add_one_func, main_func],
            entry_point: "main".to_string(),
        };

        let suggestions = suggester.suggest(&program);

        // 関数インライン展開とルールベース最適化の候補が生成されるはず
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_composite_suggester_with_thread_partition() {
        use crate::ast::helper::*;

        // ThreadPartitionSuggesterを含む統合テスト
        let suggester = CompositeSuggester::new(vec![
            Box::new(RuleBaseSuggester::new(all_rules_with_search())),
            Box::new(ThreadPartitionSuggester::new()),
        ]);

        // 1D FlatParallel Kernelを作成
        let body = AstNode::If {
            condition: Box::new(lt(var("tid"), var("shape_0") * var("shape_1"))),
            then_body: Box::new(store(var("output"), var("tid"), var("value"))),
            else_body: None,
        };

        let kernel = AstNode::Kernel {
            name: Some("test_kernel".to_string()),
            params: vec![
                VarDecl {
                    name: "tid".to_string(),
                    dtype: DType::Int,
                    mutability: Mutability::Immutable,
                    kind: VarKind::ThreadId(0),
                },
                VarDecl {
                    name: "input_0".to_string(),
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
            body: Box::new(body),
            default_grid_size: [
                Box::new(const_int(1024)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
            default_thread_group_size: [
                Box::new(const_int(256)),
                Box::new(const_int(1)),
                Box::new(const_int(1)),
            ],
        };

        let suggestions = suggester.suggest(&kernel);

        // ThreadPartitionSuggesterからの候補が生成されるはず
        assert!(!suggestions.is_empty());

        // 少なくとも1つのKernelがtid_0, tid_1を持つ（多次元化されている）
        let has_multidim_kernel = suggestions.iter().any(|s| {
            if let AstNode::Kernel { params, .. } = s {
                params.iter().any(|p| p.name == "tid_0") && params.iter().any(|p| p.name == "tid_1")
            } else {
                false
            }
        });
        assert!(
            has_multidim_kernel,
            "Should have multidim partitioned kernel"
        );
    }
}

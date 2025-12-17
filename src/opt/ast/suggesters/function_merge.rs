//! Function統合のためのSuggester実装
//!
//! 複数のAstNode::Functionを1つのFunctionに統合し、
//! 後段のLoopFusionSuggesterでループ融合を可能にします。

use crate::ast::{AstNode, Scope, VarDecl};
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};

/// Function統合を提案するSuggester
///
/// Program内の隣接するFunction同士をマージし、
/// 各Functionのbodyを1つのBlock内に配置します。
/// 後段でLoopFusionSuggesterが同一境界のRangeを融合します。
pub struct FunctionMergeSuggester;

impl FunctionMergeSuggester {
    /// 新しいFunctionMergeSuggesterを作成
    pub fn new() -> Self {
        Self
    }

    /// 2つのFunctionのパラメータを統合（重複除去）
    fn merge_params(params1: &[VarDecl], params2: &[VarDecl]) -> Vec<VarDecl> {
        let mut merged = params1.to_vec();

        for p2 in params2 {
            // 同じ名前のパラメータがなければ追加
            if !merged.iter().any(|p1| p1.name == p2.name) {
                merged.push(p2.clone());
            }
        }

        merged
    }

    /// 2つのFunctionをマージ
    fn merge_functions(f1: &AstNode, f2: &AstNode) -> Option<AstNode> {
        if let (
            AstNode::Function {
                name: name1,
                params: params1,
                return_type: ret1,
                body: body1,
            },
            AstNode::Function {
                name: name2,
                params: params2,
                return_type: _ret2,
                body: body2,
            },
        ) = (f1, f2)
        {
            // パラメータを統合
            let merged_params = Self::merge_params(params1, params2);

            // 新しい名前を生成
            let new_name = match (name1, name2) {
                (Some(n1), Some(n2)) => Some(format!("merged_{}_{}", n1, n2)),
                (Some(n1), None) => Some(format!("merged_{}", n1)),
                (None, Some(n2)) => Some(format!("merged_{}", n2)),
                (None, None) => Some("merged".to_string()),
            };

            // bodyをBlockのstatementsとして連結
            let merged_body = AstNode::Block {
                statements: vec![body1.as_ref().clone(), body2.as_ref().clone()],
                scope: Box::new(Scope::new()),
            };

            Some(AstNode::Function {
                name: new_name,
                params: merged_params,
                return_type: ret1.clone(),
                body: Box::new(merged_body),
            })
        } else {
            None
        }
    }

    /// 2つのKernelをマージ
    fn merge_kernels(k1: &AstNode, k2: &AstNode) -> Option<AstNode> {
        if let (
            AstNode::Kernel {
                name: name1,
                params: params1,
                return_type: ret1,
                body: body1,
                default_grid_size: grid1,
                default_thread_group_size: tg1,
            },
            AstNode::Kernel {
                name: name2,
                params: params2,
                return_type: _ret2,
                body: body2,
                default_grid_size: _grid2,
                default_thread_group_size: _tg2,
            },
        ) = (k1, k2)
        {
            // パラメータを統合
            let merged_params = Self::merge_params(params1, params2);

            // 新しい名前を生成
            let new_name = match (name1, name2) {
                (Some(n1), Some(n2)) => Some(format!("merged_{}_{}", n1, n2)),
                (Some(n1), None) => Some(format!("merged_{}", n1)),
                (None, Some(n2)) => Some(format!("merged_{}", n2)),
                (None, None) => Some("merged".to_string()),
            };

            // bodyをBlockのstatementsとして連結
            let merged_body = AstNode::Block {
                statements: vec![body1.as_ref().clone(), body2.as_ref().clone()],
                scope: Box::new(Scope::new()),
            };

            Some(AstNode::Kernel {
                name: new_name,
                params: merged_params,
                return_type: ret1.clone(),
                body: Box::new(merged_body),
                default_grid_size: grid1.clone(),
                default_thread_group_size: tg1.clone(),
            })
        } else {
            None
        }
    }

    /// Program内の隣接するFunction/Kernelをマージ
    fn try_merge_in_program(&self, functions: &[AstNode]) -> Option<Vec<AstNode>> {
        if functions.len() < 2 {
            return None;
        }

        let mut new_functions = Vec::new();
        let mut i = 0;
        let mut merged = false;

        while i < functions.len() {
            if i + 1 < functions.len() {
                // Function同士のマージを試みる
                if let Some(merged_func) = Self::merge_functions(&functions[i], &functions[i + 1]) {
                    debug!(
                        "Merged functions: {:?} and {:?}",
                        Self::get_function_name(&functions[i]),
                        Self::get_function_name(&functions[i + 1])
                    );
                    new_functions.push(merged_func);
                    merged = true;
                    i += 2;
                    continue;
                }

                // Kernel同士のマージを試みる
                if let Some(merged_kernel) = Self::merge_kernels(&functions[i], &functions[i + 1]) {
                    debug!(
                        "Merged kernels: {:?} and {:?}",
                        Self::get_kernel_name(&functions[i]),
                        Self::get_kernel_name(&functions[i + 1])
                    );
                    new_functions.push(merged_kernel);
                    merged = true;
                    i += 2;
                    continue;
                }
            }

            new_functions.push(functions[i].clone());
            i += 1;
        }

        if merged {
            trace!(
                "Merged functions, reduced {} to {} functions",
                functions.len(),
                new_functions.len()
            );
            Some(new_functions)
        } else {
            None
        }
    }

    /// Functionの名前を取得（デバッグ用）
    fn get_function_name(node: &AstNode) -> Option<&String> {
        if let AstNode::Function { name, .. } = node {
            name.as_ref()
        } else {
            None
        }
    }

    /// Kernelの名前を取得（デバッグ用）
    fn get_kernel_name(node: &AstNode) -> Option<&String> {
        if let AstNode::Kernel { name, .. } = node {
            name.as_ref()
        } else {
            None
        }
    }

    /// マージ候補を収集
    fn collect_merge_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        if let AstNode::Program {
            functions,
            execution_order,
        } = ast
            && let Some(merged_functions) = self.try_merge_in_program(functions)
        {
            candidates.push(AstNode::Program {
                functions: merged_functions,
                execution_order: execution_order.clone(),
            });
        }

        candidates
    }
}

impl Default for FunctionMergeSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for FunctionMergeSuggester {
    fn name(&self) -> &str {
        "FunctionMerge"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        trace!("FunctionMergeSuggester: Generating function merge suggestions");
        let candidates = self.collect_merge_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);

        debug!(
            "FunctionMergeSuggester: Generated {} unique suggestions",
            suggestions.len()
        );

        suggestions
            .into_iter()
            .map(|ast| AstSuggestResult::with_description(ast, self.name(), "merge functions"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{const_int, range, store, var};
    use crate::ast::{DType, Mutability, VarKind};

    fn make_simple_function(name: &str, output_var: &str) -> AstNode {
        let body = range(
            "i",
            const_int(0),
            const_int(1),
            const_int(100),
            store(var(output_var), var("i"), const_int(1)),
        );

        AstNode::Function {
            name: Some(name.to_string()),
            params: vec![VarDecl {
                name: output_var.to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::Tuple(vec![]),
            body: Box::new(body),
        }
    }

    #[test]
    fn test_simple_function_merge() {
        let suggester = FunctionMergeSuggester::new();

        let f1 = make_simple_function("f1", "a");
        let f2 = make_simple_function("f2", "b");

        let program = AstNode::Program {
            functions: vec![f1, f2],
            execution_order: None,
        };

        let suggestions = suggester.suggest(&program);

        assert_eq!(suggestions.len(), 1);

        if let AstNode::Program { functions, .. } = &suggestions[0].ast {
            assert_eq!(functions.len(), 1);

            if let AstNode::Function {
                name, params, body, ..
            } = &functions[0]
            {
                // 名前が統合されている
                assert_eq!(name.as_ref().unwrap(), "merged_f1_f2");

                // パラメータが統合されている
                assert_eq!(params.len(), 2);

                // bodyがBlockで2つのRangeを含む
                if let AstNode::Block { statements, .. } = body.as_ref() {
                    assert_eq!(statements.len(), 2);
                    assert!(matches!(statements[0], AstNode::Range { .. }));
                    assert!(matches!(statements[1], AstNode::Range { .. }));
                } else {
                    panic!("Expected Block body");
                }
            } else {
                panic!("Expected Function");
            }
        } else {
            panic!("Expected Program");
        }
    }

    #[test]
    fn test_no_merge_single_function() {
        let suggester = FunctionMergeSuggester::new();

        let f1 = make_simple_function("f1", "a");

        let program = AstNode::Program {
            functions: vec![f1],
            execution_order: None,
        };

        let suggestions = suggester.suggest(&program);

        // 単一のFunctionはマージ対象外
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_merge_three_functions() {
        let suggester = FunctionMergeSuggester::new();

        let f1 = make_simple_function("f1", "a");
        let f2 = make_simple_function("f2", "b");
        let f3 = make_simple_function("f3", "c");

        let program = AstNode::Program {
            functions: vec![f1, f2, f3],
            execution_order: None,
        };

        let suggestions = suggester.suggest(&program);

        assert_eq!(suggestions.len(), 1);

        if let AstNode::Program { functions, .. } = &suggestions[0].ast {
            // f1とf2がマージされ、f3は残る
            assert_eq!(functions.len(), 2);
        }
    }

    #[test]
    fn test_param_deduplication() {
        let suggester = FunctionMergeSuggester::new();

        // 同じパラメータを持つ2つのFunction
        let body1 = store(var("shared"), const_int(0), const_int(1));
        let body2 = store(var("shared"), const_int(1), const_int(2));

        let f1 = AstNode::Function {
            name: Some("f1".to_string()),
            params: vec![VarDecl {
                name: "shared".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::Tuple(vec![]),
            body: Box::new(body1),
        };

        let f2 = AstNode::Function {
            name: Some("f2".to_string()),
            params: vec![VarDecl {
                name: "shared".to_string(),
                dtype: DType::Ptr(Box::new(DType::F32)),
                mutability: Mutability::Mutable,
                kind: VarKind::Normal,
            }],
            return_type: DType::Tuple(vec![]),
            body: Box::new(body2),
        };

        let program = AstNode::Program {
            functions: vec![f1, f2],
            execution_order: None,
        };

        let suggestions = suggester.suggest(&program);

        assert_eq!(suggestions.len(), 1);

        if let AstNode::Program { functions, .. } = &suggestions[0].ast
            && let AstNode::Function { params, .. } = &functions[0]
        {
            // 重複が除去され、1つのパラメータのみ
            assert_eq!(params.len(), 1);
            assert_eq!(params[0].name, "shared");
        }
    }
}

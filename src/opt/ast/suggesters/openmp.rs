//! OpenMP並列化Suggester
//!
//! ループを分析して並列化可能な場合に `ParallelInfo` を設定します。
//! リダクション変数も検出して適切に処理します。

use crate::ast::{AstNode, ParallelInfo, ParallelKind, ReductionOp};
use crate::opt::ast::{AstSuggestResult, AstSuggester};

use super::parallelization_common::LoopAnalyzer;

/// OpenMP並列化を提案するSuggester
///
/// ループを分析して並列化可能な場合に `ParallelInfo { is_parallel: true, kind: OpenMP, ... }`
/// を設定した AST を提案します。
///
/// # 並列化条件
///
/// - Store操作のオフセットがループ変数に依存していること（レースコンディション回避）
/// - 外部変数への書き込みがある場合、リダクションパターンとして検出可能であること
///
/// # OpenMP特有の挙動
///
/// GPU並列化と異なり、動的分岐（If文）があっても並列化可能です。
/// OpenMPはスレッド間で異なる命令を実行できるため、分岐ダイバージェンスの問題がありません。
#[derive(Debug, Clone, Default)]
pub struct OpenMPParallelizationSuggester {
    /// 最小ループ回数（これ未満は並列化しない）
    min_iterations: Option<i64>,
}

impl OpenMPParallelizationSuggester {
    /// 新しいSuggesterを作成
    pub fn new() -> Self {
        Self {
            min_iterations: None,
        }
    }

    /// 最小ループ回数を設定
    pub fn with_min_iterations(mut self, min: i64) -> Self {
        self.min_iterations = Some(min);
        self
    }

    /// Function内のRangeループを並列化
    fn process_function(&self, func: &AstNode) -> Vec<AstSuggestResult> {
        let AstNode::Function {
            name,
            params,
            return_type,
            body,
        } = func
        else {
            return vec![];
        };

        // 最外側のRangeループを探す
        if let Some(transformed_body) = self.try_parallelize_range(body) {
            let new_func = AstNode::Function {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(transformed_body),
            };

            return vec![AstSuggestResult::with_description(
                new_func,
                self.name(),
                "Parallelize loop with OpenMP".to_string(),
            )];
        }

        vec![]
    }

    /// Program内の関数を処理
    fn process_program(&self, program: &AstNode) -> Vec<AstSuggestResult> {
        let AstNode::Program {
            functions,
            execution_waves,
        } = program
        else {
            return vec![];
        };

        let mut results = vec![];

        for (idx, func) in functions.iter().enumerate() {
            for suggestion in self.process_function(func) {
                // 提案されたASTをProgram内の対応する関数に適用
                let mut new_functions = functions.clone();
                new_functions[idx] = suggestion.ast;

                results.push(AstSuggestResult::with_description(
                    AstNode::Program {
                        functions: new_functions,
                        execution_waves: execution_waves.clone(),
                    },
                    self.name(),
                    suggestion.description,
                ));
            }
        }

        results
    }

    /// Rangeループの並列化を試みる
    fn try_parallelize_range(&self, node: &AstNode) -> Option<AstNode> {
        match node {
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
                parallel,
            } => {
                // 既に並列化済みならスキップ
                if parallel.is_parallel {
                    return None;
                }

                // ループ回数のチェック（最小回数が設定されている場合）
                if let Some(min) = self.min_iterations
                    && let (Some(start_val), Some(stop_val)) = (
                        Self::try_get_const_i64(start),
                        Self::try_get_const_i64(stop),
                    ) {
                        let iterations = stop_val - start_val;
                        if iterations < min {
                            return None;
                        }
                    }

                // ループ本体を分析
                let mut analyzer = LoopAnalyzer::new(var);
                analyzer.analyze(body);

                // 並列化可能かチェック（リダクションを許容）
                if !analyzer.is_parallelizable_with_reduction() {
                    return None;
                }

                // リダクション変数を検出
                let reductions = self.detect_reductions(body, analyzer.external_writes());

                // 外部書き込みがあるがリダクションとして検出できなかった場合は並列化不可
                if !analyzer.external_writes().is_empty() && reductions.is_empty() {
                    log::trace!(
                        "Loop not parallelizable (OpenMP): external writes but no reduction pattern detected"
                    );
                    return None;
                }

                // 並列化情報を設定
                let new_parallel = ParallelInfo {
                    is_parallel: true,
                    kind: ParallelKind::OpenMP,
                    reductions,
                };

                Some(AstNode::Range {
                    var: var.clone(),
                    start: start.clone(),
                    step: step.clone(),
                    stop: stop.clone(),
                    body: body.clone(),
                    parallel: new_parallel,
                })
            }
            AstNode::Block { statements, scope } => {
                // Block内の最初のRangeを並列化
                for (idx, stmt) in statements.iter().enumerate() {
                    if let Some(transformed) = self.try_parallelize_range(stmt) {
                        let mut new_statements = statements.clone();
                        new_statements[idx] = transformed;
                        return Some(AstNode::Block {
                            statements: new_statements,
                            scope: scope.clone(),
                        });
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// 定数整数値を取得
    fn try_get_const_i64(node: &AstNode) -> Option<i64> {
        match node {
            AstNode::Const(lit) => lit.as_i64(),
            _ => None,
        }
    }

    /// リダクション変数を検出
    fn detect_reductions(
        &self,
        body: &AstNode,
        external_writes: &std::collections::HashSet<String>,
    ) -> Vec<(String, ReductionOp)> {
        let mut reductions = Vec::new();

        for var_name in external_writes {
            if let Some(op) = self.find_reduction_pattern(body, var_name) {
                reductions.push((var_name.clone(), op));
            }
        }

        reductions
    }

    /// 指定された変数に対するリダクションパターンを探す
    fn find_reduction_pattern(&self, body: &AstNode, var_name: &str) -> Option<ReductionOp> {
        match body {
            AstNode::Assign { var, value } if var == var_name => {
                // var = var + expr, var = expr + var
                if let AstNode::Add(left, right) = value.as_ref()
                    && (Self::is_var(left, var_name) || Self::is_var(right, var_name)) {
                        return Some(ReductionOp::Add);
                    }
                // var = var * expr, var = expr * var
                if let AstNode::Mul(left, right) = value.as_ref()
                    && (Self::is_var(left, var_name) || Self::is_var(right, var_name)) {
                        return Some(ReductionOp::Mul);
                    }
                // var = max(var, expr), var = max(expr, var)
                if let AstNode::Max(left, right) = value.as_ref()
                    && (Self::is_var(left, var_name) || Self::is_var(right, var_name)) {
                        return Some(ReductionOp::Max);
                    }
                None
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    if let Some(op) = self.find_reduction_pattern(stmt, var_name) {
                        return Some(op);
                    }
                }
                None
            }
            AstNode::If {
                then_body,
                else_body,
                ..
            } => {
                if let Some(op) = self.find_reduction_pattern(then_body, var_name) {
                    return Some(op);
                }
                if let Some(else_b) = else_body
                    && let Some(op) = self.find_reduction_pattern(else_b, var_name) {
                        return Some(op);
                    }
                None
            }
            _ => None,
        }
    }

    /// 指定された名前の変数かどうかを判定
    fn is_var(node: &AstNode, name: &str) -> bool {
        matches!(node, AstNode::Var(n) if n == name)
    }
}

impl AstSuggester for OpenMPParallelizationSuggester {
    fn name(&self) -> &str {
        "OpenMPParallelization"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        match ast {
            AstNode::Program { .. } => self.process_program(ast),
            AstNode::Function { .. } => self.process_function(ast),
            AstNode::Range { .. } => {
                if let Some(transformed) = self.try_parallelize_range(ast) {
                    vec![AstSuggestResult::with_description(
                        transformed,
                        self.name(),
                        "Parallelize loop with OpenMP".to_string(),
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
    use crate::ast::DType;
    use crate::ast::helper::{assign, const_int, load, range, store, var};

    #[test]
    fn test_simple_parallel_for() {
        // for i in 0..N { output[i] = input[i] * 2 }
        let body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32) * const_int(2),
        );
        let loop_node = range("i", const_int(0), const_int(1), var("N"), body);

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&loop_node);

        assert_eq!(suggestions.len(), 1);

        // 提案されたASTでParallelInfoが設定されていることを確認
        if let AstNode::Range { parallel, .. } = &suggestions[0].ast {
            assert!(parallel.is_parallel);
            assert_eq!(parallel.kind, ParallelKind::OpenMP);
            assert!(parallel.reductions.is_empty());
        } else {
            panic!("Expected Range node");
        }
    }

    #[test]
    fn test_reduction_add() {
        // sum = 0; for i in 0..N { sum = sum + input[i] }
        let body = assign("sum", var("sum") + load(var("input"), var("i"), DType::F32));
        let loop_node = range("i", const_int(0), const_int(1), var("N"), body);

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&loop_node);

        assert_eq!(suggestions.len(), 1);

        // リダクションが検出されていることを確認
        if let AstNode::Range { parallel, .. } = &suggestions[0].ast {
            assert!(parallel.is_parallel);
            assert_eq!(parallel.kind, ParallelKind::OpenMP);
            assert_eq!(parallel.reductions.len(), 1);
            assert_eq!(
                parallel.reductions[0],
                ("sum".to_string(), ReductionOp::Add)
            );
        } else {
            panic!("Expected Range node");
        }
    }

    #[test]
    fn test_reduction_max() {
        // max_val = max(max_val, input[i])
        let body = assign(
            "max_val",
            AstNode::Max(
                Box::new(var("max_val")),
                Box::new(load(var("input"), var("i"), DType::F32)),
            ),
        );
        let loop_node = range("i", const_int(0), const_int(1), var("N"), body);

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&loop_node);

        assert_eq!(suggestions.len(), 1);

        if let AstNode::Range { parallel, .. } = &suggestions[0].ast {
            assert!(parallel.is_parallel);
            assert_eq!(parallel.reductions.len(), 1);
            assert_eq!(
                parallel.reductions[0],
                ("max_val".to_string(), ReductionOp::Max)
            );
        } else {
            panic!("Expected Range node");
        }
    }

    #[test]
    fn test_non_parallelizable_race() {
        // for i in 0..N { output[0] = input[i] }
        // -> 並列化不可（レースコンディション: 固定オフセット）
        let body = store(
            var("output"),
            const_int(0), // 固定オフセット
            load(var("input"), var("i"), DType::F32),
        );
        let loop_node = range("i", const_int(0), const_int(1), var("N"), body);

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&loop_node);

        // 並列化不可なので提案なし
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_non_parallelizable_unknown_pattern() {
        // for i in 0..N { result = input[i] }  // 単純な代入（リダクションではない）
        // -> 並列化不可（外部書き込みだがリダクションパターンではない）
        let body = assign("result", load(var("input"), var("i"), DType::F32));
        let loop_node = range("i", const_int(0), const_int(1), var("N"), body);

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&loop_node);

        // リダクションパターンではないので並列化不可
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_with_dynamic_branch() {
        // for i in 0..N { if (i % 2 == 0) { output[i] = input[i] } }
        // -> OpenMPでは並列化可能
        let inner_body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );
        let if_node = AstNode::If {
            condition: Box::new(AstNode::Lt(
                Box::new(AstNode::Rem(Box::new(var("i")), Box::new(const_int(2)))),
                Box::new(const_int(1)),
            )),
            then_body: Box::new(inner_body),
            else_body: None,
        };
        let loop_node = range("i", const_int(0), const_int(1), var("N"), if_node);

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&loop_node);

        // OpenMPでは動的分岐があっても並列化可能
        assert_eq!(suggestions.len(), 1);

        if let AstNode::Range { parallel, .. } = &suggestions[0].ast {
            assert!(parallel.is_parallel);
            assert_eq!(parallel.kind, ParallelKind::OpenMP);
        } else {
            panic!("Expected Range node");
        }
    }

    #[test]
    fn test_already_parallelized() {
        // 既に並列化済みのループはスキップ
        let body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );
        let loop_node = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(var("N")),
            body: Box::new(body),
            parallel: ParallelInfo {
                is_parallel: true,
                kind: ParallelKind::OpenMP,
                reductions: vec![],
            },
        };

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&loop_node);

        // 既に並列化済みなので提案なし
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_min_iterations() {
        // 小さいループは並列化しない
        let body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );
        let loop_node = range("i", const_int(0), const_int(1), const_int(10), body);

        let suggester = OpenMPParallelizationSuggester::new().with_min_iterations(100);
        let suggestions = suggester.suggest(&loop_node);

        // ループ回数が100未満なので並列化しない
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_function_parallelization() {
        // Function内のループを並列化
        let body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );
        let loop_node = range("i", const_int(0), const_int(1), var("N"), body);

        let func = AstNode::Function {
            name: Some("test_func".to_string()),
            params: vec![],
            return_type: DType::Void,
            body: Box::new(loop_node),
        };

        let suggester = OpenMPParallelizationSuggester::new();
        let suggestions = suggester.suggest(&func);

        assert_eq!(suggestions.len(), 1);

        // Function内のRangeが並列化されていることを確認
        if let AstNode::Function { body, .. } = &suggestions[0].ast {
            if let AstNode::Range { parallel, .. } = body.as_ref() {
                assert!(parallel.is_parallel);
                assert_eq!(parallel.kind, ParallelKind::OpenMP);
            } else {
                panic!("Expected Range node in function body");
            }
        } else {
            panic!("Expected Function node");
        }
    }
}

use crate::ast::AstNode;
use log::{debug, trace};
use std::collections::HashMap;
use std::rc::Rc;

/// パターンマッチングの結果
pub type MatchResult = Option<HashMap<String, AstNode>>;

/// 書き換え関数の型
pub type RewriteFn = Rc<dyn Fn(&HashMap<String, AstNode>) -> AstNode>;

/// 条件関数の型
pub type ConditionFn = Rc<dyn Fn(&HashMap<String, AstNode>) -> bool>;

/// ASTの書き換えルール
pub struct AstRewriteRule {
    /// パターン（Wildcard含むAstNode）
    pattern: AstNode,
    /// 書き換え関数（マッチした変数を受け取り、新しいASTを返す）
    rewriter: RewriteFn,
    /// 条件関数（マッチした変数を受け取り、このルールを適用するか判定）
    condition: ConditionFn,
}

impl AstRewriteRule {
    /// 新しい書き換えルールを作成
    pub fn new<F, C>(pattern: AstNode, rewriter: F, condition: C) -> Rc<Self>
    where
        F: Fn(&HashMap<String, AstNode>) -> AstNode + 'static,
        C: Fn(&HashMap<String, AstNode>) -> bool + 'static,
    {
        Rc::new(AstRewriteRule {
            pattern,
            rewriter: Rc::new(rewriter),
            condition: Rc::new(condition),
        })
    }

    /// パターンマッチングを試みる
    pub fn try_match(&self, ast: &AstNode) -> MatchResult {
        trace!("Trying to match pattern: {:?}", self.pattern);
        trace!("Against AST: {:?}", ast);
        let mut bindings = HashMap::new();
        if self.pattern_match(&self.pattern, ast, &mut bindings) {
            debug!("Pattern matched! Bindings: {:?}", bindings);
            Some(bindings)
        } else {
            trace!("Pattern did not match");
            None
        }
    }

    /// パターンマッチングの内部実装
    #[allow(clippy::only_used_in_recursion)]
    fn pattern_match(
        &self,
        pattern: &AstNode,
        ast: &AstNode,
        bindings: &mut HashMap<String, AstNode>,
    ) -> bool {
        // 二項演算子のマッチングを処理するマクロ
        macro_rules! match_binary_op {
            ($variant:ident) => {
                if let AstNode::$variant(pl, pr) = pattern {
                    if let AstNode::$variant(al, ar) = ast {
                        return self.pattern_match(pl, al, bindings)
                            && self.pattern_match(pr, ar, bindings);
                    } else {
                        return false;
                    }
                }
            };
        }

        // 単項演算子のマッチングを処理するマクロ
        macro_rules! match_unary_op {
            ($variant:ident) => {
                if let AstNode::$variant(p) = pattern {
                    if let AstNode::$variant(a) = ast {
                        return self.pattern_match(p, a, bindings);
                    } else {
                        return false;
                    }
                }
            };
        }

        // 二項演算子のマッチング
        match_binary_op!(Add);
        match_binary_op!(Mul);
        match_binary_op!(Max);
        match_binary_op!(Rem);
        match_binary_op!(Idiv);

        // 単項演算子のマッチング
        match_unary_op!(Recip);
        match_unary_op!(Sqrt);
        match_unary_op!(Log2);
        match_unary_op!(Exp2);
        match_unary_op!(Sin);

        match pattern {
            AstNode::Wildcard(name) => {
                // すでにバインドされている場合は、同じノードか確認
                if let Some(bound) = bindings.get(name) {
                    bound == ast
                } else {
                    bindings.insert(name.clone(), ast.clone());
                    true
                }
            }
            AstNode::Const(p) => {
                if let AstNode::Const(a) = ast {
                    p == a
                } else {
                    false
                }
            }
            AstNode::Cast(p, pt) => {
                if let AstNode::Cast(a, at) = ast {
                    pt == at && self.pattern_match(p, a, bindings)
                } else {
                    false
                }
            }
            AstNode::Var(pv) => {
                if let AstNode::Var(av) = ast {
                    pv == av
                } else {
                    false
                }
            }
            AstNode::Load {
                ptr: p_ptr,
                offset: p_offset,
                count: p_count,
                dtype: p_dtype,
            } => {
                if let AstNode::Load {
                    ptr: a_ptr,
                    offset: a_offset,
                    count: a_count,
                    dtype: a_dtype,
                } = ast
                {
                    p_count == a_count
                        && p_dtype == a_dtype
                        && self.pattern_match(p_ptr, a_ptr, bindings)
                        && self.pattern_match(p_offset, a_offset, bindings)
                } else {
                    false
                }
            }
            AstNode::Store {
                ptr: p_ptr,
                offset: p_offset,
                value: p_value,
            } => {
                if let AstNode::Store {
                    ptr: a_ptr,
                    offset: a_offset,
                    value: a_value,
                } = ast
                {
                    self.pattern_match(p_ptr, a_ptr, bindings)
                        && self.pattern_match(p_offset, a_offset, bindings)
                        && self.pattern_match(p_value, a_value, bindings)
                } else {
                    false
                }
            }
            AstNode::Assign {
                var: p_var,
                value: p_value,
            } => {
                if let AstNode::Assign {
                    var: a_var,
                    value: a_value,
                } = ast
                {
                    p_var == a_var && self.pattern_match(p_value, a_value, bindings)
                } else {
                    false
                }
            }
            // その他の未実装ノードは一致しない
            _ => false,
        }
    }

    /// ASTノードに対してルールを適用（再帰的に探索）
    pub fn apply(&self, ast: &AstNode) -> AstNode {
        // まず現在のノードにマッチを試みる
        if let Some(bindings) = self.try_match(ast) {
            if (self.condition)(&bindings) {
                let result = (self.rewriter)(&bindings);
                debug!("Applied rewrite rule:");
                debug!("  Before: {:?}", ast);
                debug!("  After:  {:?}", result);
                return result;
            } else {
                trace!("Pattern matched but condition failed");
            }
        }

        // マッチしなければ、子ノードに対して再帰的に適用
        ast.map_children(&|child| self.apply(child))
    }
}

/// 複数のルールを管理し、順次適用する
pub struct AstRewriter {
    rules: Vec<Rc<AstRewriteRule>>,
    max_iterations: usize,
}

impl AstRewriter {
    /// 新しいリライタを作成
    pub fn new(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        AstRewriter {
            rules,
            max_iterations: 100,
        }
    }

    /// 最大反復回数を設定
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    /// すべてのルールを適用（変化がなくなるまで繰り返す）
    pub fn apply(&self, mut ast: AstNode) -> AstNode {
        debug!("Starting AST rewriter with {} rules", self.rules.len());
        trace!("Initial AST: {:?}", ast);

        for iteration in 0..self.max_iterations {
            let prev = format!("{:?}", ast);

            // すべてのルールを順番に適用
            for (rule_idx, rule) in self.rules.iter().enumerate() {
                trace!("Applying rule {} in iteration {}", rule_idx, iteration);
                ast = rule.apply(&ast);
            }

            // 変化がなければ終了
            if format!("{:?}", ast) == prev {
                debug!("AST rewriter converged after {} iterations", iteration);
                break;
            } else {
                trace!("Iteration {} completed, AST changed", iteration);
            }
        }

        debug!("Final AST after rewriting: {:?}", ast);
        ast
    }
}

/// パターンを構築するヘルパーマクロ
///
/// 使用例:
/// ```
/// use eclat::astpat;
/// let rule = astpat!(|a, b| (a + b) => (b + a));
/// ```
#[macro_export]
macro_rules! astpat {
    // 条件なしパターン: |vars...| pattern => replacement
    (|$($var:ident),*| $pattern:expr => $replacement:expr) => {
        $crate::ast::pat::AstRewriteRule::new(
            astpat!(@pattern $pattern; $($var),*),
            {
                #[allow(unused_variables)]
                let vars = vec![$(stringify!($var)),*];
                move |bindings: &std::collections::HashMap<String, $crate::ast::AstNode>| {
                    astpat!(@replace $replacement, bindings; $($var),*)
                }
            },
            |_| true
        )
    };

    // パターン内の変数を実際のWildcardに変換
    (@pattern $pattern:expr; $($var:ident),*) => {{
        // 変数名のリストを作成
        $(let $var = $crate::ast::AstNode::Wildcard(stringify!($var).to_string());)*
        $pattern
    }};

    // 置き換え式内の変数をbindingsから取得
    (@replace $replacement:expr, $bindings:expr; $($var:ident),*) => {{
        $(
            let $var = $bindings.get(stringify!($var))
                .expect(&format!("Variable {} not found in bindings", stringify!($var)))
                .clone();
        )*
        $replacement
    }};
}

/// 複数のルールをまとめてリライタを作成するマクロ
#[macro_export]
macro_rules! ast_rewriter {
    ($($rule:expr),* $(,)?) => {
        $crate::ast::pat::AstRewriter::new(vec![$($rule),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_wildcard_pattern() {
        // パターン: Wildcard("x")
        let pattern = AstNode::Wildcard("x".to_string());
        let rule = AstRewriteRule::new(
            pattern,
            |bindings| {
                // x を取得して、それを2倍にする: x * 2
                let x = bindings.get("x").unwrap().clone();
                AstNode::Mul(Box::new(x), Box::new(AstNode::Const(Literal::I64(2))))
            },
            |_| true,
        );

        let input = AstNode::Const(Literal::I64(5));
        let result = rule.apply(&input);

        // 5 -> 5 * 2
        match result {
            AstNode::Mul(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::I64(5)));
                assert_eq!(*right, AstNode::Const(Literal::I64(2)));
            }
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_rewriter_apply() {
        // パターン: Add(Wildcard("a"), Wildcard("b"))
        // 置き換え: Add(b, a) - 交換法則
        let pattern = AstNode::Add(
            Box::new(AstNode::Wildcard("a".to_string())),
            Box::new(AstNode::Wildcard("b".to_string())),
        );
        let rule = AstRewriteRule::new(
            pattern,
            |bindings| {
                let a = bindings.get("a").unwrap().clone();
                let b = bindings.get("b").unwrap().clone();
                AstNode::Add(Box::new(b), Box::new(a))
            },
            |_| true,
        );

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::I64(1))),
            Box::new(AstNode::Const(Literal::I64(2))),
        );
        let result = rule.apply(&input);

        // Add(1, 2) -> Add(2, 1)
        match result {
            AstNode::Add(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::I64(2)));
                assert_eq!(*right, AstNode::Const(Literal::I64(1)));
            }
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_astpat_macro_simple() {
        // マクロを使った簡単なパターン: a + b => b + a
        let rule = astpat!(|a, b| {
            AstNode::Add(Box::new(a), Box::new(b))
        } => {
            AstNode::Add(Box::new(b), Box::new(a))
        });

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::I64(3))),
            Box::new(AstNode::Const(Literal::I64(4))),
        );
        let result = rule.apply(&input);

        match result {
            AstNode::Add(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::I64(4)));
                assert_eq!(*right, AstNode::Const(Literal::I64(3)));
            }
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_ast_rewriter_macro() {
        // 複数のルールを組み合わせたリライタ
        // Add(a, 0) -> a という簡約ルールを使う
        let rule1 = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::I64(0))))
        } => {
            a
        });

        let rewriter = ast_rewriter![rule1];

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::I64(5))),
            Box::new(AstNode::Const(Literal::I64(0))),
        );
        let result = rewriter.apply(input);

        // Add(5, 0) -> 5
        assert_eq!(result, AstNode::Const(Literal::I64(5)));
    }

    #[test]
    fn test_pattern_wildcard_matching() {
        // ワイルドカードが同じ変数名で複数回現れる場合、同じ値にマッチする必要がある
        // パターン: Add(x, x) - 同じ値の加算
        let pattern = AstNode::Add(
            Box::new(AstNode::Wildcard("x".to_string())),
            Box::new(AstNode::Wildcard("x".to_string())),
        );
        let rule = AstRewriteRule::new(
            pattern,
            |bindings| {
                let x = bindings.get("x").unwrap().clone();
                // x + x -> x * 2
                AstNode::Mul(Box::new(x), Box::new(AstNode::Const(Literal::I64(2))))
            },
            |_| true,
        );

        // Add(5, 5) にマッチするはず
        let input1 = AstNode::Add(
            Box::new(AstNode::Const(Literal::I64(5))),
            Box::new(AstNode::Const(Literal::I64(5))),
        );
        let result1 = rule.apply(&input1);
        match result1 {
            AstNode::Mul(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::I64(5)));
                assert_eq!(*right, AstNode::Const(Literal::I64(2)));
            }
            _ => panic!("Expected Mul node for Add(5, 5)"),
        }

        // Add(5, 6) にはマッチしないはず（異なる値）
        let input2 = AstNode::Add(
            Box::new(AstNode::Const(Literal::I64(5))),
            Box::new(AstNode::Const(Literal::I64(6))),
        );
        let result2 = rule.apply(&input2);
        // マッチしないので元のまま
        assert_eq!(result2, input2);
    }

    #[test]
    fn test_rewriter_multiple_passes() {
        // 複数回の適用が必要なケース
        // ルール1: Add(a, 0) -> a
        let rule1 = AstRewriteRule::new(
            AstNode::Add(
                Box::new(AstNode::Wildcard("a".to_string())),
                Box::new(AstNode::Const(Literal::I64(0))),
            ),
            |bindings| bindings.get("a").unwrap().clone(),
            |_| true,
        );

        // ルール2: Mul(a, 1) -> a
        let rule2 = AstRewriteRule::new(
            AstNode::Mul(
                Box::new(AstNode::Wildcard("a".to_string())),
                Box::new(AstNode::Const(Literal::I64(1))),
            ),
            |bindings| bindings.get("a").unwrap().clone(),
            |_| true,
        );

        let rewriter = AstRewriter::new(vec![rule1, rule2]);

        // Mul(Add(x, 0), 1) -> Mul(x, 1) -> x
        let input = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::I64(42))),
                Box::new(AstNode::Const(Literal::I64(0))),
            )),
            Box::new(AstNode::Const(Literal::I64(1))),
        );

        let result = rewriter.apply(input);
        assert_eq!(result, AstNode::Const(Literal::I64(42)));
    }

    #[test]
    fn test_max_iterations() {
        // 無限ループを防ぐテスト
        // ルール: Add(a, b) -> Add(b, a) (交換するだけなので無限に繰り返す可能性)
        let rule = AstRewriteRule::new(
            AstNode::Add(
                Box::new(AstNode::Wildcard("a".to_string())),
                Box::new(AstNode::Wildcard("b".to_string())),
            ),
            |bindings| {
                let a = bindings.get("a").unwrap().clone();
                let b = bindings.get("b").unwrap().clone();
                AstNode::Add(Box::new(b), Box::new(a))
            },
            |_| true,
        );

        let rewriter = AstRewriter::new(vec![rule]).with_max_iterations(10);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::I64(1))),
            Box::new(AstNode::Const(Literal::I64(2))),
        );

        // 最大反復回数で停止するはず
        let _result = rewriter.apply(input);
        // パニックしなければOK
    }
}

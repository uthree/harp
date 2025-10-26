use crate::ast::AstNode;
use std::collections::HashMap;
use std::rc::Rc;

/// パターンマッチングの結果
pub type MatchResult = Option<HashMap<String, AstNode>>;

/// ASTの書き換えルール
pub struct AstRewriteRule {
    /// パターン（Wildcard含むAstNode）
    pattern: AstNode,
    /// 書き換え関数（マッチした変数を受け取り、新しいASTを返す）
    rewriter: Rc<dyn Fn(&HashMap<String, AstNode>) -> AstNode>,
    /// 条件関数（マッチした変数を受け取り、このルールを適用するか判定）
    condition: Rc<dyn Fn(&HashMap<String, AstNode>) -> bool>,
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
        let mut bindings = HashMap::new();
        if self.pattern_match(&self.pattern, ast, &mut bindings) {
            Some(bindings)
        } else {
            None
        }
    }

    /// パターンマッチングの内部実装
    fn pattern_match(
        &self,
        pattern: &AstNode,
        ast: &AstNode,
        bindings: &mut HashMap<String, AstNode>,
    ) -> bool {
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
            AstNode::Add(pl, pr) => {
                if let AstNode::Add(al, ar) = ast {
                    self.pattern_match(pl, al, bindings) && self.pattern_match(pr, ar, bindings)
                } else {
                    false
                }
            }
            AstNode::Mul(pl, pr) => {
                if let AstNode::Mul(al, ar) = ast {
                    self.pattern_match(pl, al, bindings) && self.pattern_match(pr, ar, bindings)
                } else {
                    false
                }
            }
            AstNode::Max(pl, pr) => {
                if let AstNode::Max(al, ar) = ast {
                    self.pattern_match(pl, al, bindings) && self.pattern_match(pr, ar, bindings)
                } else {
                    false
                }
            }
            AstNode::Rem(pl, pr) => {
                if let AstNode::Rem(al, ar) = ast {
                    self.pattern_match(pl, al, bindings) && self.pattern_match(pr, ar, bindings)
                } else {
                    false
                }
            }
            AstNode::Idiv(pl, pr) => {
                if let AstNode::Idiv(al, ar) = ast {
                    self.pattern_match(pl, al, bindings) && self.pattern_match(pr, ar, bindings)
                } else {
                    false
                }
            }
            AstNode::Recip(p) => {
                if let AstNode::Recip(a) = ast {
                    self.pattern_match(p, a, bindings)
                } else {
                    false
                }
            }
            AstNode::Sqrt(p) => {
                if let AstNode::Sqrt(a) = ast {
                    self.pattern_match(p, a, bindings)
                } else {
                    false
                }
            }
            AstNode::Log2(p) => {
                if let AstNode::Log2(a) = ast {
                    self.pattern_match(p, a, bindings)
                } else {
                    false
                }
            }
            AstNode::Exp2(p) => {
                if let AstNode::Exp2(a) = ast {
                    self.pattern_match(p, a, bindings)
                } else {
                    false
                }
            }
            AstNode::Sin(p) => {
                if let AstNode::Sin(a) = ast {
                    self.pattern_match(p, a, bindings)
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
            AstNode::Load | AstNode::Store => {
                // Load/Storeは具体的なノードとしてのみマッチ
                pattern == ast
            }
        }
    }

    /// ASTノードに対してルールを適用（再帰的に探索）
    pub fn apply(&self, ast: &AstNode) -> AstNode {
        // まず現在のノードにマッチを試みる
        if let Some(bindings) = self.try_match(ast) {
            if (self.condition)(&bindings) {
                return (self.rewriter)(&bindings);
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
        for _ in 0..self.max_iterations {
            let prev = format!("{:?}", ast);

            // すべてのルールを順番に適用
            for rule in &self.rules {
                ast = rule.apply(&ast);
            }

            // 変化がなければ終了
            if format!("{:?}", ast) == prev {
                break;
            }
        }
        ast
    }
}

/// パターンを構築するヘルパーマクロ
///
/// 使用例:
/// ```ignore
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
                AstNode::Mul(Box::new(x), Box::new(AstNode::Const(Literal::Isize(2))))
            },
            |_| true,
        );

        let input = AstNode::Const(Literal::Isize(5));
        let result = rule.apply(&input);

        // 5 -> 5 * 2
        match result {
            AstNode::Mul(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::Isize(5)));
                assert_eq!(*right, AstNode::Const(Literal::Isize(2)));
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
            Box::new(AstNode::Const(Literal::Isize(1))),
            Box::new(AstNode::Const(Literal::Isize(2))),
        );
        let result = rule.apply(&input);

        // Add(1, 2) -> Add(2, 1)
        match result {
            AstNode::Add(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::Isize(2)));
                assert_eq!(*right, AstNode::Const(Literal::Isize(1)));
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
            Box::new(AstNode::Const(Literal::Isize(3))),
            Box::new(AstNode::Const(Literal::Isize(4))),
        );
        let result = rule.apply(&input);

        match result {
            AstNode::Add(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::Isize(4)));
                assert_eq!(*right, AstNode::Const(Literal::Isize(3)));
            }
            _ => panic!("Expected Add node"),
        }
    }

    #[test]
    fn test_ast_rewriter_macro() {
        // 複数のルールを組み合わせたリライタ
        // Add(a, 0) -> a という簡約ルールを使う
        let rule1 = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Isize(0))))
        } => {
            a
        });

        let rewriter = ast_rewriter![rule1];

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(5))),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );
        let result = rewriter.apply(input);

        // Add(5, 0) -> 5
        assert_eq!(result, AstNode::Const(Literal::Isize(5)));
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
                AstNode::Mul(Box::new(x), Box::new(AstNode::Const(Literal::Isize(2))))
            },
            |_| true,
        );

        // Add(5, 5) にマッチするはず
        let input1 = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(5))),
            Box::new(AstNode::Const(Literal::Isize(5))),
        );
        let result1 = rule.apply(&input1);
        match result1 {
            AstNode::Mul(left, right) => {
                assert_eq!(*left, AstNode::Const(Literal::Isize(5)));
                assert_eq!(*right, AstNode::Const(Literal::Isize(2)));
            }
            _ => panic!("Expected Mul node for Add(5, 5)"),
        }

        // Add(5, 6) にはマッチしないはず（異なる値）
        let input2 = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(5))),
            Box::new(AstNode::Const(Literal::Isize(6))),
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
                Box::new(AstNode::Const(Literal::Isize(0))),
            ),
            |bindings| bindings.get("a").unwrap().clone(),
            |_| true,
        );

        // ルール2: Mul(a, 1) -> a
        let rule2 = AstRewriteRule::new(
            AstNode::Mul(
                Box::new(AstNode::Wildcard("a".to_string())),
                Box::new(AstNode::Const(Literal::Isize(1))),
            ),
            |bindings| bindings.get("a").unwrap().clone(),
            |_| true,
        );

        let rewriter = AstRewriter::new(vec![rule1, rule2]);

        // Mul(Add(x, 0), 1) -> Mul(x, 1) -> x
        let input = AstNode::Mul(
            Box::new(AstNode::Add(
                Box::new(AstNode::Const(Literal::Isize(42))),
                Box::new(AstNode::Const(Literal::Isize(0))),
            )),
            Box::new(AstNode::Const(Literal::Isize(1))),
        );

        let result = rewriter.apply(input);
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));
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
            Box::new(AstNode::Const(Literal::Isize(1))),
            Box::new(AstNode::Const(Literal::Isize(2))),
        );

        // 最大反復回数で停止するはず
        let _result = rewriter.apply(input);
        // パニックしなければOK
    }
}

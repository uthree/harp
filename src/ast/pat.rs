use crate::ast::{AstNode, AstOp, ConstValue};
use std::rc::Rc;

type RewriterFn = Box<dyn Fn(&[AstNode]) -> AstNode>;
type ConditionFn = Box<dyn Fn(&[AstNode]) -> bool>;

/// キャプチャノードを作成するヘルパー関数
pub fn capture(id: isize) -> AstNode {
    AstNode::new(AstOp::Capture(id))
}

pub struct AstRewriteRule {
    pattern: AstNode,
    rewriter: RewriterFn,
    condition: ConditionFn,
}

impl AstRewriteRule {
    pub fn new(pattern: AstNode, rewriter: RewriterFn, condition: ConditionFn) -> Rc<Self> {
        Rc::new(Self {
            pattern,
            rewriter,
            condition,
        })
    }

    /// パターンがASTにマッチするかチェックし、キャプチャした部分木を返す
    pub fn try_match(&self, node: &AstNode) -> Option<Vec<AstNode>> {
        let mut captures = Vec::new();
        if self.match_recursive(&self.pattern, node, &mut captures) {
            if (self.condition)(&captures) {
                Some(captures)
            } else {
                None
            }
        } else {
            None
        }
    }

    /// 再帰的にパターンマッチングを行う
    fn match_recursive(
        &self,
        pattern: &AstNode,
        target: &AstNode,
        captures: &mut Vec<AstNode>,
    ) -> bool {
        match &pattern.op {
            AstOp::Capture(id) => {
                // キャプチャIDに基づいて位置を決定
                let index = *id as usize;
                // capabilitiesを拡張して、必要なインデックスまで確保
                while captures.len() <= index {
                    captures.push(AstNode::new(AstOp::Const(ConstValue::Bool(false))));
                }
                captures[index] = target.clone();
                true
            }
            AstOp::Const(p_value) => {
                // 定数は値が一致する必要がある
                matches!(&target.op, AstOp::Const(t_value) if p_value == t_value)
            }
            AstOp::Var(p_name) => {
                // 変数は名前が一致する必要がある
                matches!(&target.op, AstOp::Var(t_name) if p_name == t_name)
            }
            AstOp::Cast(_, p_dtype) => {
                // Castの場合は型も一致する必要がある
                if let AstOp::Cast(_, t_dtype) = &target.op {
                    if p_dtype != t_dtype {
                        return false;
                    }
                    // 型が一致したら子ノードをチェック
                    self.match_children(pattern, target, captures)
                } else {
                    false
                }
            }
            _ => {
                // それ以外の場合、opの種類が一致し、全ての子ノードがマッチすればOK
                if std::mem::discriminant(&pattern.op) != std::mem::discriminant(&target.op) {
                    return false;
                }
                self.match_children(pattern, target, captures)
            }
        }
    }

    /// パターンとターゲットの子ノードを再帰的にマッチング
    fn match_children(
        &self,
        pattern: &AstNode,
        target: &AstNode,
        captures: &mut Vec<AstNode>,
    ) -> bool {
        let p_children = pattern.children();
        let t_children = target.children();

        if p_children.len() != t_children.len() {
            return false;
        }

        p_children
            .iter()
            .zip(t_children.iter())
            .all(|(p, t)| self.match_recursive(p, t, captures))
    }

    /// マッチしたノードを置き換える
    pub fn apply(&self, node: &AstNode) -> Option<AstNode> {
        self.try_match(node)
            .map(|captures| (self.rewriter)(&captures))
    }

    /// ASTを再帰的に走査して、マッチする部分を全て置き換える
    pub fn rewrite(&self, node: &AstNode) -> AstNode {
        // まず現在のノードに適用を試みる
        if let Some(rewritten) = self.apply(node) {
            return rewritten;
        }

        // マッチしなければ、子ノードを再帰的に処理
        let children = node.children();
        if children.is_empty() {
            // 子ノードがない場合はそのまま返す
            node.clone()
        } else {
            // 子ノードを再帰的に書き換え
            let new_children = children.iter().map(|child| self.rewrite(child)).collect();
            node.with_children(new_children)
        }
    }
}

/// 複数のリライトルールを管理し、順番に適用する
pub struct AstRewriter {
    rules: Vec<Rc<AstRewriteRule>>,
}

impl AstRewriter {
    /// 新しいAstRewriterを作成
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// ルールを追加
    pub fn add_rule(&mut self, rule: Rc<AstRewriteRule>) {
        self.rules.push(rule);
    }

    /// ルールのリストからAstRewriterを作成
    pub fn from_rules(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self { rules }
    }

    /// 全てのルールを順番に適用
    pub fn apply(&self, node: &AstNode) -> AstNode {
        let mut current = node.clone();
        for rule in &self.rules {
            current = rule.rewrite(&current);
        }
        current
    }

    /// 変化がなくなるまで繰り返し適用
    pub fn apply_until_fixed(&self, node: &AstNode) -> AstNode {
        let mut current = node.clone();
        loop {
            let next = self.apply(&current);
            // AstNodeを比較するために、Debug表現を使用（簡易的）
            if format!("{:?}", current) == format!("{:?}", next) {
                return current;
            }
            current = next;
        }
    }

    /// 他のリライターと融合して、新しいリライターを作成
    /// 融合後のリライターは、self のルール → other のルールの順で適用
    pub fn merge(&self, other: &AstRewriter) -> AstRewriter {
        let mut rules = self.rules.clone();
        rules.extend(other.rules.clone());
        AstRewriter { rules }
    }
}

impl Default for AstRewriter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Add for AstRewriter {
    type Output = AstRewriter;

    fn add(self, other: AstRewriter) -> AstRewriter {
        self.merge(&other)
    }
}

impl std::ops::Add for &AstRewriter {
    type Output = AstRewriter;

    fn add(self, other: &AstRewriter) -> AstRewriter {
        self.merge(other)
    }
}

impl std::ops::Add<&AstRewriter> for AstRewriter {
    type Output = AstRewriter;

    fn add(self, other: &AstRewriter) -> AstRewriter {
        self.merge(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{add, const_f32, mul, neg, DType};

    #[test]
    fn test_simple_pattern_match() {
        // パターン: Capture(0) + Capture(1)
        let pattern = add(capture(0), capture(1));

        // ターゲット: 1.0 + 2.0
        let target = add(const_f32(1.0), const_f32(2.0));

        let rule = AstRewriteRule::new(
            pattern,
            Box::new(|_captures| const_f32(0.0)),
            Box::new(|_| true),
        );

        let result = rule.try_match(&target);
        assert!(result.is_some());

        let captures = result.unwrap();
        assert_eq!(captures.len(), 2);
    }

    #[test]
    fn test_nested_pattern_match() {
        // パターン: Neg(Capture(0))
        let pattern = neg(capture(0));

        // ターゲット: -(1.0 + 2.0)
        let inner = add(const_f32(1.0), const_f32(2.0));
        let target = neg(inner.clone());

        let rule = AstRewriteRule::new(
            pattern,
            Box::new(|_captures| const_f32(0.0)),
            Box::new(|_| true),
        );

        let result = rule.try_match(&target);
        assert!(result.is_some());

        let captures = result.unwrap();
        assert_eq!(captures.len(), 1);
        // キャプチャした部分が正しいかチェック
        if let AstOp::Add(_, _) = &captures[0].op {
            // 成功
        } else {
            panic!("Expected Add operation");
        }
    }

    #[test]
    fn test_rewrite_double_negation() {
        // パターン: -(-x) を x に置き換える
        let pattern = neg(neg(capture(0)));

        let rule = AstRewriteRule::new(
            pattern,
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        // ターゲット: -(-5.0)
        let target = neg(neg(const_f32(5.0)));

        let result = rule.rewrite(&target);

        // 結果は 5.0 のみになるべき
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 5.0);
        } else {
            panic!("Expected constant 5.0");
        }
    }

    #[test]
    fn test_rewrite_nested() {
        // パターン: x * 1.0 を x に置き換える
        let pattern = AstNode {
            op: AstOp::Mul(
                Box::new(capture(0).with_dtype(DType::F32)),
                Box::new(const_f32(1.0)),
            ),
            dtype: DType::F32,
        };

        let rule = AstRewriteRule::new(
            pattern,
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        // ターゲット: (2.0 + 3.0) * 1.0
        let inner = add(const_f32(2.0), const_f32(3.0));
        let target = mul(inner, const_f32(1.0));

        let result = rule.rewrite(&target);

        // 結果は 2.0 + 3.0 になるべき
        if let AstOp::Add(_, _) = result.op {
            // 成功
        } else {
            panic!("Expected Add operation");
        }
    }

    #[test]
    fn test_condition_check() {
        // パターン: Capture(0) + Capture(1) で、両方が定数の場合のみマッチ
        let pattern = add(capture(0), capture(1));

        let rule = AstRewriteRule::new(
            pattern,
            Box::new(|captures| {
                // 両方の値を取り出して足し合わせる
                if let (AstOp::Const(ConstValue::F32(a)), AstOp::Const(ConstValue::F32(b))) =
                    (&captures[0].op, &captures[1].op)
                {
                    const_f32(a + b)
                } else {
                    captures[0].clone()
                }
            }),
            Box::new(|captures| {
                // 両方が定数の場合のみマッチ
                matches!(captures[0].op, AstOp::Const(_))
                    && matches!(captures[1].op, AstOp::Const(_))
            }),
        );

        // ターゲット1: 1.0 + 2.0 (両方定数)
        let target1 = add(const_f32(1.0), const_f32(2.0));
        let result1 = rule.try_match(&target1);
        assert!(result1.is_some(), "Should match two constants");

        // ターゲット2: x + 2.0 (片方が変数)
        let target2 = add(
            AstNode::new(AstOp::Var("x".to_string())).with_dtype(DType::F32),
            const_f32(2.0),
        );
        let result2 = rule.try_match(&target2);
        assert!(result2.is_none(), "Should not match when one is a variable");
    }

    #[test]
    fn test_constant_folding() {
        // 定数畳み込みのテスト: 2.0 + 3.0 を 5.0 に置き換える
        let pattern = add(capture(0), capture(1));

        let rule = AstRewriteRule::new(
            pattern,
            Box::new(|captures| {
                if let (AstOp::Const(ConstValue::F32(a)), AstOp::Const(ConstValue::F32(b))) =
                    (&captures[0].op, &captures[1].op)
                {
                    const_f32(a + b)
                } else {
                    add(captures[0].clone(), captures[1].clone())
                }
            }),
            Box::new(|captures| {
                matches!(captures[0].op, AstOp::Const(_))
                    && matches!(captures[1].op, AstOp::Const(_))
            }),
        );

        let target = add(const_f32(2.0), const_f32(3.0));
        let result = rule.rewrite(&target);

        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 5.0);
        } else {
            panic!("Expected constant 5.0");
        }
    }

    #[test]
    fn test_ast_rewriter_single_rule() {
        // 単一のルールを適用
        let rule = AstRewriteRule::new(
            neg(neg(capture(0))),
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let mut rewriter = AstRewriter::new();
        rewriter.add_rule(rule);

        let target = neg(neg(const_f32(5.0)));
        let result = rewriter.apply(&target);

        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 5.0);
        } else {
            panic!("Expected constant 5.0");
        }
    }

    #[test]
    fn test_ast_rewriter_multiple_rules() {
        // 複数のルールを順番に適用
        // ルール1: -(-x) -> x
        let rule1 = AstRewriteRule::new(
            neg(neg(capture(0))),
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        // ルール2: x * 1.0 -> x
        let rule2 = AstRewriteRule::new(
            AstNode {
                op: AstOp::Mul(
                    Box::new(capture(0).with_dtype(DType::F32)),
                    Box::new(const_f32(1.0)),
                ),
                dtype: DType::F32,
            },
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let rewriter = AstRewriter::from_rules(vec![rule1, rule2]);

        // ターゲット: -(-5.0) * 1.0
        let target = mul(neg(neg(const_f32(5.0))), const_f32(1.0));
        let result = rewriter.apply(&target);

        // 結果は 5.0 になるべき
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 5.0);
        } else {
            panic!("Expected constant 5.0");
        }
    }

    #[test]
    fn test_ast_rewriter_apply_until_fixed() {
        // 定数畳み込みルール: a + b -> (a+b)
        let add_rule = AstRewriteRule::new(
            add(capture(0), capture(1)),
            Box::new(|captures| {
                if let (AstOp::Const(ConstValue::F32(a)), AstOp::Const(ConstValue::F32(b))) =
                    (&captures[0].op, &captures[1].op)
                {
                    const_f32(a + b)
                } else {
                    add(captures[0].clone(), captures[1].clone())
                }
            }),
            Box::new(|captures| {
                matches!(captures[0].op, AstOp::Const(_))
                    && matches!(captures[1].op, AstOp::Const(_))
            }),
        );

        // 乗算の定数畳み込みルール: a * b -> (a*b)
        let mul_rule = AstRewriteRule::new(
            mul(capture(0), capture(1)),
            Box::new(|captures| {
                if let (AstOp::Const(ConstValue::F32(a)), AstOp::Const(ConstValue::F32(b))) =
                    (&captures[0].op, &captures[1].op)
                {
                    const_f32(a * b)
                } else {
                    mul(captures[0].clone(), captures[1].clone())
                }
            }),
            Box::new(|captures| {
                matches!(captures[0].op, AstOp::Const(_))
                    && matches!(captures[1].op, AstOp::Const(_))
            }),
        );

        let rewriter = AstRewriter::from_rules(vec![add_rule, mul_rule]);

        // ターゲット: (1.0 + 2.0) * (3.0 + 4.0)
        let target = mul(
            add(const_f32(1.0), const_f32(2.0)),
            add(const_f32(3.0), const_f32(4.0)),
        );

        let result = rewriter.apply_until_fixed(&target);

        // 結果は (1.0 + 2.0) * (3.0 + 4.0) = 3.0 * 7.0 = 21.0
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 21.0);
        } else {
            panic!("Expected constant 21.0");
        }
    }

    #[test]
    fn test_ast_rewriter_merge() {
        // 2つのリライターをmergeで融合
        let rule1 = AstRewriteRule::new(
            neg(neg(capture(0))),
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let rule2 = AstRewriteRule::new(
            AstNode {
                op: AstOp::Mul(
                    Box::new(capture(0).with_dtype(DType::F32)),
                    Box::new(const_f32(1.0)),
                ),
                dtype: DType::F32,
            },
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let rewriter1 = AstRewriter::from_rules(vec![rule1]);
        let rewriter2 = AstRewriter::from_rules(vec![rule2]);

        let merged = rewriter1.merge(&rewriter2);

        // ターゲット: -(-5.0) * 1.0
        let target = mul(neg(neg(const_f32(5.0))), const_f32(1.0));
        let result = merged.apply(&target);

        // 結果は 5.0 になるべき
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 5.0);
        } else {
            panic!("Expected constant 5.0");
        }
    }

    #[test]
    fn test_ast_rewriter_add_operator() {
        // +演算子でリライターを融合
        let rule1 = AstRewriteRule::new(
            neg(neg(capture(0))),
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let rule2 = AstRewriteRule::new(
            add(capture(0), capture(1)),
            Box::new(|captures| {
                if let (AstOp::Const(ConstValue::F32(a)), AstOp::Const(ConstValue::F32(b))) =
                    (&captures[0].op, &captures[1].op)
                {
                    const_f32(a + b)
                } else {
                    add(captures[0].clone(), captures[1].clone())
                }
            }),
            Box::new(|captures| {
                matches!(captures[0].op, AstOp::Const(_))
                    && matches!(captures[1].op, AstOp::Const(_))
            }),
        );

        let rewriter1 = AstRewriter::from_rules(vec![rule1]);
        let rewriter2 = AstRewriter::from_rules(vec![rule2]);

        // +演算子で融合
        let merged = rewriter1 + rewriter2;

        // ターゲット: -(-1.0) + -(-2.0)
        let target = add(neg(neg(const_f32(1.0))), neg(neg(const_f32(2.0))));
        let result = merged.apply(&target);

        // 結果は 1.0 + 2.0 = 3.0 になるべき
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 3.0);
        } else {
            panic!("Expected constant 3.0");
        }
    }

    #[test]
    fn test_ast_rewriter_add_operator_ref() {
        // +演算子（参照版）でリライターを融合
        let rule1 = AstRewriteRule::new(
            neg(neg(capture(0))),
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let rule2 = AstRewriteRule::new(
            AstNode {
                op: AstOp::Mul(
                    Box::new(capture(0).with_dtype(DType::F32)),
                    Box::new(const_f32(1.0)),
                ),
                dtype: DType::F32,
            },
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let rewriter1 = AstRewriter::from_rules(vec![rule1]);
        let rewriter2 = AstRewriter::from_rules(vec![rule2]);

        // 参照で+演算子を使用（元のリライターは保持される）
        let merged = &rewriter1 + &rewriter2;

        // ターゲット: -(-5.0) * 1.0
        let target = mul(neg(neg(const_f32(5.0))), const_f32(1.0));
        let result = merged.apply(&target);

        // 結果は 5.0 になるべき
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 5.0);
        } else {
            panic!("Expected constant 5.0");
        }

        // 元のリライターも使用可能
        let target2 = neg(neg(const_f32(3.0)));
        let result2 = rewriter1.apply(&target2);
        if let AstOp::Const(ConstValue::F32(val)) = result2.op {
            assert_eq!(val, 3.0);
        } else {
            panic!("Expected constant 3.0");
        }
    }

    #[test]
    fn test_ast_rewriter_chain_merge() {
        // 複数のリライターを連鎖的に融合
        let rule1 = AstRewriteRule::new(
            neg(neg(capture(0))),
            Box::new(|captures| captures[0].clone()),
            Box::new(|_| true),
        );

        let rule2 = AstRewriteRule::new(
            add(capture(0), capture(1)),
            Box::new(|captures| {
                if let (AstOp::Const(ConstValue::F32(a)), AstOp::Const(ConstValue::F32(b))) =
                    (&captures[0].op, &captures[1].op)
                {
                    const_f32(a + b)
                } else {
                    add(captures[0].clone(), captures[1].clone())
                }
            }),
            Box::new(|captures| {
                matches!(captures[0].op, AstOp::Const(_))
                    && matches!(captures[1].op, AstOp::Const(_))
            }),
        );

        let rule3 = AstRewriteRule::new(
            mul(capture(0), capture(1)),
            Box::new(|captures| {
                if let (AstOp::Const(ConstValue::F32(a)), AstOp::Const(ConstValue::F32(b))) =
                    (&captures[0].op, &captures[1].op)
                {
                    const_f32(a * b)
                } else {
                    mul(captures[0].clone(), captures[1].clone())
                }
            }),
            Box::new(|captures| {
                matches!(captures[0].op, AstOp::Const(_))
                    && matches!(captures[1].op, AstOp::Const(_))
            }),
        );

        let rewriter1 = AstRewriter::from_rules(vec![rule1]);
        let rewriter2 = AstRewriter::from_rules(vec![rule2]);
        let rewriter3 = AstRewriter::from_rules(vec![rule3]);

        // 3つのリライターを連鎖的に融合
        let merged = &rewriter1 + &rewriter2 + &rewriter3;

        // ターゲット: (-(-2.0) + -(-3.0)) * -(-4.0)
        let target = mul(
            add(neg(neg(const_f32(2.0))), neg(neg(const_f32(3.0)))),
            neg(neg(const_f32(4.0))),
        );

        let result = merged.apply_until_fixed(&target);

        // 結果は (2.0 + 3.0) * 4.0 = 5.0 * 4.0 = 20.0
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 20.0);
        } else {
            panic!("Expected constant 20.0");
        }
    }
}

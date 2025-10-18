use crate::ast::{AstNode, AstRewriter};
use crate::opt::AstOptimizer;

/// AstRewriterを使った最適化器
pub struct RewriterOptimizer {
    rewriter: AstRewriter,
    /// 変化がなくなるまで繰り返し適用するかどうか
    apply_until_fixed: bool,
}

impl RewriterOptimizer {
    /// 新しいRewriterOptimizerを作成
    /// デフォルトでは1回だけ適用
    pub fn new(rewriter: AstRewriter) -> Self {
        Self {
            rewriter,
            apply_until_fixed: false,
        }
    }

    /// 変化がなくなるまで繰り返し適用するモードを有効化
    pub fn with_fixed_point(mut self) -> Self {
        self.apply_until_fixed = true;
        self
    }

    /// 複数のリライターを統合してOptimizerを作成
    pub fn from_rewriters(rewriters: Vec<AstRewriter>) -> Self {
        let combined = rewriters
            .into_iter()
            .fold(AstRewriter::new(), |acc, r| acc + r);
        Self::new(combined)
    }
}

impl AstOptimizer for RewriterOptimizer {
    fn apply(&self, ast: &AstNode) -> AstNode {
        if self.apply_until_fixed {
            self.rewriter.apply_until_fixed(ast)
        } else {
            self.rewriter.apply(ast)
        }
    }
}

/// 複数のOptimizerを順番に適用する合成Optimizer
pub struct ComposedOptimizer {
    optimizers: Vec<Box<dyn AstOptimizer>>,
}

impl ComposedOptimizer {
    /// 新しいComposedOptimizerを作成
    pub fn new() -> Self {
        Self {
            optimizers: Vec::new(),
        }
    }

    /// Optimizerを追加
    pub fn add_optimizer<O: AstOptimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizers.push(Box::new(optimizer));
        self
    }

    /// 複数のOptimizerから作成
    pub fn from_optimizers(optimizers: Vec<Box<dyn AstOptimizer>>) -> Self {
        Self { optimizers }
    }
}

impl Default for ComposedOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AstOptimizer for ComposedOptimizer {
    fn apply(&self, ast: &AstNode) -> AstNode {
        let mut current = ast.clone();
        for optimizer in &self.optimizers {
            current = optimizer.apply(&current);
        }
        current
    }

    /// ComposedOptimizerの場合、既存のoptimizer群に新しいoptimizerを追加
    fn compose(mut self, other: impl AstOptimizer + 'static) -> ComposedOptimizer
    where
        Self: Sized + 'static,
    {
        self.optimizers.push(Box::new(other));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        add, ast_rewriter, ast_rule, capture, const_f32, mul, neg, AstOp, ConstValue,
    };

    #[test]
    fn test_rewriter_optimizer_basic() {
        let rewriter = ast_rewriter! {
            ast_rule!(|x| neg(neg(capture(0))) => x.clone()),
        };
        let optimizer = RewriterOptimizer::new(rewriter);
        let ast = neg(neg(const_f32(5.0)));
        let result = optimizer.apply(&ast);
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 5.0);
        } else {
            panic!("Expected constant 5.0");
        }
    }

    #[test]
    fn test_rewriter_optimizer_with_fixed_point() {
        let rewriter = ast_rewriter! {
            ast_rule!(
                |a, b| add(capture(0), capture(1)) => {
                    if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                        (&a.op, &b.op)
                    {
                        const_f32(av + bv)
                    } else {
                        add(a.clone(), b.clone())
                    }
                },
                if |caps: &[AstNode]| {
                    matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
                }
            ),
            ast_rule!(
                |a, b| mul(capture(0), capture(1)) => {
                    if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                        (&a.op, &b.op)
                    {
                        const_f32(av * bv)
                    } else {
                        mul(a.clone(), b.clone())
                    }
                },
                if |caps: &[AstNode]| {
                    matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
                }
            ),
        };
        let optimizer = RewriterOptimizer::new(rewriter).with_fixed_point();
        let ast = mul(
            add(const_f32(1.0), const_f32(2.0)),
            add(const_f32(3.0), const_f32(4.0)),
        );
        let result = optimizer.apply(&ast);
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 21.0);
        } else {
            panic!("Expected constant 21.0");
        }
    }

    #[test]
    fn test_composed_optimizer() {
        let rewriter1 = ast_rewriter! {
            ast_rule!(|x| neg(neg(capture(0))) => x.clone()),
        };
        let opt1 = RewriterOptimizer::new(rewriter1);
        let rewriter2 = ast_rewriter! {
            ast_rule!(
                |a, b| add(capture(0), capture(1)) => {
                    if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                        (&a.op, &b.op)
                    {
                        const_f32(av + bv)
                    } else {
                        add(a.clone(), b.clone())
                    }
                },
                if |caps: &[AstNode]| {
                    matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
                }
            ),
        };
        let opt2 = RewriterOptimizer::new(rewriter2).with_fixed_point();
        let composed = ComposedOptimizer::new()
            .add_optimizer(opt1)
            .add_optimizer(opt2);
        let ast = add(neg(neg(const_f32(1.0))), neg(neg(const_f32(2.0))));
        let result = composed.apply(&ast);
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 3.0);
        } else {
            panic!("Expected constant 3.0");
        }
    }

    #[test]
    fn test_compose_method_chaining() {
        let opt1 = RewriterOptimizer::new(ast_rewriter! {
            ast_rule!(|x| neg(neg(capture(0))) => x.clone()),
        });
        let opt2 = RewriterOptimizer::new(ast_rewriter! {
            ast_rule!(
                |a, b| add(capture(0), capture(1)) => {
                    if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                        (&a.op, &b.op)
                    {
                        const_f32(av + bv)
                    } else {
                        add(a.clone(), b.clone())
                    }
                },
                if |caps: &[AstNode]| {
                    matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
                }
            ),
        });
        let opt3 = RewriterOptimizer::new(ast_rewriter! {
            ast_rule!(
                |a, b| mul(capture(0), capture(1)) => {
                    if let (AstOp::Const(ConstValue::F32(av)), AstOp::Const(ConstValue::F32(bv))) =
                        (&a.op, &b.op)
                    {
                        const_f32(av * bv)
                    } else {
                        mul(a.clone(), b.clone())
                    }
                },
                if |caps: &[AstNode]| {
                    matches!(caps[0].op, AstOp::Const(_)) && matches!(caps[1].op, AstOp::Const(_))
                }
            ),
        });
        let composed = opt1.compose(opt2).compose(opt3);
        let ast = mul(
            add(neg(neg(const_f32(2.0))), neg(neg(const_f32(3.0)))),
            neg(neg(const_f32(4.0))),
        );
        let result = composed.apply(&ast);
        if let AstOp::Const(ConstValue::F32(val)) = result.op {
            assert_eq!(val, 20.0);
        } else {
            panic!("Expected constant 20.0, got {:?}", result);
        }
    }
}

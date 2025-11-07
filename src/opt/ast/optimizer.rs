use crate::ast::AstNode;
use crate::ast::pat::{AstRewriteRule, AstRewriter};
use indicatif::{ProgressBar, ProgressStyle};
use log::debug;
use std::rc::Rc;

use super::estimator::CostEstimator;
use super::suggester::Suggester;

/// ASTを最適化するトレイト
pub trait Optimizer {
    /// ASTを最適化して返す
    fn optimize(&self, ast: AstNode) -> AstNode;
}

/// ルールベースの最適化器
pub struct RuleBaseOptimizer {
    rewriter: AstRewriter,
}

impl RuleBaseOptimizer {
    /// 新しい最適化器を作成
    pub fn new(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self {
            rewriter: AstRewriter::new(rules),
        }
    }

    /// 最大反復回数を設定
    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.rewriter = self.rewriter.with_max_iterations(max);
        self
    }
}

impl Optimizer for RuleBaseOptimizer {
    fn optimize(&self, ast: AstNode) -> AstNode {
        debug!("RuleBaseOptimizer: Starting optimization");
        let result = self.rewriter.apply(ast);
        debug!("RuleBaseOptimizer: Optimization complete");
        result
    }
}

/// ビームサーチ最適化器
pub struct BeamSearchOptimizer<S, E>
where
    S: Suggester,
    E: CostEstimator,
{
    suggester: S,
    estimator: E,
    beam_width: usize,
    max_depth: usize,
    show_progress: bool,
}

impl<S, E> BeamSearchOptimizer<S, E>
where
    S: Suggester,
    E: CostEstimator,
{
    /// 新しいビームサーチ最適化器を作成
    pub fn new(suggester: S, estimator: E) -> Self {
        Self {
            suggester,
            estimator,
            beam_width: 10,
            max_depth: 10,
            show_progress: true,
        }
    }

    /// ビーム幅を設定
    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    /// 最大深さを設定
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// プログレスバーの表示/非表示を設定
    pub fn with_progress(mut self, show: bool) -> Self {
        self.show_progress = show;
        self
    }
}

impl<S, E> Optimizer for BeamSearchOptimizer<S, E>
where
    S: Suggester,
    E: CostEstimator,
{
    fn optimize(&self, ast: AstNode) -> AstNode {
        debug!("BeamSearchOptimizer: Starting beam search optimization");

        let mut beam = vec![ast];

        let pb = if self.show_progress {
            let pb = ProgressBar::new(self.max_depth as u64);

            // Cargoスタイルのプログレスバー
            pb.set_style(
                ProgressStyle::with_template("{prefix:>12.cyan.bold} [{bar:24}] {pos}/{len} {msg}")
                    .unwrap()
                    .progress_chars("=> "),
            );
            pb.set_prefix("Optimizing");
            Some(pb)
        } else {
            None
        };

        for depth in 0..self.max_depth {
            if let Some(ref pb) = pb {
                pb.set_message(format!("depth {}", depth + 1));
                pb.set_position(depth as u64);
            }

            let mut candidates = Vec::new();

            // 現在のビーム内の各候補から新しい候補を生成
            for ast in &beam {
                let new_candidates = self.suggester.suggest(ast);
                candidates.extend(new_candidates);
            }

            if candidates.is_empty() {
                debug!("BeamSearchOptimizer: No more candidates at depth {}", depth);
                if let Some(ref pb) = pb {
                    pb.set_position(self.max_depth as u64);
                }
                break;
            }

            debug!(
                "BeamSearchOptimizer: Found {} candidates at depth {}",
                candidates.len(),
                depth
            );

            // コストでソートして上位beam_width個を残す
            candidates.sort_by(|a, b| {
                self.estimator
                    .estimate(a)
                    .partial_cmp(&self.estimator.estimate(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            beam = candidates.into_iter().take(self.beam_width).collect();
        }

        if let Some(pb) = pb {
            pb.finish_with_message("Complete");
        }

        debug!("BeamSearchOptimizer: Beam search optimization complete");

        // 最良の候補を返す
        beam.into_iter()
            .min_by(|a, b| {
                self.estimator
                    .estimate(a)
                    .partial_cmp(&self.estimator.estimate(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;
    use crate::astpat;
    use crate::opt::ast::estimator::SimpleCostEstimator;
    use crate::opt::ast::suggester::RuleBaseSuggester;

    #[test]
    fn test_rule_base_optimizer() {
        // Add(a, 0) -> a というルール
        let rule = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Isize(0))))
        } => {
            a
        });

        let optimizer = RuleBaseOptimizer::new(vec![rule]);

        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(42))),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );

        let result = optimizer.optimize(input);
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));
    }

    #[test]
    fn test_beam_search_optimizer() {
        // 交換則と単位元除去のルール
        let rule1 = astpat!(|a, b| {
            AstNode::Add(Box::new(a), Box::new(b))
        } => {
            AstNode::Add(Box::new(b), Box::new(a))
        });

        let rule2 = astpat!(|a| {
            AstNode::Add(Box::new(a), Box::new(AstNode::Const(Literal::Isize(0))))
        } => {
            a
        });

        let suggester = RuleBaseSuggester::new(vec![rule1, rule2]);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(5)
            .with_max_depth(5)
            .with_progress(false); // テスト中はプログレスバーを非表示

        // (42 + 0) を最適化
        let input = AstNode::Add(
            Box::new(AstNode::Const(Literal::Isize(42))),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );

        let result = optimizer.optimize(input);
        // 最終的に42に簡約されるはず
        assert_eq!(result, AstNode::Const(Literal::Isize(42)));
    }

    #[test]
    fn test_beam_search_optimizer_complex() {
        use crate::opt::ast::rules::{add_commutative, all_algebraic_rules};

        let mut rules = all_algebraic_rules();
        rules.push(add_commutative());

        let suggester = RuleBaseSuggester::new(rules);
        let estimator = SimpleCostEstimator::new();

        let optimizer = BeamSearchOptimizer::new(suggester, estimator)
            .with_beam_width(10)
            .with_max_depth(10)
            .with_progress(false);

        // ((2 + 3) * 1) + 0 を最適化
        let input = AstNode::Add(
            Box::new(AstNode::Mul(
                Box::new(AstNode::Add(
                    Box::new(AstNode::Const(Literal::Isize(2))),
                    Box::new(AstNode::Const(Literal::Isize(3))),
                )),
                Box::new(AstNode::Const(Literal::Isize(1))),
            )),
            Box::new(AstNode::Const(Literal::Isize(0))),
        );

        let result = optimizer.optimize(input);
        // 最終的に5に簡約されるはず
        assert_eq!(result, AstNode::Const(Literal::Isize(5)));
    }
}

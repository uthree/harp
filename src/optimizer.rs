use crate::autotuner::{Configuration, OptimizationRule};
use crate::pats;
use crate::pattern::{PatternMatcher, UPat};
use crate::uop::{Op, UOp};
use log::debug;

/// Applies a set of pattern-based optimizations to a `UOp` graph.
pub struct Optimizer {
    matcher: PatternMatcher,
}

impl Optimizer {
    /// Creates a new `Optimizer` with a set of baseline rules that are always beneficial.
    pub fn new_baseline() -> Self {
        let rules: Vec<UPat> = pats!({
            // --- Algebraic Simplification (f32) ---
            (a) | &a + &UOp::from(0.0f32) => a,
            (a) | &a * &UOp::from(1.0f32) => a,
            (a) | &a * &UOp::from(0.0f32) => UOp::from(0.0f32),

            // --- Algebraic Simplification (i32) ---
            (a) | &a + &UOp::from(0i32) => a,
            (a) | &a * &UOp::from(1i32) => a,
            (a) | &a * &UOp::from(0i32) => UOp::from(0i32),
        })
        .into_iter()
        .collect();

        Self {
            matcher: PatternMatcher::new(rules),
        }
    }

    /// Creates a new `Optimizer` for tunable rules based on a given configuration.
    ///
    /// # Arguments
    /// * `config` - The configuration specifying which tunable optimization rules to enable.
    pub fn new_for_tuning(config: &Configuration) -> Self {
        let mut rules: Vec<UPat> = Vec::new();
        let enabled_rules = &config.enabled_rules;

        if enabled_rules.contains(&OptimizationRule::RecipRecip) {
            rules.extend(pats!({
                (a) | a.recip().recip() => a,
            }));
        }

        Self {
            matcher: PatternMatcher::new(rules),
        }
    }

    /// Applies optimization rules to the `UOp` graph until a fixed point is reached.
    pub fn optimize(&self, uop: &UOp) -> UOp {
        if self.matcher.is_empty() {
            return uop.clone();
        }
        debug!("Before optimization: {uop:?}");
        // The limit is a safeguard against potential infinite loops in the rules.
        let optimized_uop = self.matcher.apply_all_with_limit(uop, 100);
        debug!("After optimization: {optimized_uop:?}");
        optimized_uop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uop::DType;
    use rustc_hash::FxHashSet;

    #[test]
    fn test_baseline_optimizations() {
        let optimizer = Optimizer::new_baseline();
        let a = UOp::var("a", DType::F32);

        // Test x + 0 => x
        let expr1 = &a + &UOp::from(0.0f32);
        assert_eq!(optimizer.optimize(&expr1), a);

        // Test x * 1 => x
        let expr2 = &a * &UOp::from(1.0f32);
        assert_eq!(optimizer.optimize(&expr2), a);

        // Test x * 0 => 0
        let expr3 = &a * &UOp::from(0.0f32);
        assert_eq!(optimizer.optimize(&expr3), UOp::from(0.0f32));
    }

    #[test]
    fn test_chained_baseline_optimizations() {
        let optimizer = Optimizer::new_baseline();
        let a = UOp::var("a", DType::F32);

        // Test (a * 1) + 0 => a
        let expr = (&a * &UOp::from(1.0f32)) + &UOp::from(0.0f32);
        assert_eq!(optimizer.optimize(&expr), a);
    }

    #[test]
    fn test_tunable_optimization_rule() {
        let a = UOp::var("a", DType::F32);
        let expr = a.recip().recip();

        // Test with the rule enabled
        let mut enabled_rules = FxHashSet::default();
        enabled_rules.insert(OptimizationRule::RecipRecip);
        let config_enabled = Configuration {
            enabled_rules,
            ..Default::default()
        };
        let optimizer_enabled = Optimizer::new_for_tuning(&config_enabled);
        assert_eq!(
            optimizer_enabled.optimize(&expr),
            a,
            "Should optimize with rule enabled"
        );

        // Test with the rule disabled
        let config_disabled = Configuration::default(); // Default has no rules enabled
        let optimizer_disabled = Optimizer::new_for_tuning(&config_disabled);
        assert_eq!(
            optimizer_disabled.optimize(&expr),
            expr,
            "Should not optimize with rule disabled"
        );
    }
}

use crate::autotuner::{Configuration, OptimizationRule};
use crate::pats;
use crate::pattern::{PatternMatcher, UPat};
use crate::uop::{Op, UOp};
use log::debug;

/// Applies a set of pattern-based optimizations to a `UOp` graph.
///
/// The `Optimizer` uses a `PatternMatcher` to repeatedly apply a list of
/// algebraic simplification rules to a `UOp` graph until a fixed point
/// is reached.
pub struct Optimizer {
    matcher: PatternMatcher,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new(&Configuration::default())
    }
}

impl Optimizer {
    /// Creates a new `Optimizer` with a given configuration.
    ///
    /// # Arguments
    /// * `config` - The configuration specifying which optimization rules to enable.
    pub fn new(config: &Configuration) -> Self {
        let mut rules: Vec<UPat> = Vec::new();
        let enabled_rules = &config.enabled_rules;

        if enabled_rules.contains(&OptimizationRule::AddZero) {
            rules.extend(pats!({
                (a) | &a + &UOp::from(0.0f32) => a,
                (a) | &a + &UOp::from(0i32) => a,
            }));
        }
        if enabled_rules.contains(&OptimizationRule::MulOne) {
            rules.extend(pats!({
                (a) | &a * &UOp::from(1.0f32) => a,
                (a) | &a * &UOp::from(1i32) => a,
            }));
        }
        if enabled_rules.contains(&OptimizationRule::MulZero) {
            rules.extend(pats!({
                (a) | &a * &UOp::from(0.0f32) => UOp::from(0.0f32),
                (a) | &a * &UOp::from(0i32) => UOp::from(0i32),
            }));
        }
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
    ///
    /// # Arguments
    /// * `uop` - The root of the `UOp` graph to optimize.
    ///
    /// # Returns
    /// The optimized `UOp` graph.
    pub fn optimize(&self, uop: &UOp) -> UOp {
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

    fn default_optimizer() -> Optimizer {
        Optimizer::new(&Configuration::default())
    }

    #[test]
    fn test_algebraic_simplification() {
        let optimizer = default_optimizer();
        let a = UOp::var("a", DType::F32);

        // Test x + 0 => x
        let expr1 = &a + &UOp::from(0.0f32);
        let optimized1 = optimizer.optimize(&expr1);
        assert_eq!(optimized1, a);

        // Test x * 1 => x
        let expr2 = &a * &UOp::from(1.0f32);
        let optimized2 = optimizer.optimize(&expr2);
        assert_eq!(optimized2, a);

        // Test x * 0 => 0
        let expr3 = &a * &UOp::from(0.0f32);
        let optimized3 = optimizer.optimize(&expr3);
        assert_eq!(optimized3, UOp::from(0.0f32));
    }

    #[test]
    fn test_chained_optimizations() {
        let optimizer = default_optimizer();
        let a = UOp::var("a", DType::F32);

        // Test (a * 1) + 0 => a
        let expr = (&a * &UOp::from(1.0f32)) + &UOp::from(0.0f32);
        let optimized = optimizer.optimize(&expr);
        assert_eq!(optimized, a);
    }

    #[test]
    fn test_optimization_with_disabled_rule() {
        let a = UOp::var("a", DType::F32);
        let expr = &a + &UOp::from(0.0f32);

        // First, test with the rule enabled (default)
        let default_config = Configuration::default();
        let optimizer_enabled = Optimizer::new(&default_config);
        let optimized_enabled = optimizer_enabled.optimize(&expr);
        assert_eq!(optimized_enabled, a, "Should optimize with rule enabled");

        // Then, test with the rule disabled
        let mut disabled_rules = FxHashSet::default();
        disabled_rules.remove(&OptimizationRule::AddZero);
        let mut config_disabled = Configuration::default();
        config_disabled.enabled_rules = disabled_rules;

        let optimizer_disabled = Optimizer::new(&config_disabled);
        let optimized_disabled = optimizer_disabled.optimize(&expr);
        assert_eq!(
            optimized_disabled, expr,
            "Should not optimize with rule disabled"
        );
    }
}

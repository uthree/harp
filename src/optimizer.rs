use crate::pats;
use crate::pattern::{PatternMatcher, UPat};
use crate::uop::{Op, UOp};
use log::debug;
use rustc_hash::FxHashMap;

pub struct Optimizer {
    matcher: PatternMatcher,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimizer {
    pub fn new() -> Self {
        let rules: Vec<UPat> = pats!({
            // --- Algebraic Simplification (f32) ---
            (a) | &a + &UOp::from(0.0f32) => a,
            (a) | &a * &UOp::from(1.0f32) => a,
            (a) | &a * &UOp::from(0.0f32) => UOp::from(0.0f32),
            (a) | a.recip().recip() => a,

            // --- Algebraic Simplification (i32) ---
            (a) | &a + &UOp::from(0i32) => a,
            (a) | &a * &UOp::from(1i32) => a,
            (a) | &a * &UOp::from(0i32) => UOp::from(0i32),
        });
        Self {
            matcher: PatternMatcher::new(rules),
        }
    }

    /// Applies optimization rules to the UOp graph until a fixed point is reached.
    pub fn optimize(&self, uop: &UOp) -> UOp {
        debug!("Before optimization: {uop:?}");
        let optimized_uop = self.matcher.apply_all_with_limit(uop, 100);
        debug!("After optimization: {optimized_uop:?}");
        optimized_uop
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uop::DType;

    #[test]
    fn test_algebraic_simplification() {
        let optimizer = Optimizer::new();
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
        let optimizer = Optimizer::new();
        let a = UOp::var("a", DType::F32);

        // Test (a * 1) + 0 => a
        let expr = (&a * &UOp::from(1.0f32)) + &UOp::from(0.0f32);
        let optimized = optimizer.optimize(&expr);
        assert_eq!(optimized, a);
    }
}

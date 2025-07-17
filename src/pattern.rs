use crate::uop::{Op, UOp};
use std::collections::HashMap;

use std::rc::Rc;

// A single pattern rule
pub struct UPat {
    pattern: UOp,
    replacer: Box<dyn Fn(&HashMap<usize, UOp>) -> UOp>,
}

impl UPat {
    pub fn new<F>(pattern: UOp, replacer: F) -> Self
    where
        F: Fn(&HashMap<usize, UOp>) -> UOp + 'static,
    {
        UPat {
            pattern,
            replacer: Box::new(replacer),
        }
    }

    fn apply(&self, target: &UOp) -> UOp {
        if let Some(captures) = self.matcher(target) {
            return (self.replacer)(&captures);
        }

        let new_src: Vec<UOp> = target.0.src.iter().map(|s| self.apply(s)).collect();

        if target.0.src.iter().zip(&new_src).any(|(a, b)| !a.eq(b)) {
            UOp::new(target.0.op.clone(), target.0.dtype.clone(), new_src)
        } else {
            target.clone()
        }
    }

    fn matcher(&self, target: &UOp) -> Option<HashMap<usize, UOp>> {
        let mut captures = HashMap::new();
        if self.internal_matcher(target, &self.pattern, &mut captures) {
            Some(captures)
        } else {
            None
        }
    }

    fn internal_matcher(
        &self,
        target: &UOp,
        pattern: &UOp,
        captures: &mut HashMap<usize, UOp>,
    ) -> bool {
        if let Op::Capture(id) = &pattern.0.op {
            if let Some(existing) = captures.get(id) {
                return target == existing;
            }
            captures.insert(*id, target.clone());
            return true;
        }

        if target.0.op != pattern.0.op || target.0.src.len() != pattern.0.src.len() {
            return false;
        }

        target
            .0
            .src
            .iter()
            .zip(pattern.0.src.iter())
            .all(|(s, p)| self.internal_matcher(s, p, captures))
    }
}

// A collection of patterns to be applied sequentially
#[derive(Clone)]
pub struct PatternMatcher {
    rules: Rc<Vec<UPat>>,
    children: Rc<Vec<PatternMatcher>>,
}

impl PatternMatcher {
    pub fn new(rules: Vec<UPat>) -> Self {
        Self {
            rules: Rc::new(rules),
            children: Rc::new(vec![]),
        }
    }

    pub fn apply(&self, uop: &UOp) -> UOp {
        let mut new_uop = uop.clone();
        for p in self.rules.iter() {
            new_uop = p.apply(&new_uop);
        }
        for child in self.children.iter() {
            new_uop = child.apply(&new_uop);
        }
        new_uop
    }

    pub fn apply_all_with_limit(&self, uop: &UOp, limit: usize) -> UOp {
        let mut current_uop = uop.clone();
        for _ in 0..limit {
            let new_uop = self.apply(&current_uop);
            if new_uop == current_uop {
                return new_uop;
            }
            current_uop = new_uop;
        }
        current_uop
    }
}

impl std::ops::Add for PatternMatcher {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            rules: Rc::new(vec![]),
            children: Rc::new(vec![self, rhs]),
        }
    }
}

#[macro_export]
macro_rules! pats {
    ({ $($arms:tt)* }) => {
        {
            let mut rules = Vec::new();
            pats!(@internal rules, $($arms)*);
            rules
        }
    };
    (@internal $rules:expr, ($($cap_var:ident),*) | $pattern:expr => $replacer:expr, $($rest:tt)*) => {
        let rule = {
            let pattern_uop = {
                let mut counter = 0..;
                $(
                    #[allow(unused_variables)]
                    let $cap_var = UOp::new(Op::Capture(counter.next().unwrap()), $crate::uop::DType::U8, vec![]);
                )*
                $pattern
            };
            let replacer_fn = move |caps: &HashMap<usize, UOp>| {
                let mut counter = 0..;
                $(
                    #[allow(unused_variables)]
                    let $cap_var = caps.get(&counter.next().unwrap()).unwrap().clone();
                )*
                $replacer
            };
            UPat::new(pattern_uop, replacer_fn)
        };
        $rules.push(rule);
        pats!(@internal $rules, $($rest)*);
    };
    (@internal $rules:expr,) => {};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uop::UOp;
    use rstest::rstest;

    #[rstest]
    fn test_distributive_law_pattern() {
        // Define variables
        let a: UOp = 1i32.into();
        let b: UOp = 2i32.into();
        let c: UOp = 3i32.into();

        // Define expression: a * b + a * c
        let expr = &a * &b + &a * &c;

        // 1. Define a set of rules using the `pats!` macro
        let rules = pats!({
            (p_a, p_b, p_c) | &p_a * &p_b + &p_a * &p_c => &p_a * (&p_b + &p_c),
        });

        // 2. Create a matcher and apply the rules
        let matcher = PatternMatcher::new(rules);
        let replaced_expr = matcher.apply_all_with_limit(&expr, 10);

        // Define expected expression: a * (b + c)
        let expected_expr = &a * (&b + &c);

        assert_eq!(replaced_expr, expected_expr);
    }

    #[rstest]
    fn test_associative_law_pattern() {
        let a: UOp = 1i32.into();
        let b: UOp = 2i32.into();
        let c: UOp = 3i32.into();
        let expr = &a + (&b + &c);
        let rules = pats!({
            (p_a, p_b, p_c) | &p_a + (&p_b + &p_c) => (&p_a + &p_b) + &p_c,
        });
        let matcher = PatternMatcher::new(rules);
        let replaced_expr = matcher.apply_all_with_limit(&expr, 10);
        let expected_expr = (&a + &b) + &c;
        assert_eq!(replaced_expr, expected_expr);
    }

    #[rstest]
    fn test_double_negation_pattern() {
        let a: UOp = 5i32.into();
        let expr = -(-a.clone());
        let rules = pats!({
            (p_a) | -(-p_a.clone()) => p_a,
        });
        let matcher = PatternMatcher::new(rules);
        let replaced_expr = matcher.apply_all_with_limit(&expr, 10);
        assert_eq!(replaced_expr, a);
    }

    #[rstest]
    fn test_no_match_pattern() {
        let a: UOp = 1i32.into();
        let b: UOp = 2i32.into();
        let expr = &a * &b;
        let rules = pats!({
            (p_x, p_y) | &p_x + &p_y => &p_y + &p_x,
        });
        let matcher = PatternMatcher::new(rules);
        let replaced_expr = matcher.apply_all_with_limit(&expr, 10);
        assert_eq!(replaced_expr, expr);
    }

    #[rstest]
    fn test_constant_folding_pattern() {
        let a: UOp = 2i32.into();
        let b: UOp = 3i32.into();
        let expr = &a * (&b + &a);
        let rules = pats!({
            (p_a, p_b) | &p_a * (&p_b + &p_a) => &p_a * &p_b + &p_a * &p_a,
        });
        let matcher = PatternMatcher::new(rules);
        let replaced_expr = matcher.apply_all_with_limit(&expr, 10);
        let expected_expr = &a * &b + &a * &a;
        assert_eq!(replaced_expr, expected_expr);
    }

    #[test]
    fn test_apply_all_with_limit() {
        use crate::uop::{DType, Number};
        // Define variables
        let a: UOp = UOp::var("a", DType::I32);
        let one: UOp = 1i32.into();
        let zero: UOp = 0i32.into();

        // Define patterns. The order matters for the test.
        let rules = pats!({
            // This rule will be applied first, but won't match the top level of the input
            (x) | x.clone() * one.clone() => x,
            // This rule will be applied second, and will match the top level of the intermediate graph
            (y) | y.clone() + zero.clone() => y,
        });

        let matcher = PatternMatcher::new(rules);

        // Input: (a * 1) + 0
        let input_uop = (a.clone() * one.clone()) + zero.clone();

        // After one pass of `matcher.apply`, the graph becomes `a + 0`.
        // Let's trace:
        // 1. Rule `x*1=>x` is applied to `(a*1)+0`.
        //    - Top level (Add) doesn't match.
        //    - Recurses to `src[0]` which is `a*1`. It matches! Returns `a`.
        //    - The graph becomes `a + 0`.
        // 2. Rule `y+0=>y` is applied to the result `a + 0`.
        //    - It matches! Returns `a`.
        // So, a single call to `matcher.apply` fully optimizes the graph.
        // The test logic was flawed. The `apply` method is too powerful.

        // To properly test `apply_all_with_limit`, we need `apply` to be less "deep".
        // Let's modify `PatternMatcher::apply` to only apply rules at the top level.
        // No, let's modify the test to be more robust.

        // New test plan:
        // The key is that one rule's output is the input for another rule in the *next* pass.
        let rules = pats!({
            // This rule is checked first.
            (y) | y.clone() + UOp::from(0i32) => y,
            // This rule is checked second. It creates input for the first rule.
            (x) | x.clone() * one.clone() => x.clone() + UOp::from(0i32),
        });
        let matcher = PatternMatcher::new(rules);
        let input_uop = a.clone() * one.clone(); // input is `a * 1`

        // ... (rest of the test)

        // Test with limit = 1
        let partially_optimized = matcher.apply_all_with_limit(&input_uop, 1);
        assert_eq!(partially_optimized, a.clone() + UOp::from(0i32));

        // Test with limit = 10
        let fully_optimized = matcher.apply_all_with_limit(&input_uop, 10);
        assert_eq!(fully_optimized, a);
    }

    #[test]
    fn test_matcher_fusion() {
        use crate::uop::DType;
        let a: UOp = UOp::var("a", DType::I32);
        let one: UOp = 1i32.into();
        let zero: UOp = 0i32.into();

        // Matcher 1: Simplifies multiplication by one
        let mul_rules = pats!({ (x) | x.clone() * one.clone() => x, });
        let mul_matcher = PatternMatcher::new(mul_rules);

        // Matcher 2: Simplifies addition of zero
        let add_rules = pats!({ (y) | y.clone() + zero.clone() => y, });
        let add_matcher = PatternMatcher::new(add_rules);

        // Fuse the matchers
        let fused_matcher = mul_matcher + add_matcher;

        // Input: (a * 1) + 0
        let input_uop = (a.clone() * one) + zero;

        // Apply the fused matcher
        let optimized_uop = fused_matcher.apply_all_with_limit(&input_uop, 10);

        // The result should be fully simplified
        assert_eq!(optimized_uop, a);
    }
}

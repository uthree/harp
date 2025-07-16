use crate::uop::{Ops, UOp};
use std::collections::HashMap;

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
        if let Ops::Capture(id) = &pattern.0.op {
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
pub struct PatternMatcher {
    rules: Vec<UPat>,
}

impl PatternMatcher {
    pub fn new(rules: Vec<UPat>) -> Self {
        Self { rules }
    }

    pub fn apply_all(&self, target: &UOp) -> UOp {
        let mut current_expr = target.clone();
        for rule in &self.rules {
            current_expr = rule.apply(&current_expr);
        }
        current_expr
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
                let mut i = 0;
                $(
                    #[allow(unused_variables)]
                    let $cap_var = UOp::new(Ops::Capture({let old_i = i; i += 1; old_i}), DType::I32, vec![]);
                )*
                $pattern
            };
            let replacer_fn = |caps: &HashMap<usize, UOp>| {
                let mut i = 0;
                $(
                    #[allow(unused_variables)]
                    let $cap_var = caps.get(&{let old_i = i; i += 1; old_i}).unwrap().clone();
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
        let replaced_expr = matcher.apply_all(&expr);

        // Define expected expression: a * (b + c)
        let expected_expr = &a * (&b + &c);

        assert_eq!(replaced_expr, expected_expr);
    }
}

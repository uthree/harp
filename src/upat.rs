use crate::uop::{Op, UOp};
use rustc_hash::FxHashMap;
use std::rc::Rc;

pub struct UPat {
    pattern: UOp,
    rewriter: Box<dyn Fn(Vec<UOp>) -> UOp>,
}

impl UPat {
    fn scan(&self, target: &UOp, pattern: &UOp, store: &mut FxHashMap<usize, UOp>) -> bool {
        // If the pat node is a capture, try capture the target node.
        if let Op::Capture(id) = pattern.op {
            if let Some(existing) = store.get(&id) {
                return target == existing;
            }
            store.insert(id, target.clone());
            return true;
        }

        // Check if the operation and number of children match.
        if target.op != pattern.op || target.src.len() != pattern.src.len() {
            return false;
        }

        // recursively match children
        target
            .src
            .iter()
            .zip(pattern.src.iter())
            .all(|(s, p)| self.scan(s, p, store))
    }
    fn capture(&self, target: &UOp) -> Option<Vec<UOp>> {
        let mut captures = FxHashMap::default();
        if self.scan(target, &self.pattern, &mut captures) {
            let mut captures = captures.into_iter().collect::<Vec<(usize, UOp)>>();
            captures.sort_by_key(|&(i, _)| i);
            Some(captures.into_iter().map(|(_, v)| v).collect())
        } else {
            None
        }
    }

    // Apply rewrite rule recursively
    pub fn apply(&self, target: &UOp) -> UOp {
        if let Some(captures) = self.capture(target) {
            return (self.rewriter)(captures);
        }
        let new_src: Vec<UOp> = target.src.iter().map(|s| self.apply(s)).collect();
        if target.src.iter().zip(&new_src).any(|(a, b)| !a.eq(b)) {
            UOp::new(target.op.clone(), new_src, target.dtype.clone())
        } else {
            target.clone()
        }
    }

    pub fn new<F>(pattern: UOp, rewriter: F) -> Rc<UPat>
    where
        F: Fn(Vec<UOp>) -> UOp + 'static,
    {
        Rc::new(UPat {
            pattern,
            rewriter: Box::new(rewriter),
        })
    }
}

#[derive(Clone)]
pub struct UPatternMatcher {
    patterns: Vec<Rc<UPat>>,
}
impl UPatternMatcher {
    pub fn new(patterns: Vec<Rc<UPat>>) -> Self {
        UPatternMatcher { patterns }
    }

    // apply all pattern
    pub fn apply(&self, target: UOp) -> UOp {
        let mut node = target.clone();
        for pat in self.patterns.iter() {
            node = pat.apply(&node);
        }
        node
    }

    pub fn merge(&mut self, other: &Self) {
        other
            .patterns
            .iter()
            .for_each(|pat| self.patterns.push(pat.clone()));
    }

    pub fn push(&mut self, pat: Rc<UPat>) {
        self.patterns.push(pat.clone());
    }
}

#[macro_export]
macro_rules! upat {
    (| $($capture: ident),* | $pattern: expr => $rewriter: expr ) => {
        {
            let mut counter = 0..;
            $(
                let $capture = UOp::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = move |captured_uops: Vec<UOp>| {
                let mut counter = 0..;
                $(
                    let $capture = captured_uops[counter.next().unwrap()].clone();
                )*
                $rewriter
            };
            UPat::new(pattern, rewriter)
        }
    };
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::uop::{DType, Op, UOp};

    // Helper to create a variable UOp for tests
    fn var(name: &str, dtype: DType) -> UOp {
        UOp::new(Op::Var(name.to_string()), vec![], dtype)
    }

    #[test]
    fn test_upat_macro() {
        // Test if the macro compiles and creates a UPat
        let _pat = upat!(|a, b| a + b => b + a);
    }

    #[test]
    fn test_scan_simple_match() {
        let target = var("x", DType::F32) + 1.0f32;
        let pattern = var("x", DType::F32) + 1.0f32;
        let pat = UPat::new(pattern, |_| panic!("should not be called"));
        let mut store = FxHashMap::default();
        assert!(pat.scan(&target, &pat.pattern, &mut store));
        assert!(store.is_empty());
    }

    #[test]
    fn test_scan_simple_mismatch() {
        let target = var("x", DType::F32) + 1.0f32;
        let pattern = var("y", DType::F32) + 1.0f32; // Different var name
        let pat = UPat::new(pattern, |_| panic!("should not be called"));
        let mut store = FxHashMap::default();
        assert!(!pat.scan(&target, &pat.pattern, &mut store));
    }

    #[test]
    fn test_scan_capture() {
        let target = var("x", DType::F32) + 1.0f32;
        let a = UOp::capture(0);
        let b = UOp::capture(1);
        let pattern = a + b;
        let pat = UPat::new(pattern, |_| panic!("should not be called"));
        let mut store = FxHashMap::default();

        assert!(pat.scan(&target, &pat.pattern, &mut store));
        assert_eq!(store.len(), 2);
        assert_eq!(store[&0], var("x", DType::F32));
        assert_eq!(store[&1], UOp::from(1.0f32));
    }

    #[test]
    fn test_scan_capture_consistency() {
        let target = var("x", DType::F32) + var("x", DType::F32);
        let a = UOp::capture(0);
        let pattern = a.clone() + a; // a + a
        let pat = UPat::new(pattern, |_| panic!("should not be called"));
        let mut store = FxHashMap::default();
        assert!(pat.scan(&target, &pat.pattern, &mut store));
        assert_eq!(store.len(), 1);
        assert_eq!(store[&0], var("x", DType::F32));

        // Mismatch case: x + y should not match a + a
        let target_mismatch = var("x", DType::F32) + var("y", DType::F32);
        let mut store_mismatch = FxHashMap::default();
        assert!(!pat.scan(&target_mismatch, &pat.pattern, &mut store_mismatch));
    }

    #[test]
    fn test_capture() {
        let pat = upat!(|a, b| a + b => b + a);
        let target = var("x", DType::F32) + 1.0f32;
        let captures = pat.capture(&target).unwrap();
        assert_eq!(captures.len(), 2);
        assert_eq!(captures[0], var("x", DType::F32));
        assert_eq!(captures[1], UOp::from(1.0f32));
    }

    #[test]
    fn test_capture_no_match() {
        let pat = upat!(|a| a.clone() * a => a);
        let target = var("x", DType::F32) + 1.0f32;
        assert!(pat.capture(&target).is_none());
    }

    #[test]
    fn test_apply_simple_rewrite() {
        let pat = upat!(|a, b| a + b => b + a);
        let target = var("x", DType::F32) + 1.0f32;
        let expected = UOp::from(1.0f32) + var("x", DType::F32);
        let result = pat.apply(&target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_apply_no_rewrite() {
        let pat = upat!(|a, b| a.clone() * b => b.clone() * a);
        let target = var("x", DType::F32) + 1.0f32;
        // UOp::clone is cheap, so we can clone it.
        let result = pat.apply(&target);
        assert_eq!(result, target);
    }

    #[test]
    fn test_apply_recursive() {
        // Pattern to simplify x - x => 0
        let pat = upat!(|a| a.clone() - a => {let _ = a; UOp::from(0.0f32)});
        let x = var("x", DType::F32);
        let y = var("y", DType::F32);
        // target: (y - y) + (x - x)
        let target = (y.clone() - y) + (x.clone() - x);

        // Expected: 0.0 + 0.0
        let expected = UOp::from(0.0f32) + UOp::from(0.0f32);
        let result = pat.apply(&target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_upattern_matcher() {
        let patterns = vec![
            // Rule 1: a + b => b + a (Commutative property of addition)
            upat!(|a, b| a + b => b + a),
            // Rule 2: a * 1 => a (Identity property of multiplication)
            upat!(|a| a.clone() * UOp::from(1.0f32) => a),
        ];
        let matcher = UPatternMatcher::new(patterns);

        let x = var("x", DType::F32);
        // target: (1.0 * x) + y
        let target = (UOp::from(1.0f32) * x.clone()) + var("y", DType::F32);

        // After rule 2 is not applied since pattern is a * 1 but target is 1 * x.
        // After rule 1: y + (1.0 * x)
        let expected = var("y", DType::F32) + (UOp::from(1.0f32) * x);
        let result = matcher.apply(target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_upattern_matcher_reordered() {
        let patterns = vec![
            // Rule 1: a * 1 => a
            upat!(|a| a.clone() * UOp::from(1.0f32) => a),
            // Rule 2: 1 * a => a
            upat!(|a| UOp::from(1.0f32) * a.clone() => a),
            // Rule 3: a + b => b + a
            upat!(|a, b| a + b => b + a),
        ];
        let matcher = UPatternMatcher::new(patterns);

        let x = var("x", DType::F32);
        let y = var("y", DType::F32);
        // target: y + (1.0 * x)
        let target = y.clone() + (UOp::from(1.0f32) * x.clone());

        // After rule 2: y + x
        // After rule 3: x + y
        let expected = x + y;
        let result = matcher.apply(target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_distributive_law_pattern() {
        // a * (b + c) => a * b + a * c
        let pat = upat!(|a, b, c| a.clone() * (b.clone() + c.clone()) => a.clone() * b + a * c);
        let a = var("a", DType::F32);
        let b = var("b", DType::F32);
        let c = var("c", DType::F32);

        let target = a.clone() * (b.clone() + c.clone());
        let expected = a.clone() * b + a * c;
        let result = pat.apply(&target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_subtraction_simplification() {
        // a - a => 0
        let pat = upat!(|a| a.clone() - a => {let _=a; UOp::from(0.0f32)});
        let a = var("a", DType::F32);
        let target = a.clone() - a;
        let expected = UOp::from(0.0f32);
        let result = pat.apply(&target);
        assert_eq!(result, expected);
    }
}

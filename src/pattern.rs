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
            pattern: pattern,
            rewriter: Box::new(rewriter),
        })
    }
}

pub struct UPatternMatcher {
    name: String,
    patterns: Vec<Rc<UPat>>,
}

impl UPatternMatcher {
    pub fn new(name: &str, patterns: Vec<Rc<UPat>>) -> Self {
        UPatternMatcher {
            name: name.to_string(),
            patterns: patterns,
        }
    }

    pub fn apply(&self, target: UOp) -> UOp {
        let mut node = target.clone();
        for pat in self.patterns.iter() {
            node = pat.apply(&node);
        }
        node
    }
}

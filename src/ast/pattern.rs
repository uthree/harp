use crate::ast::{AstNode, Op};
use rustc_hash::FxHashMap;
use std::rc::Rc;

pub struct RewriteRule {
    pattern: AstNode,
    rewriter: Box<dyn Fn(Vec<AstNode>) -> AstNode>,
}

impl RewriteRule {
    fn scan(
        &self,
        target: &AstNode,
        pattern: &AstNode,
        store: &mut FxHashMap<usize, AstNode>,
    ) -> bool {
        if let Op::Capture(id) = &pattern.op {
            if let Some(existing) = store.get(id) {
                return target == existing;
            }
            store.insert(*id, target.clone());
            return true;
        }

        if target.op != pattern.op || target.src.len() != pattern.src.len() {
            return false;
        }

        target
            .src
            .iter()
            .zip(pattern.src.iter())
            .all(|(s, p)| self.scan(s, p, store))
    }

    fn capture(&self, target: &AstNode) -> Option<Vec<AstNode>> {
        let mut captures = FxHashMap::default();
        if self.scan(target, &self.pattern, &mut captures) {
            let mut captures = captures.into_iter().collect::<Vec<(usize, AstNode)>>();
            captures.sort_by_key(|&(i, _)| i);
            Some(captures.into_iter().map(|(_, v)| v).collect())
        } else {
            None
        }
    }

    pub fn apply_all(&self, target: AstNode) -> AstNode {
        // Apply to children first (post-order traversal)
        let new_src = target
            .src
            .into_iter()
            .map(|s| Box::new(self.apply_all(*s)))
            .collect();

        let new_target = AstNode::new(target.op, new_src, target.dtype);

        // Then apply to the current node
        if let Some(captures) = self.capture(&new_target) {
            (self.rewriter)(captures)
        } else {
            new_target
        }
    }

    pub fn new<F>(pattern: AstNode, rewriter: F) -> Rc<RewriteRule>
    where
        F: Fn(Vec<AstNode>) -> AstNode + 'static,
    {
        Rc::new(RewriteRule {
            pattern,
            rewriter: Box::new(rewriter),
        })
    }
}

#[derive(Clone)]
pub struct AstRewriter {
    rules: Vec<Rc<RewriteRule>>,
    max_iterations: usize,
}

impl AstRewriter {
    pub fn new(rules: Vec<Rc<RewriteRule>>) -> Self {
        AstRewriter {
            rules,
            max_iterations: 1000,
        }
    }

    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    pub fn apply(&self, mut node: AstNode) -> AstNode {
        for _ in 0..self.max_iterations {
            let mut changed = false;
            for rule in &self.rules {
                let new_node = rule.apply_all(node.clone());
                if new_node != node {
                    node = new_node;
                    changed = true;
                }
            }
            if !changed {
                return node;
            }
        }
        node
    }

    pub fn merge(&mut self, other: &Self) {
        other
            .rules
            .iter()
            .for_each(|rule| self.rules.push(rule.clone()));
    }

    pub fn push(&mut self, rule: Rc<RewriteRule>) {
        self.rules.push(rule);
    }
}

#[macro_export]
macro_rules! rule {
    (| $($capture: ident),* | $pattern: expr => $rewriter: expr ) => {
        {
            let mut counter = 0..;
            $(
                let $capture = AstNode::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = move |captured_nodes: Vec<AstNode>| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $rewriter
            };
            RewriteRule::new(pattern, rewriter)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstNode, DType, Op};

    fn var(name: &str, dtype: DType) -> AstNode {
        AstNode::new(Op::Var(name.to_string()), vec![], dtype)
    }

    #[test]
    fn test_rule_macro() {
        let _rule = rule!(|a, b| a + b => b + a);
    }

    #[test]
    fn test_apply_all() {
        let x = var("x", DType::F32);
        let y = var("y", DType::F32);
        // (x + y) + (x + y)
        let target = (x.clone() + y.clone()) + (x.clone() + y.clone());
        // (x - y) - (x - y)
        let expected = (x.clone() - y.clone()) - (x.clone() - y.clone());
        
        let rule = rule!(|a, b| a + b => a - b);
        let result = rule.apply_all(target);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_ast_rewriter_fixed_point() {
        let x = var("x", DType::F32);
        // x * 1.0 * 1.0
        let target = (x.clone() * 1.0f32) * 1.0f32;
        let expected = x;

        let rewriter = AstRewriter::new(vec![
            rule!(|a| a.clone() * 1.0f32 => a)
        ]);

        let result = rewriter.apply(target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ast_rewriter_commutative() {
        let x = var("x", DType::F32);
        let y = var("y", DType::F32);
        let target = y.clone() + x.clone();
        let expected = x.clone() + y.clone();

        let rewriter = AstRewriter::new(vec![
            rule!(|a, b| a + b => b + a),
        ]).with_max_iterations(1); // Should only need one iteration

        let result = rewriter.apply(target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ast_rewriter_distributive_law() {
        let a = var("a", DType::F32);
        let b = var("b", DType::F32);
        let c = var("c", DType::F32);
        // a * (b + c)
        let target = a.clone() * (b.clone() + c.clone());
        // a * b + a * c
        let expected = (a.clone() * b.clone()) + (a.clone() * c.clone());

        let rewriter = AstRewriter::new(vec![
            rule!(|x, y, z| x.clone() * (y.clone() + z.clone()) => (x.clone() * y) + (x * z))
        ]);

        let result = rewriter.apply(target);
        assert_eq!(result, expected);
    }
}

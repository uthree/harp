use crate::ast::{AstNode, AstOp};
use crate::backend::KernelDetails;
use log::{debug, info, trace};
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// A trait for applying deterministic optimizations to an `AstNode`.
///
/// This typically involves applying a set of rewrite rules to simplify
/// the AST.
pub trait DeterministicAstOptimizer {
    /// Optimizes the given `AstNode` and returns a new, optimized `AstNode`.
    fn optimize(&self, node: AstNode, details: &KernelDetails) -> AstNode;
}

/// A trait for suggesting possible optimization steps for a given `AstNode`.
///
/// Instead of applying a single rule, a suggester returns a list of potential
/// new `AstNode`s that could replace the original one.
pub trait OptimizationSuggester {
    fn suggest_optimizations(&self, node: &AstNode) -> Vec<AstNode>;
}

/// A trait for estimating the computational cost of an `AstNode`.
///
/// The cost can be a metric like estimated execution time, memory usage,
/// or simply the number of operations.
pub trait CostEstimator {
    /// Estimates the cost of the given `AstNode`. A lower value is better.
    fn estimate_cost(&self, node: &AstNode, details: &KernelDetails) -> f32;
}

pub struct RewriteRule {
    name: String,
    pattern: AstNode,
    pub rewriter: Box<dyn Fn(Vec<AstNode>) -> AstNode>,
}

impl RewriteRule {
    fn scan(target: &AstNode, pattern: &AstNode, store: &mut FxHashMap<usize, AstNode>) -> bool {
        trace!(
            "Scanning target node {:?} with pattern node {:?}",
            target.op, pattern.op
        );
        if let AstOp::Capture(id, _) = &pattern.op {
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
            .all(|(s, p)| Self::scan(s, p, store))
    }

    pub fn capture(&self, target: &AstNode) -> Option<Vec<AstNode>> {
        let mut captures = FxHashMap::default();
        if Self::scan(target, &self.pattern, &mut captures) {
            let mut captures = captures.into_iter().collect::<Vec<(usize, AstNode)>>();
            captures.sort_by_key(|&(i, _)| i);
            Some(captures.into_iter().map(|(_, v)| v).collect())
        } else {
            None
        }
    }

    pub fn apply_all(&self, target: AstNode) -> AstNode {
        // Apply to children first (post-order traversal)
        let new_src = target.src.into_iter().map(|s| self.apply_all(s)).collect();

        let new_target = AstNode::new(target.op, new_src, target.dtype);

        // Then apply to the current node
        if let Some(captures) = self.capture(&new_target) {
            debug!(
                "Rule '{}' matched node {:?}. Applying rewrite.",
                self.name, new_target.op
            );
            let rewritten_node = (self.rewriter)(captures);
            trace!(
                "Node {:?} rewritten to {:?}",
                new_target.op, rewritten_node.op
            );
            rewritten_node
        } else {
            new_target
        }
    }

    pub fn new<F>(name: &str, pattern: AstNode, rewriter: F) -> Rc<RewriteRule>
    where
        F: Fn(Vec<AstNode>) -> AstNode + 'static,
    {
        Rc::new(RewriteRule {
            name: name.to_string(),
            pattern,
            rewriter: Box::new(rewriter),
        })
    }
}

#[derive(Clone)]
pub struct AstRewriter {
    pub rules: Vec<Rc<RewriteRule>>,
    max_iterations: usize,
}

impl DeterministicAstOptimizer for AstRewriter {
    fn optimize(&self, node: AstNode, _details: &KernelDetails) -> AstNode {
        self.apply(node)
    }
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
        info!("Starting AST rewrite process...");
        for i in 0..self.max_iterations {
            let mut changed = false;
            let original_node = node.clone();

            for rule in &self.rules {
                trace!("Applying rule '{}'", rule.name);
                node = rule.apply_all(node);
            }

            if original_node != node {
                changed = true;
            }

            if !changed {
                info!("Rewrite reached fixed point after {i} iterations.");
                return node;
            }
            debug!("AST changed in iteration {i}. Continuing...");
        }
        info!(
            "Rewrite finished after reaching max iterations ({}).",
            self.max_iterations
        );
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
    ($name: expr, | $($capture: ident),* | $pattern: expr => $rewriter: expr ) => {
        {
            use $crate::ast::DType;
            let mut counter = 0..;
            $(
                let $capture = AstNode::capture(counter.next().unwrap(), DType::Any);
            )*
            let pattern = $pattern;
            let rewriter = move |captured_nodes: Vec<AstNode>| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $rewriter
            };
            RewriteRule::new($name, pattern, rewriter)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    fn setup_logger() {
        // Initialize the logger for tests, ignoring errors if it's already set up
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_rule_macro() {
        setup_logger();
        let _rule = rule!("test_rule", |a, b| a + b => b + a);
    }

    #[test]
    fn test_apply_all() {
        setup_logger();
        let x = AstNode::var("x");
        let y = AstNode::var("y");
        // (x + y) + (x + y)
        let target = (x.clone() + y.clone()) + (x.clone() + y.clone());
        // (x - y) - (x - y)
        let expected = (x.clone() - y.clone()) - (x.clone() - y.clone());

        let rule = rule!("plus_to_minus", |a, b| a + b => a - b);
        let result = rule.apply_all(target);

        assert_eq!(result, expected);
    }

    #[test]
    fn test_ast_rewriter_fixed_point() {
        setup_logger();
        use super::RewriteRule;
        use crate::ast::DType;
        let x = AstNode::var("x").with_type(DType::F32);
        // x * 1.0 * 1.0
        let target = (x.clone() * 1.0f32) * 1.0f32;
        let expected = x;

        let pattern = AstNode::capture(0, DType::F32) * 1.0f32;
        let rewriter_fn = |nodes: Vec<AstNode>| nodes[0].clone();
        let rule = RewriteRule::new("mul_by_one", pattern, rewriter_fn);
        let rewriter = AstRewriter::new(vec![rule]);

        let result = rewriter.apply(target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ast_rewriter_commutative() {
        setup_logger();
        let x = AstNode::var("x");
        let y = AstNode::var("y");
        let target = y.clone() + x.clone();
        let expected = x.clone() + y.clone();

        let rewriter = AstRewriter::new(vec![rule!("commutative_add", |a, b| a + b => b + a)])
            .with_max_iterations(1); // Should only need one iteration

        let result = rewriter.apply(target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ast_rewriter_distributive_law() {
        setup_logger();
        let a = AstNode::var("a");
        let b = AstNode::var("b");
        let c = AstNode::var("c");
        // a * (b + c)
        let target = a.clone() * (b.clone() + c.clone());
        // a * b + a * c
        let expected = (a.clone() * b.clone()) + (a.clone() * c.clone());

        let rewriter = AstRewriter::new(vec![rule!("distributive", |x, y, z| x.clone()
            * (y.clone() + z.clone()) =>
            (x.clone() * y) + (x * z))]);

        let result = rewriter.apply(target);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ast_rewriter_type_match() {
        setup_logger();
        use crate::ast::DType;

        let a_real = AstNode::var("a").with_type(DType::F32);
        let a_int = AstNode::var("a").with_type(DType::I32);

        // This rule should only apply to Real types.
        // The pattern captures any node.
        let pattern = AstNode::capture(0, DType::Any);
        let rewriter_fn = |nodes: Vec<AstNode>| {
            let node = nodes[0].clone();
            // The type check is now performed inside the rewriter.
            if node.dtype.is_real() {
                node.with_type(DType::F64) // Apply the rewrite
            } else {
                node // Return the original node if the type does not match
            }
        };
        let rule = RewriteRule::new("real_to_f64", pattern, rewriter_fn);

        let rewriter = AstRewriter::new(vec![rule]);

        // The rule should apply to a_real (F32 is a Real)
        let result_real = rewriter.apply(a_real.clone());
        assert_eq!(result_real.dtype, DType::F64);

        // The rule should NOT apply to a_int
        let result_int = rewriter.apply(a_int.clone());
        assert_eq!(result_int, a_int);
    }
}

/// A collection of algebraic simplification rules.
#[derive(Clone)]
pub struct AlgebraicSimplification {
    rewriter: AstRewriter,
}

impl Default for AlgebraicSimplification {
    fn default() -> Self {
        let rules = vec![
            // F32 rules
            rule!("mul_by_one_f32", |a| a.clone() * AstNode::from(1.0f32) => a),
            rule!("mul_by_zero_f32", |_a| _a * AstNode::from(0.0f32) => AstNode::from(0.0f32)),
            rule!("add_zero_f32", |a| a.clone() + AstNode::from(0.0f32) => a),
            // F64 rules
            rule!("mul_by_one_f64", |a| a.clone() * AstNode::from(1.0f64) => a),
            rule!("mul_by_zero_f64", |_a| _a * AstNode::from(0.0f64) => AstNode::from(0.0f64)),
            rule!("add_zero_f64", |a| a.clone() + AstNode::from(0.0f64) => a),
        ];
        Self {
            rewriter: AstRewriter::new(rules),
        }
    }
}

impl AlgebraicSimplification {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn rules(&self) -> Vec<Rc<RewriteRule>> {
        self.rewriter.rules.clone()
    }
}

impl OptimizationSuggester for AlgebraicSimplification {
    fn suggest_optimizations(&self, node: &AstNode) -> Vec<AstNode> {
        let dummy_details = KernelDetails::default();
        let optimized_node = self.rewriter.optimize(node.clone(), &dummy_details);
        if &optimized_node != node {
            vec![optimized_node]
        } else {
            vec![]
        }
    }
}

/// A suggester for loop unrolling optimization.
#[derive(Clone)]
pub struct LoopUnrolling {
    unroll_factor: usize,
}

impl LoopUnrolling {
    pub fn new(unroll_factor: usize) -> Self {
        Self { unroll_factor }
    }
}

impl Default for LoopUnrolling {
    fn default() -> Self {
        Self::new(4) // Default unroll factor
    }
}

impl OptimizationSuggester for LoopUnrolling {
    fn suggest_optimizations(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();
        if let AstOp::Range { loop_var } = &node.op {
            if let Some(AstOp::Const(const_val)) = node.src.get(0).map(|n| &n.op) {
                if let Some(end) = const_val.to_usize() {
                    if end >= self.unroll_factor {
                        let num_unrolled_iterations = end / self.unroll_factor;
                        let remaining_iterations = end % self.unroll_factor;

                        let mut unrolled_body = Vec::new();
                        let loop_body = &node.src[1..];

                        // Create the unrolled loop
                        for i in 0..num_unrolled_iterations {
                            for j in 0..self.unroll_factor {
                                let current_index = i * self.unroll_factor + j;
                                let substitution = AstNode::from(current_index as i32);
                                for stmt in loop_body {
                                    unrolled_body.push(replace_var(stmt, loop_var, &substitution));
                                }
                            }
                        }

                        // Handle remaining iterations
                        for i in (end - remaining_iterations)..end {
                            let substitution = AstNode::from(i as i32);
                            for stmt in loop_body {
                                unrolled_body.push(replace_var(stmt, loop_var, &substitution));
                            }
                        }

                        suggestions.push(AstNode::new(
                            AstOp::Block,
                            unrolled_body,
                            node.dtype.clone(),
                        ));
                    }
                }
            }
        }
        suggestions
    }
}

/// Helper function to replace a variable in an AST node.
fn replace_var(node: &AstNode, var_name: &str, substitution: &AstNode) -> AstNode {
    if let AstOp::Var(name) = &node.op {
        if name == var_name {
            return substitution.clone();
        }
    }
    let new_src = node
        .src
        .iter()
        .map(|n| replace_var(n, var_name, substitution))
        .collect();
    AstNode::new(node.op.clone(), new_src, node.dtype.clone())
}

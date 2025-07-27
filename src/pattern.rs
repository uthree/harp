use crate::tensor::{Tensor, TensorOp};
use crate::uop::{Op, UOp};
use log::debug;
use rustc_hash::FxHashMap;
use std::rc::Rc;

/// A pattern for matching subgraphs within a `Tensor` computation graph.
#[derive(Clone, Debug)]
pub enum TPat {
    /// Matches any single `Tensor`.
    Wildcard,
    /// Captures a matched `Tensor`. The `usize` is the capture ID.
    Capture(usize),
    /// Matches a `Tensor` created by a unary operation.
    Unary(Op, Box<TPat>),
    /// Matches a `Tensor` created by a binary operation.
    Binary(Op, Box<TPat>, Box<TPat>),
    /// Matches a `Tensor` created by a reduce operation.
    Reduce(usize, Op, Box<TPat>),
    /// Matches a `Tensor` created by a scan operation.
    Scan(usize, Op, Box<TPat>),
    /// Matches a constant `Tensor`.
    Constant,
    /// Matches a load `Tensor`.
    Load,
}

/// Represents a single pattern matching rule for `UOp` graphs.
///
/// A `UPat` consists of a `pattern` `UOp` to match against and a `replacer`
/// function that generates a replacement `UOp` based on captured subgraphs.
pub struct UPat {
    pattern: UOp,
    replacer: Box<dyn Fn(&FxHashMap<usize, UOp>) -> UOp>,
}

/// Represents a single pattern matching rule for `Tensor` graphs.
pub struct TPatRule {
    pub name: String,
    pattern: TPat,
    condition: Box<dyn Fn(&FxHashMap<usize, Tensor>) -> bool>,
    replacer: Box<dyn Fn(&FxHashMap<usize, Tensor>) -> Option<Tensor>>,
}

impl TPatRule {
    /// Creates a new pattern matching rule with a condition.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the rule for logging.
    /// * `pattern` - The `TPat` to match against the `Tensor` graph.
    /// * `condition` - A closure that takes the captured tensors and returns `true`
    ///   if the rule should be applied. This allows for checking properties
    ///   that are not part of the graph structure, like a constant's value or a
    ///   tensor's shape.
    /// * `replacer` - A closure that takes the captured tensors and returns a
    ///   new `Tensor` to replace the matched subgraph.
    pub fn new<C, R>(name: &str, pattern: TPat, condition: C, replacer: R) -> Self
    where
        C: Fn(&FxHashMap<usize, Tensor>) -> bool + 'static,
        R: Fn(&FxHashMap<usize, Tensor>) -> Option<Tensor> + 'static,
    {
        TPatRule {
            name: name.to_string(),
            pattern,
            condition: Box::new(condition),
            replacer: Box::new(replacer),
        }
    }
}

/// A collection of `TPatRule`s that can be applied to a `Tensor` graph.
pub struct TensorPatternMatcher {
    rules: Vec<TPatRule>,
}

impl TensorPatternMatcher {
    pub fn new(rules: Vec<TPatRule>) -> Self {
        Self { rules }
    }

    pub fn apply(&self, tensor: &Tensor) -> Tensor {
        // Apply optimizations recursively, starting from the leaves of the graph.
        self.apply_recursive(tensor)
    }

    fn apply_recursive(&self, tensor: &Tensor) -> Tensor {
        // First, optimize the source tensors.
        let new_src: Vec<Tensor> = tensor.src.iter().map(|s| self.apply_recursive(s)).collect();

        // Rebuild the current tensor if any of its sources have changed.
        let mut current_tensor = if tensor.src.iter().zip(&new_src).any(|(a, b)| !Rc::ptr_eq(&a.0, &b.0)) {
            Tensor::new(
                tensor.op.clone(),
                new_src,
                tensor.tracker.clone(),
                tensor.dtype.clone(),
                tensor.backend.clone(),
            )
        } else {
            tensor.clone()
        };

        // Now, try to apply rules to the (potentially rebuilt) current tensor.
        // This loop allows for iterative optimization at the same node.
        loop {
            let mut optimized = false;
            for rule in &self.rules {
                if let Some(captures) = self.match_pattern(&current_tensor, &rule.pattern) {
                    if (rule.condition)(&captures) {
                        if let Some(replacement) = (rule.replacer)(&captures) {
                            debug!("Applied Tensor Pattern: {}", rule.name);
                            current_tensor = replacement;
                            optimized = true;
                            // Restart the rule application process from the beginning for the new tensor.
                            break;
                        }
                    }
                }
            }
            if !optimized {
                break; // No more rules could be applied at this node.
            }
        }

        current_tensor
    }

    fn match_pattern(&self, tensor: &Tensor, pattern: &TPat) -> Option<FxHashMap<usize, Tensor>> {
        let mut captures = FxHashMap::default();
        if self.internal_matcher(tensor, pattern, &mut captures) {
            Some(captures)
        } else {
            None
        }
    }

    fn internal_matcher<'p>(
        &self,
        tensor: &Tensor,
        pattern: &'p TPat,
        captures: &mut FxHashMap<usize, Tensor>,
    ) -> bool {
        match pattern {
            TPat::Wildcard => true,
            TPat::Capture(id) => {
                if let Some(existing) = captures.get(id) {
                    return Rc::ptr_eq(&tensor.0, &existing.0);
                }
                captures.insert(*id, tensor.clone());
                true
            }
            TPat::Constant => matches!(tensor.op, TensorOp::Constant(_)),
            TPat::Load => matches!(tensor.op, TensorOp::Load),
            TPat::Unary(op, pat_src) => {
                if let TensorOp::Unary(tensor_op) = &tensor.op {
                    op == tensor_op && self.internal_matcher(&tensor.src[0], pat_src, captures)
                } else {
                    false
                }
            }
            TPat::Binary(op, pat_lhs, pat_rhs) => {
                if let TensorOp::Binary(tensor_op) = &tensor.op {
                    op == tensor_op
                        && self.internal_matcher(&tensor.src[0], pat_lhs, captures)
                        && self.internal_matcher(&tensor.src[1], pat_rhs, captures)
                } else {
                    false
                }
            }
            TPat::Reduce(axis, op, pat_src) => {
                if let TensorOp::Reduce(tensor_axis, tensor_op) = &tensor.op {
                    axis == tensor_axis
                        && op == tensor_op
                        && self.internal_matcher(&tensor.src[0], pat_src, captures)
                } else {
                    false
                }
            }
            TPat::Scan(axis, op, pat_src) => {
                if let TensorOp::Scan {
                    axis: tensor_axis,
                    op: tensor_op,
                } = &tensor.op
                {
                    axis == tensor_axis
                        && op == tensor_op
                        && self.internal_matcher(&tensor.src[0], pat_src, captures)
                } else {
                    false
                }
            }
        }
    }
}

impl UPat {
    /// Creates a new pattern rule.
    ///
    /// # Arguments
    /// * `pattern` - A `UOp` graph where `Op::Capture(id)` nodes mark parts to be captured.
    /// * `replacer` - A closure that takes the map of captured `UOp`s and returns the replacement `UOp`.
    pub fn new<F>(pattern: UOp, replacer: F) -> Self
    where
        F: Fn(&FxHashMap<usize, UOp>) -> UOp + 'static,
    {
        UPat {
            pattern,
            replacer: Box::new(replacer),
        }
    }

    /// Recursively applies the pattern to a target `UOp` and its children.
    ///
    /// If the pattern matches at the current node, it's replaced. Otherwise,
    /// the function descends into the children and rebuilds the node if any
    /// of them were changed.
    fn apply(&self, target: &UOp) -> UOp {
        // Try to match at the current node
        if let Some(captures) = self.matcher(target) {
            return (self.replacer)(&captures);
        }

        // If no match, recurse into source nodes
        let new_src: Vec<UOp> = target.0.src.iter().map(|s| self.apply(s)).collect();

        // Rebuild the node only if any of its children have changed
        if target.0.src.iter().zip(&new_src).any(|(a, b)| !a.eq(b)) {
            UOp::new(target.0.op.clone(), target.0.dtype.clone(), new_src)
        } else {
            target.clone()
        }
    }

    /// Matches the pattern against a target `UOp`.
    ///
    /// Returns a map of captured nodes if the match is successful.
    fn matcher(&self, target: &UOp) -> Option<FxHashMap<usize, UOp>> {
        let mut captures = FxHashMap::default();
        if self.internal_matcher(target, &self.pattern, &mut captures) {
            Some(captures)
        } else {
            None
        }
    }

    /// The internal recursive matching logic.
    fn internal_matcher(
        &self,
        target: &UOp,
        pattern: &UOp,
        captures: &mut FxHashMap<usize, UOp>,
    ) -> bool {
        // If the pattern node is a capture, try to capture the target node.
        if let Op::Capture(id) = &pattern.0.op {
            if let Some(existing) = captures.get(id) {
                // If already captured, ensure it's the same as the current target.
                return target == existing;
            }
            captures.insert(*id, target.clone());
            return true;
        }

        // Check if the operation and number of children match.
        if target.0.op != pattern.0.op || target.0.src.len() != pattern.0.src.len() {
            return false;
        }

        // Recursively match children.
        target
            .0
            .src
            .iter()
            .zip(pattern.0.src.iter())
            .all(|(s, p)| self.internal_matcher(s, p, captures))
    }
}

/// A collection of `UPat` rules that can be applied to a `UOp` graph.
#[derive(Clone)]
pub struct PatternMatcher {
    rules: Rc<Vec<UPat>>,
    children: Rc<Vec<PatternMatcher>>,
}

impl PatternMatcher {
    /// Creates a new `PatternMatcher` with a given set of rules.
    pub fn new(rules: Vec<UPat>) -> Self {
        Self {
            rules: Rc::new(rules),
            children: Rc::new(vec![]),
        }
    }

    /// Returns `true` if the matcher has no rules.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty() && self.children.is_empty()
    }

    /// Applies all rules in the matcher to a `UOp` graph.
    pub fn apply(&self, uop: &UOp) -> UOp {
        let mut new_uop = uop.clone();
        for p in self.rules.iter() {
            new_uop = p.apply(&new_uop);
        }
        // Apply rules from child matchers (if any)
        for child in self.children.iter() {
            new_uop = child.apply(&new_uop);
        }
        new_uop
    }

    /// Applies all rules repeatedly until the `UOp` graph reaches a fixed point.
    ///
    /// # Arguments
    /// * `uop` - The `UOp` to optimize.
    /// * `limit` - A safeguard to prevent infinite loops, limiting the number of iterations.
    pub fn apply_all_with_limit(&self, uop: &UOp, limit: usize) -> UOp {
        let mut current_uop = uop.clone();
        for _ in 0..limit {
            let new_uop = self.apply(&current_uop);
            if new_uop == current_uop {
                return new_uop; // Fixed point reached
            }
            current_uop = new_uop;
        }
        current_uop
    }
}

impl std::ops::Add for PatternMatcher {
    type Output = Self;

    /// Combines two `PatternMatcher`s.
    ///
    /// This allows for composing sets of rules. The resulting matcher will
    /// apply the rules from `self` first, then the rules from `rhs`.
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            rules: Rc::new(vec![]),
            children: Rc::new(vec![self, rhs]),
        }
    }
}

/// A macro for concisely defining `UPat` rules.
///
/// # Example
///
/// ```
/// use harp::{pats, uop::{UOp, Op, DType}};
/// use harp::pattern::PatternMatcher;
/// use rustc_hash::FxHashMap;
///
/// let rules = pats!({
///     // Rule to replace `x + 0` with `x`
///     (x) | &x + &UOp::from(0.0f32) => x,
/// });
///
/// let matcher = PatternMatcher::new(rules);
/// let a = UOp::var("a", DType::F32);
/// let expr = &a + &UOp::from(0.0f32);
/// let optimized = matcher.apply(&expr);
///
/// assert_eq!(optimized, a);
/// ```
#[macro_export]
macro_rules! pats {
    ({ $($arms:tt)* }) => {
        {
            use $crate::pattern::UPat;
            use rustc_hash::FxHashMap;
            let mut rules = Vec::new();
            pats!(@internal rules, $($arms)*);
            rules
        }
    };
    // This arm handles a single rule definition.
    (@internal $rules:expr, ($($cap_var:ident),*) | $pattern:expr => $replacer:expr, $($rest:tt)*) => {
        let rule = {
            // Create the pattern UOp, replacing capture variables with `Op::Capture`.
            let pattern_uop = {
                let mut counter = 0..;
                $(
                    #[allow(unused_variables)]
                    let $cap_var = UOp::new(Op::Capture(counter.next().unwrap()), $crate::uop::DType::Unit, vec![]);
                )*
                $pattern
            };
            // Create the replacer function.
            let replacer_fn = move |caps: &FxHashMap<usize, UOp>| {
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
        // Recurse to handle the rest of the rules.
        pats!(@internal $rules, $($rest)*);
    };
    // Base case for the recursion.
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
        use crate::uop::DType;
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

        let _matcher = PatternMatcher::new(rules);

        // Input: (a * 1) + 0
        let _input_uop = (a.clone() * one.clone()) + zero.clone();

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

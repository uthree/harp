use crate::ast::{pattern::AstRewriteRule, AstNode};
use crate::opt::ast::heuristic::RewriteSuggester;
use std::rc::Rc;

/// A suggester that uses rewrite rules to propose alternative ASTs.
#[derive(Clone)]
pub struct RuleBasedSuggester {
    rules: Vec<Rc<AstRewriteRule>>,
}

impl RuleBasedSuggester {
    pub fn new(rules: Vec<Rc<AstRewriteRule>>) -> Self {
        Self { rules }
    }
}

impl RewriteSuggester for RuleBasedSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();
        for rule in &self.rules {
            suggestions.extend(rule.get_possible_rewrites(node));
        }
        suggestions
    }
}

/// A suggester for commutative operations (Add, Mul, Max).
/// Suggests swapping the operands of commutative operations.
pub struct CommutativeSuggester;

impl RewriteSuggester for CommutativeSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Try swapping commutative operations at the current node
        match node {
            AstNode::Add(a, b) => {
                suggestions.push(AstNode::Add(b.clone(), a.clone()));
            }
            AstNode::Mul(a, b) => {
                suggestions.push(AstNode::Mul(b.clone(), a.clone()));
            }
            AstNode::Max(a, b) => {
                suggestions.push(AstNode::Max(b.clone(), a.clone()));
            }
            _ => {}
        }

        // Recursively suggest swaps in children
        for (i, child) in node.children().iter().enumerate() {
            for suggested_child in self.suggest(child) {
                let mut new_children: Vec<AstNode> =
                    node.children().iter().map(|c| (*c).clone()).collect();
                new_children[i] = suggested_child;
                suggestions.push(node.clone().replace_children(new_children));
            }
        }

        suggestions
    }
}

/// A suggester that proposes removing redundant operations.
/// For example, suggests removing a Store operation that is never read.
pub struct RedundancyRemovalSuggester;

impl RewriteSuggester for RedundancyRemovalSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // For Block nodes, suggest removing each statement
        if let AstNode::Block { scope, statements } = node {
            for i in 0..statements.len() {
                let mut new_statements = statements.clone();
                new_statements.remove(i);
                suggestions.push(AstNode::Block {
                    scope: scope.clone(),
                    statements: new_statements,
                });
            }
        }

        // Recursively suggest removals in children
        for (i, child) in node.children().iter().enumerate() {
            for suggested_child in self.suggest(child) {
                let mut new_children: Vec<AstNode> =
                    node.children().iter().map(|c| (*c).clone()).collect();
                new_children[i] = suggested_child;
                suggestions.push(node.clone().replace_children(new_children));
            }
        }

        suggestions
    }
}

/// A suggester that removes inverse operations like log2(exp2(x)) -> x
pub struct InverseOperationSuggester;

impl RewriteSuggester for InverseOperationSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // log2(exp2(x)) -> x
        if let AstNode::Log2(inner) = node {
            if let AstNode::Exp2(x) = &**inner {
                suggestions.push((**x).clone());
            }
        }

        // exp2(log2(x)) -> x
        if let AstNode::Exp2(inner) = node {
            if let AstNode::Log2(x) = &**inner {
                suggestions.push((**x).clone());
            }
        }

        // sqrt(x * x) -> x (assuming x >= 0)
        if let AstNode::Sqrt(inner) = node {
            if let AstNode::Mul(a, b) = &**inner {
                if a == b {
                    suggestions.push((**a).clone());
                }
            }
        }

        // Recursively suggest in children
        for (i, child) in node.children().iter().enumerate() {
            for suggested_child in self.suggest(child) {
                let mut new_children: Vec<AstNode> =
                    node.children().iter().map(|c| (*c).clone()).collect();
                new_children[i] = suggested_child;
                suggestions.push(node.clone().replace_children(new_children));
            }
        }

        suggestions
    }
}

/// A suggester for factorization and expansion
/// (a + b) * (a - b) <-> a*a - b*b
pub struct FactorizationSuggester;

impl RewriteSuggester for FactorizationSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // (a + b) * (a - b) -> a*a - b*b
        if let AstNode::Mul(left, right) = node {
            // Check if left is (a + b) and right is (a - b)
            if let (AstNode::Add(a1, b1), AstNode::Add(a2, b2_neg)) = (&**left, &**right) {
                if let AstNode::Neg(b2) = &**b2_neg {
                    if a1 == a2 && b1 == b2 {
                        // (a + b) * (a - b) -> a*a - b*b
                        suggestions.push(
                            AstNode::Mul(a1.clone(), a1.clone())
                                + AstNode::Neg(Box::new(AstNode::Mul(b1.clone(), b1.clone()))),
                        );
                    }
                }
            }
        }

        // a*a - b*b -> (a + b) * (a - b)
        if let AstNode::Add(left, right) = node {
            if let AstNode::Neg(b_squared) = &**right {
                if let (AstNode::Mul(a1, a2), AstNode::Mul(b1, b2)) = (&**left, &**b_squared) {
                    if a1 == a2 && b1 == b2 {
                        // a*a - b*b -> (a + b) * (a - b)
                        suggestions.push(AstNode::Mul(
                            Box::new(AstNode::Add(a1.clone(), b1.clone())),
                            Box::new(AstNode::Add(a1.clone(), Box::new(AstNode::Neg(b1.clone())))),
                        ));
                    }
                }
            }
        }

        // Recursively suggest in children
        for (i, child) in node.children().iter().enumerate() {
            for suggested_child in self.suggest(child) {
                let mut new_children: Vec<AstNode> =
                    node.children().iter().map(|c| (*c).clone()).collect();
                new_children[i] = suggested_child;
                suggestions.push(node.clone().replace_children(new_children));
            }
        }

        suggestions
    }
}

/// A suggester for distributive and associative laws
pub struct AlgebraicLawSuggester;

impl RewriteSuggester for AlgebraicLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Distributive law: a * (b + c) <-> a*b + a*c
        if let AstNode::Mul(a, bc) = node {
            if let AstNode::Add(b, c) = &**bc {
                // a * (b + c) -> a*b + a*c
                suggestions
                    .push(AstNode::Mul(a.clone(), b.clone()) + AstNode::Mul(a.clone(), c.clone()));
            }
        }

        // Reverse distributive: a*b + a*c -> a * (b + c)
        if let AstNode::Add(left, right) = node {
            if let (AstNode::Mul(a1, b), AstNode::Mul(a2, c)) = (&**left, &**right) {
                if a1 == a2 {
                    // a*b + a*c -> a * (b + c)
                    suggestions.push(AstNode::Mul(
                        a1.clone(),
                        Box::new(AstNode::Add(b.clone(), c.clone())),
                    ));
                }
            }
        }

        // Associative law for addition: (a + b) + c <-> a + (b + c)
        if let AstNode::Add(ab, c) = node {
            if let AstNode::Add(a, b) = &**ab {
                // (a + b) + c -> a + (b + c)
                suggestions.push(AstNode::Add(
                    a.clone(),
                    Box::new(AstNode::Add(b.clone(), c.clone())),
                ));
            }
        }

        if let AstNode::Add(a, bc) = node {
            if let AstNode::Add(b, c) = &**bc {
                // a + (b + c) -> (a + b) + c
                suggestions.push(AstNode::Add(
                    Box::new(AstNode::Add(a.clone(), b.clone())),
                    c.clone(),
                ));
            }
        }

        // Associative law for multiplication: (a * b) * c <-> a * (b * c)
        if let AstNode::Mul(ab, c) = node {
            if let AstNode::Mul(a, b) = &**ab {
                // (a * b) * c -> a * (b * c)
                suggestions.push(AstNode::Mul(
                    a.clone(),
                    Box::new(AstNode::Mul(b.clone(), c.clone())),
                ));
            }
        }

        if let AstNode::Mul(a, bc) = node {
            if let AstNode::Mul(b, c) = &**bc {
                // a * (b * c) -> (a * b) * c
                suggestions.push(AstNode::Mul(
                    Box::new(AstNode::Mul(a.clone(), b.clone())),
                    c.clone(),
                ));
            }
        }

        // Recursively suggest in children
        for (i, child) in node.children().iter().enumerate() {
            for suggested_child in self.suggest(child) {
                let mut new_children: Vec<AstNode> =
                    node.children().iter().map(|c| (*c).clone()).collect();
                new_children[i] = suggested_child;
                suggestions.push(node.clone().replace_children(new_children));
            }
        }

        suggestions
    }
}

/// A suggester that proposes equivalent loop transformations.
/// For example, suggests loop unrolling or loop fusion.
pub struct LoopTransformSuggester;

impl RewriteSuggester for LoopTransformSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Suggest loop unrolling for small loops
        if let AstNode::Range {
            counter_name,
            max,
            body,
        } = node
        {
            // If max is a small constant, suggest unrolling
            if let AstNode::Const(lit) = max.as_ref() {
                let max_val = match lit {
                    crate::ast::ConstLiteral::Usize(u) => *u,
                    crate::ast::ConstLiteral::Isize(i) => *i as usize,
                    crate::ast::ConstLiteral::F32(f) => *f as usize,
                };

                // Only unroll loops with iteration count <= 8
                if max_val > 0 && max_val <= 8 {
                    let mut unrolled_statements = Vec::new();
                    for i in 0..max_val {
                        // Replace counter variable with the iteration index
                        let counter_value = AstNode::from(i);
                        let body_with_index = body
                            .as_ref()
                            .clone()
                            .replace_node(&AstNode::Var(counter_name.clone()), counter_value);
                        unrolled_statements.push(body_with_index);
                    }

                    // Create a block with all unrolled iterations
                    suggestions.push(AstNode::Block {
                        scope: crate::ast::Scope {
                            declarations: vec![],
                        },
                        statements: unrolled_statements,
                    });
                }
            }
        }

        // Suggest loop fusion for consecutive loops
        if let AstNode::Block { scope, statements } = node {
            for i in 0..statements.len().saturating_sub(1) {
                if let (
                    AstNode::Range {
                        counter_name: counter1,
                        max: max1,
                        body: body1,
                    },
                    AstNode::Range {
                        counter_name: counter2,
                        max: max2,
                        body: body2,
                    },
                ) = (&statements[i], &statements[i + 1])
                {
                    // Only fuse if they have the same max
                    if max1 == max2 && counter1 == counter2 {
                        let fused_body = AstNode::Block {
                            scope: crate::ast::Scope {
                                declarations: vec![],
                            },
                            statements: vec![body1.as_ref().clone(), body2.as_ref().clone()],
                        };

                        let fused_loop = AstNode::Range {
                            counter_name: counter1.clone(),
                            max: max1.clone(),
                            body: Box::new(fused_body),
                        };

                        let mut new_statements = statements.clone();
                        new_statements[i] = fused_loop;
                        new_statements.remove(i + 1);

                        suggestions.push(AstNode::Block {
                            scope: scope.clone(),
                            statements: new_statements,
                        });
                    }
                }
            }
        }

        // Recursively suggest loop transformations in children
        for (i, child) in node.children().iter().enumerate() {
            for suggested_child in self.suggest(child) {
                let mut new_children: Vec<AstNode> =
                    node.children().iter().map(|c| (*c).clone()).collect();
                new_children[i] = suggested_child;
                suggestions.push(node.clone().replace_children(new_children));
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    fn var(name: &str) -> AstNode {
        AstNode::Var(name.to_string())
    }

    #[test]
    fn test_inverse_operation_suggester() {
        let suggester = InverseOperationSuggester;

        // log2(exp2(x)) -> x
        let ast = AstNode::Log2(Box::new(AstNode::Exp2(Box::new(var("x")))));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("x")));

        // exp2(log2(x)) -> x
        let ast = AstNode::Exp2(Box::new(AstNode::Log2(Box::new(var("x")))));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("x")));

        // sqrt(x * x) -> x
        let ast = AstNode::Sqrt(Box::new(AstNode::Mul(
            Box::new(var("x")),
            Box::new(var("x")),
        )));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("x")));
    }

    #[test]
    fn test_factorization_suggester() {
        let suggester = FactorizationSuggester;

        // (a + b) * (a - b) -> a*a - b*b
        let a = var("a");
        let b = var("b");
        let ast = AstNode::Mul(
            Box::new(AstNode::Add(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(AstNode::Add(
                Box::new(a.clone()),
                Box::new(AstNode::Neg(Box::new(b.clone()))),
            )),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(a.clone()))),
            Box::new(AstNode::Neg(Box::new(AstNode::Mul(
                Box::new(b.clone()),
                Box::new(b.clone()),
            )))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_algebraic_law_suggester() {
        let suggester = AlgebraicLawSuggester;

        // Distributive: a * (b + c) -> a*b + a*c
        let a = var("a");
        let b = var("b");
        let c = var("c");
        let ast = AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Add(Box::new(b.clone()), Box::new(c.clone()))),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(c.clone()))),
        );
        assert!(suggestions.contains(&expected));

        // Reverse distributive: a*b + a*c -> a * (b + c)
        let ast = AstNode::Add(
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(AstNode::Mul(Box::new(a.clone()), Box::new(c.clone()))),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Mul(
            Box::new(a.clone()),
            Box::new(AstNode::Add(Box::new(b.clone()), Box::new(c.clone()))),
        );
        assert!(suggestions.contains(&expected));

        // Associative: (a + b) + c -> a + (b + c)
        let ast = AstNode::Add(
            Box::new(AstNode::Add(Box::new(a.clone()), Box::new(b.clone()))),
            Box::new(c.clone()),
        );
        let suggestions = suggester.suggest(&ast);

        let expected = AstNode::Add(
            Box::new(a.clone()),
            Box::new(AstNode::Add(Box::new(b.clone()), Box::new(c.clone()))),
        );
        assert!(suggestions.contains(&expected));
    }
}

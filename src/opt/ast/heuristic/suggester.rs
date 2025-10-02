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

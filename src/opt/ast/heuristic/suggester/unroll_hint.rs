use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester that proposes adding unroll hints to loops.
/// This adds #pragma unroll directives without actually unrolling the loop.
pub struct UnrollHintSuggester {
    /// Whether to suggest full unrolling (Some(0)) or partial unrolling with factors
    suggest_full: bool,
    /// Unroll factors to try (e.g., [2, 4, 8])
    unroll_factors: Vec<usize>,
}

impl UnrollHintSuggester {
    /// Create a new UnrollHintSuggester with default settings
    pub fn new() -> Self {
        Self {
            suggest_full: true,
            unroll_factors: vec![2, 4, 8],
        }
    }

    /// Create a suggester that only suggests full unrolling
    pub fn full_only() -> Self {
        Self {
            suggest_full: true,
            unroll_factors: vec![],
        }
    }

    /// Create a suggester with custom unroll factors
    pub fn with_factors(factors: Vec<usize>) -> Self {
        Self {
            suggest_full: false,
            unroll_factors: factors,
        }
    }

    /// Create a suggester with both full and partial unrolling
    pub fn with_full_and_factors(factors: Vec<usize>) -> Self {
        Self {
            suggest_full: true,
            unroll_factors: factors,
        }
    }
}

impl Default for UnrollHintSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteSuggester for UnrollHintSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Suggest unroll hints for Range nodes that don't already have one
        if let AstNode::Range {
            counter_name,
            start,
            max,
            step,
            body,
            unroll,
        } = node
        {
            // Only suggest if no unroll hint is set
            if unroll.is_none() {
                // Suggest full unrolling if enabled
                if self.suggest_full {
                    suggestions.push(AstNode::Range {
                        counter_name: counter_name.clone(),
                        start: start.clone(),
                        max: max.clone(),
                        step: step.clone(),
                        body: body.clone(),
                        unroll: Some(0),
                    });
                }

                // Suggest partial unrolling with different factors
                for &factor in &self.unroll_factors {
                    suggestions.push(AstNode::Range {
                        counter_name: counter_name.clone(),
                        start: start.clone(),
                        max: max.clone(),
                        step: step.clone(),
                        body: body.clone(),
                        unroll: Some(factor),
                    });
                }
            }
        }

        // Recursively suggest unroll hints in children
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
    use crate::ast::ConstLiteral;

    #[test]
    fn test_unroll_hint_suggester() {
        let suggester = UnrollHintSuggester::new();

        let loop_node = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(AstNode::var("x")),
            unroll: None,
        };

        let suggestions = suggester.suggest(&loop_node);

        // Should suggest full unroll (Some(0)) + 3 partial unrolls (2, 4, 8)
        assert_eq!(suggestions.len(), 4);

        // Check that we have full unroll suggestion
        assert!(suggestions.iter().any(|s| {
            if let AstNode::Range { unroll, .. } = s {
                *unroll == Some(0)
            } else {
                false
            }
        }));

        // Check that we have factor 2, 4, 8 suggestions
        for factor in [2, 4, 8] {
            assert!(suggestions.iter().any(|s| {
                if let AstNode::Range { unroll, .. } = s {
                    *unroll == Some(factor)
                } else {
                    false
                }
            }));
        }
    }

    #[test]
    fn test_unroll_hint_suggester_full_only() {
        let suggester = UnrollHintSuggester::full_only();

        let loop_node = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(AstNode::var("x")),
            unroll: None,
        };

        let suggestions = suggester.suggest(&loop_node);

        // Should only suggest full unroll
        assert_eq!(suggestions.len(), 1);
        if let AstNode::Range { unroll, .. } = &suggestions[0] {
            assert_eq!(*unroll, Some(0));
        }
    }

    #[test]
    fn test_unroll_hint_suggester_with_factors() {
        let suggester = UnrollHintSuggester::with_factors(vec![3, 5]);

        let loop_node = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(AstNode::var("x")),
            unroll: None,
        };

        let suggestions = suggester.suggest(&loop_node);

        // Should suggest factors 3 and 5 only (no full unroll)
        assert_eq!(suggestions.len(), 2);
        assert!(suggestions.iter().any(|s| {
            if let AstNode::Range { unroll, .. } = s {
                *unroll == Some(3)
            } else {
                false
            }
        }));
        assert!(suggestions.iter().any(|s| {
            if let AstNode::Range { unroll, .. } = s {
                *unroll == Some(5)
            } else {
                false
            }
        }));
    }

    #[test]
    fn test_no_suggestion_if_already_unrolled() {
        let suggester = UnrollHintSuggester::new();

        let loop_node = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(AstNode::var("x")),
            unroll: Some(4), // Already has unroll hint
        };

        let suggestions = suggester.suggest(&loop_node);

        // Should not suggest anything for the loop itself
        // (only recursive suggestions for children, which is empty in this case)
        assert_eq!(suggestions.len(), 0);
    }
}

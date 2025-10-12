use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester that proposes adding unroll hints to loops.
/// This adds #pragma unroll directives without actually unrolling the loop.
pub struct UnrollHintSuggester {
    /// Unroll factor to use for suggestions
    unroll_factor: usize,
}

impl UnrollHintSuggester {
    /// Create a new UnrollHintSuggester with the specified unroll factor
    pub fn new(unroll_factor: usize) -> Self {
        Self { unroll_factor }
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
                suggestions.push(AstNode::Range {
                    counter_name: counter_name.clone(),
                    start: start.clone(),
                    max: max.clone(),
                    step: step.clone(),
                    body: body.clone(),
                    unroll: Some(self.unroll_factor),
                });
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
    use crate::ast::helper::var;
    use crate::ast::ConstLiteral;

    #[test]
    fn test_unroll_hint_suggester() {
        let suggester = UnrollHintSuggester::new(4);

        let loop_node = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(var("x")),
            unroll: None,
        };

        let suggestions = suggester.suggest(&loop_node);

        // Should suggest one unroll with factor 4
        assert_eq!(suggestions.len(), 1);
        if let AstNode::Range { unroll, .. } = &suggestions[0] {
            assert_eq!(*unroll, Some(4));
        }
    }

    #[test]
    fn test_no_suggestion_if_already_unrolled() {
        let suggester = UnrollHintSuggester::new(4);

        let loop_node = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(var("x")),
            unroll: Some(4), // Already has unroll hint
        };

        let suggestions = suggester.suggest(&loop_node);

        // Should not suggest anything for the loop itself
        // (only recursive suggestions for children, which is empty in this case)
        assert_eq!(suggestions.len(), 0);
    }
}

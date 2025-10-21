use crate::ast::AstNode;
use crate::opt::ast::RewriteSuggester;

/// A suggester that fully unrolls loops with constant iteration counts.
/// Converts Range loops into a Block with multiple statements.
pub struct LoopUnrollSuggester {
    /// Maximum number of iterations to unroll (prevents excessive code bloat)
    max_unroll_count: usize,
}

impl LoopUnrollSuggester {
    /// Create a new LoopUnrollSuggester with the default max unroll count (32)
    pub fn new() -> Self {
        Self {
            max_unroll_count: 32,
        }
    }

    /// Create a LoopUnrollSuggester with a custom max unroll count
    pub fn with_max_count(max_unroll_count: usize) -> Self {
        Self { max_unroll_count }
    }

    /// Replace all occurrences of a variable with a constant value in an AST node
    fn replace_var_with_const(node: &AstNode, var_name: &str, value: isize) -> AstNode {
        match node {
            AstNode::Var(name) if name == var_name => {
                AstNode::Const(crate::ast::ConstLiteral::Isize(value))
            }
            // Recursively replace in all children
            _ => {
                let children: Vec<AstNode> = node
                    .children()
                    .iter()
                    .map(|child| Self::replace_var_with_const(child, var_name, value))
                    .collect();
                if children.is_empty() {
                    node.clone()
                } else {
                    node.clone().replace_children(children)
                }
            }
        }
    }
}

impl Default for LoopUnrollSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl RewriteSuggester for LoopUnrollSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Detect Range loops with constant bounds
        if let AstNode::Range {
            counter_name,
            start,
            max,
            step,
            body,
            unroll,
        } = node
        {
            // Skip if already has an unroll hint (defer to compiler)
            if unroll.is_some() {
                return suggestions;
            }

            // Check if start, max, and step are all constants
            if let (
                AstNode::Const(crate::ast::ConstLiteral::Isize(start_val)),
                AstNode::Const(crate::ast::ConstLiteral::Isize(max_val)),
                AstNode::Const(crate::ast::ConstLiteral::Isize(step_val)),
            ) = (&**start, &**max, &**step)
            {
                // Calculate iteration count
                if *step_val <= 0 {
                    // Invalid or infinite loop, skip
                    return suggestions;
                }

                let iteration_count = ((*max_val - *start_val) / *step_val) as usize;

                // Only unroll if iteration count is within reasonable limits
                if iteration_count > 0 && iteration_count <= self.max_unroll_count {
                    // Generate unrolled statements
                    let mut statements = Vec::new();

                    for i in 0..iteration_count {
                        let counter_value = *start_val + (i as isize) * *step_val;
                        let unrolled_body =
                            Self::replace_var_with_const(body, counter_name, counter_value);
                        statements.push(unrolled_body);
                    }

                    // Create a Block node with all unrolled iterations
                    let unrolled = AstNode::Block {
                        scope: crate::ast::Scope {
                            declarations: vec![],
                        },
                        statements,
                    };

                    suggestions.push(unrolled);
                }
            }
        }

        // Recursively suggest unrolling in children
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
    use crate::ast::{ConstLiteral, RangeBuilder};

    #[test]
    fn test_simple_loop_unroll() {
        let suggester = LoopUnrollSuggester::new();

        // for(i=0; i<4; i++) { x = i }
        let loop_node = RangeBuilder::new(
            "i".to_string(),
            AstNode::Const(ConstLiteral::Isize(4)),
            AstNode::Assign("x".to_string(), Box::new(AstNode::Var("i".to_string()))),
        )
        .build();

        let suggestions = suggester.suggest(&loop_node);
        assert_eq!(suggestions.len(), 1);

        // Should be a Block with 4 statements
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 4);

            // Check that each statement has the correct constant
            for (idx, stmt) in statements.iter().enumerate() {
                if let AstNode::Assign(_, value) = stmt {
                    assert_eq!(**value, AstNode::Const(ConstLiteral::Isize(idx as isize)));
                }
            }
        } else {
            panic!("Expected Block node");
        }
    }

    #[test]
    fn test_loop_with_step() {
        let suggester = LoopUnrollSuggester::new();

        // for(i=0; i<10; i+=2) { x = i }
        let loop_node = RangeBuilder::new(
            "i".to_string(),
            AstNode::Const(ConstLiteral::Isize(10)),
            AstNode::Assign("x".to_string(), Box::new(AstNode::Var("i".to_string()))),
        )
        .step(AstNode::Const(ConstLiteral::Isize(2)))
        .build();

        let suggestions = suggester.suggest(&loop_node);
        assert_eq!(suggestions.len(), 1);

        // Should be a Block with 5 statements (0, 2, 4, 6, 8)
        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 5);

            for (idx, stmt) in statements.iter().enumerate() {
                if let AstNode::Assign(_, value) = stmt {
                    assert_eq!(
                        **value,
                        AstNode::Const(ConstLiteral::Isize((idx * 2) as isize))
                    );
                }
            }
        } else {
            panic!("Expected Block node");
        }
    }

    #[test]
    fn test_no_unroll_for_large_loops() {
        let suggester = LoopUnrollSuggester::new();

        // for(i=0; i<100; i++) { ... } - too large to unroll
        let loop_node = RangeBuilder::new(
            "i".to_string(),
            AstNode::Const(ConstLiteral::Isize(100)),
            AstNode::Assign("x".to_string(), Box::new(AstNode::Var("i".to_string()))),
        )
        .build();

        let suggestions = suggester.suggest(&loop_node);
        // Should not suggest unrolling for large loops
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_no_unroll_for_non_constant_bounds() {
        let suggester = LoopUnrollSuggester::new();

        // for(i=0; i<n; i++) { ... } - variable bound
        let loop_node = RangeBuilder::new(
            "i".to_string(),
            AstNode::Var("n".to_string()),
            AstNode::Assign("x".to_string(), Box::new(AstNode::Var("i".to_string()))),
        )
        .build();

        let suggestions = suggester.suggest(&loop_node);
        // Should not suggest unrolling for non-constant bounds
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_no_unroll_if_already_has_hint() {
        let suggester = LoopUnrollSuggester::new();

        // Loop already has unroll hint
        let loop_node = RangeBuilder::new(
            "i".to_string(),
            AstNode::Const(ConstLiteral::Isize(4)),
            AstNode::Assign("x".to_string(), Box::new(AstNode::Var("i".to_string()))),
        )
        .unroll_by(4)
        .build();

        let suggestions = suggester.suggest(&loop_node);
        // Should not suggest unrolling if already has hint
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_complex_body_unroll() {
        let suggester = LoopUnrollSuggester::new();

        // for(i=0; i<3; i++) { y = x + i * 2 }
        let loop_node = RangeBuilder::new(
            "i".to_string(),
            AstNode::Const(ConstLiteral::Isize(3)),
            AstNode::Assign(
                "y".to_string(),
                Box::new(
                    AstNode::Var("x".to_string())
                        + AstNode::Var("i".to_string()) * AstNode::Const(ConstLiteral::Isize(2)),
                ),
            ),
        )
        .build();

        let suggestions = suggester.suggest(&loop_node);
        assert_eq!(suggestions.len(), 1);

        if let AstNode::Block { statements, .. } = &suggestions[0] {
            assert_eq!(statements.len(), 3);

            // First iteration: y = x + 0 * 2
            if let AstNode::Assign(_, value) = &statements[0] {
                if let AstNode::Add(left, right) = &**value {
                    assert!(matches!(**left, AstNode::Var(_)));
                    if let AstNode::Mul(i_val, _two) = &**right {
                        assert_eq!(**i_val, AstNode::Const(ConstLiteral::Isize(0)));
                    }
                }
            }

            // Second iteration: y = x + 1 * 2
            if let AstNode::Assign(_, value) = &statements[1] {
                if let AstNode::Add(_, right) = &**value {
                    if let AstNode::Mul(i_val, _) = &**right {
                        assert_eq!(**i_val, AstNode::Const(ConstLiteral::Isize(1)));
                    }
                }
            }
        } else {
            panic!("Expected Block node");
        }
    }
}

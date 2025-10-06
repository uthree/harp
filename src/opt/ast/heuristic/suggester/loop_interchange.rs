use crate::ast::AstNode;
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester that proposes loop interchange transformations.
/// Loop interchange swaps the order of nested loops to potentially improve cache locality.
///
/// This suggester only applies to simple nested loops of the form:
/// ```text
/// for i in range1 {
///     for j in range2 {
///         body
///     }
/// }
/// ```
///
/// It will NOT apply to loops with mixed statements like:
/// ```text
/// for i in range1 {
///     statement1;
///     for j in range2 {
///         body
///     }
/// }
/// ```
pub struct LoopInterchangeSuggester;

impl RewriteSuggester for LoopInterchangeSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Check if this is a Range node with another Range as its body
        if let AstNode::Range {
            counter_name: outer_counter,
            start: outer_start,
            max: outer_max,
            step: outer_step,
            body: outer_body,
            unroll: outer_unroll,
        } = node
        {
            // Check if the body is a single Range node (not a Block with statements)
            if let AstNode::Range {
                counter_name: inner_counter,
                start: inner_start,
                max: inner_max,
                step: inner_step,
                body: inner_body,
                unroll: inner_unroll,
            } = outer_body.as_ref()
            {
                // Create the interchanged version: swap outer and inner loops
                let new_inner_loop = AstNode::Range {
                    counter_name: outer_counter.clone(),
                    start: outer_start.clone(),
                    max: outer_max.clone(),
                    step: outer_step.clone(),
                    body: inner_body.clone(),
                    unroll: *outer_unroll,
                };

                let new_outer_loop = AstNode::Range {
                    counter_name: inner_counter.clone(),
                    start: inner_start.clone(),
                    max: inner_max.clone(),
                    step: inner_step.clone(),
                    body: Box::new(new_inner_loop),
                    unroll: *inner_unroll,
                };

                suggestions.push(new_outer_loop);
            }
        }

        // Recursively suggest loop interchange in children
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
    use crate::ast::{self, ConstLiteral};

    #[test]
    fn test_simple_loop_interchange() {
        let suggester = LoopInterchangeSuggester;

        // Create nested loops: for(i=0; i<N; i++) { for(j=0; j<M; j++) { body } }
        let body = AstNode::Assign(
            "x".to_string(),
            Box::new(AstNode::Add(
                Box::new(AstNode::Var("i".to_string())),
                Box::new(AstNode::Var("j".to_string())),
            )),
        );

        let inner_loop = AstNode::Range {
            counter_name: "j".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Var("M".to_string())),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(body.clone()),
            unroll: None,
        };

        let outer_loop = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Var("N".to_string())),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(inner_loop),
            unroll: None,
        };

        let suggestions = suggester.suggest(&outer_loop);

        // Should suggest one interchange
        assert_eq!(suggestions.len(), 1);

        // Verify the interchanged structure
        if let AstNode::Range {
            counter_name: new_outer_counter,
            body: new_outer_body,
            ..
        } = &suggestions[0]
        {
            assert_eq!(new_outer_counter, "j");

            if let AstNode::Range {
                counter_name: new_inner_counter,
                body: new_inner_body,
                ..
            } = new_outer_body.as_ref()
            {
                assert_eq!(new_inner_counter, "i");
                assert_eq!(new_inner_body.as_ref(), &body);
            } else {
                panic!("Inner loop should be a Range");
            }
        } else {
            panic!("Suggested node should be a Range");
        }
    }

    #[test]
    fn test_no_interchange_for_block_body() {
        let suggester = LoopInterchangeSuggester;

        // Create a loop with a Block containing statements and a nested loop
        // This should NOT be interchanged
        let inner_loop = AstNode::Range {
            counter_name: "j".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(AstNode::Assign(
                "y".to_string(),
                Box::new(AstNode::Var("j".to_string())),
            )),
            unroll: None,
        };

        let block_with_loop = AstNode::Block {
            scope: ast::Scope {
                declarations: vec![],
            },
            statements: vec![
                AstNode::Assign("x".to_string(), Box::new(AstNode::Var("i".to_string()))),
                inner_loop,
            ],
        };

        let outer_loop = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(block_with_loop),
            unroll: None,
        };

        let suggestions = suggester.suggest(&outer_loop);

        // Should not suggest interchange at the top level (body is Block, not Range)
        // But may suggest interchange for the nested loop inside the block
        for suggestion in &suggestions {
            // The top-level loop counter should still be "i" if this is from recursive suggestion
            if let AstNode::Range { counter_name, .. } = suggestion {
                if counter_name == "j" {
                    panic!("Should not interchange when outer loop has Block body with mixed statements");
                }
            }
        }
    }

    #[test]
    fn test_interchange_preserves_unroll_hints() {
        let suggester = LoopInterchangeSuggester;

        let body = AstNode::Assign(
            "x".to_string(),
            Box::new(AstNode::Const(ConstLiteral::Isize(1))),
        );

        let inner_loop = AstNode::Range {
            counter_name: "j".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(4))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(body),
            unroll: Some(2),
        };

        let outer_loop = AstNode::Range {
            counter_name: "i".to_string(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(8))),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(inner_loop),
            unroll: Some(4),
        };

        let suggestions = suggester.suggest(&outer_loop);
        assert_eq!(suggestions.len(), 1);

        // After interchange, the unroll hints should be swapped along with the loops
        if let AstNode::Range {
            unroll: new_outer_unroll,
            body: new_outer_body,
            ..
        } = &suggestions[0]
        {
            assert_eq!(*new_outer_unroll, Some(2)); // Inner's unroll becomes outer

            if let AstNode::Range {
                unroll: new_inner_unroll,
                ..
            } = new_outer_body.as_ref()
            {
                assert_eq!(*new_inner_unroll, Some(4)); // Outer's unroll becomes inner
            }
        }
    }
}

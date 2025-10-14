use crate::ast::AstNode;
use crate::ast::RangeBuilder;
use crate::opt::ast::heuristic::RewriteSuggester;

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
            ..
        } = node
        {
            // If max is a small constant, suggest unrolling
            if let AstNode::Const(lit) = max.as_ref() {
                let max_val = match lit {
                    crate::ast::ConstLiteral::Usize(u) => *u,
                    crate::ast::ConstLiteral::Isize(i) => *i as usize,
                    crate::ast::ConstLiteral::F32(f) => *f as usize,
                    crate::ast::ConstLiteral::Bool(b) => {
                        if *b {
                            1
                        } else {
                            0
                        }
                    }
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
                        ..
                    },
                    AstNode::Range {
                        counter_name: counter2,
                        max: max2,
                        body: body2,
                        ..
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

                        let fused_loop =
                            RangeBuilder::new(counter1.clone(), *max1.clone(), fused_body).build();

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

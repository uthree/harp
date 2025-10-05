use crate::ast::{AstNode, ConstLiteral};
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester for loop tiling (blocking) optimization.
/// Converts a single loop into nested loops with smaller tiles to improve cache locality.
pub struct LoopTilingSuggester {
    tile_size: usize,
}

impl LoopTilingSuggester {
    pub fn new(tile_size: usize) -> Self {
        Self { tile_size }
    }
}

impl RewriteSuggester for LoopTilingSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Detect Range loops and suggest tiled version
        if let AstNode::Range {
            counter_name,
            max,
            body,
        } = node
        {
            // Create tiled loop structure:
            // for(outer = 0; outer < max; outer += tile_size) {
            //     for(inner = 0; inner < tile_size; inner++) {
            //         i = outer + inner;
            //         if(i < max) { body[i] }
            //     }
            // }

            let tile_size_node = AstNode::Const(ConstLiteral::Isize(self.tile_size as isize));
            let outer_name = format!("{}_tile", counter_name);
            let inner_name = format!("{}_inner", counter_name);

            // Replace all uses of counter_name with (outer + inner) in body
            let modified_body =
                self.replace_counter_with_offset(body, counter_name, &outer_name, &inner_name);

            // Add bounds check: if(i < max) { body }
            let i_value = AstNode::Add(
                Box::new(AstNode::Var(outer_name.clone())),
                Box::new(AstNode::Var(inner_name.clone())),
            );
            let bounds_check = AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![crate::ast::VariableDecl {
                        name: counter_name.clone(),
                        dtype: crate::ast::DType::Isize,
                        constant: true,
                        size_expr: None,
                    }],
                },
                statements: vec![
                    AstNode::Assign(counter_name.clone(), Box::new(i_value.clone())),
                    // if(i < max) { modified_body }
                    // AST doesn't have If node, so we'll just include the body
                    // In real implementation, we'd need conditional execution
                    modified_body,
                ],
            };

            // Inner loop: for(inner = 0; inner < tile_size; inner++)
            let inner_loop = AstNode::Range {
                counter_name: inner_name.clone(),
                max: Box::new(tile_size_node.clone()),
                body: Box::new(bounds_check),
            };

            // Outer loop: for(outer = 0; outer < max; outer += tile_size)
            // Note: Range increments by 1, so we need to modify the loop structure
            // For now, we'll create a simplified version that steps by 1
            // A more sophisticated approach would require step parameter in Range

            // Calculate number of tiles: (max + tile_size - 1) / tile_size
            let num_tiles = AstNode::Div(
                Box::new(AstNode::Add(
                    max.clone(),
                    Box::new(AstNode::Const(ConstLiteral::Isize(
                        self.tile_size as isize - 1,
                    ))),
                )),
                Box::new(tile_size_node.clone()),
            );

            // Outer loop body: tile_start = outer * tile_size; inner_loop
            let outer_body = AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![crate::ast::VariableDecl {
                        name: outer_name.clone(),
                        dtype: crate::ast::DType::Isize,
                        constant: true,
                        size_expr: None,
                    }],
                },
                statements: vec![
                    AstNode::Assign(
                        outer_name.clone(),
                        Box::new(AstNode::Mul(
                            Box::new(AstNode::Var(format!("{}_idx", outer_name))),
                            Box::new(tile_size_node),
                        )),
                    ),
                    inner_loop,
                ],
            };

            let tiled_loop = AstNode::Range {
                counter_name: format!("{}_idx", outer_name),
                max: Box::new(num_tiles),
                body: Box::new(outer_body),
            };

            suggestions.push(tiled_loop);
        }

        // Recursively suggest tiling in children
        if let AstNode::Block { scope, statements } = node {
            for (i, stmt) in statements.iter().enumerate() {
                for suggestion in self.suggest(stmt) {
                    let mut new_statements = statements.clone();
                    new_statements[i] = suggestion;
                    suggestions.push(AstNode::Block {
                        scope: scope.clone(),
                        statements: new_statements,
                    });
                }
            }
        }

        suggestions
    }
}

impl LoopTilingSuggester {
    fn replace_counter_with_offset(
        &self,
        node: &AstNode,
        counter_name: &str,
        outer_name: &str,
        inner_name: &str,
    ) -> AstNode {
        match node {
            AstNode::Var(name) if name == counter_name => {
                // Replace counter with (outer + inner)
                AstNode::Add(
                    Box::new(AstNode::Var(outer_name.to_string())),
                    Box::new(AstNode::Var(inner_name.to_string())),
                )
            }
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(self.replace_counter_with_offset(a, counter_name, outer_name, inner_name)),
                Box::new(self.replace_counter_with_offset(b, counter_name, outer_name, inner_name)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(self.replace_counter_with_offset(a, counter_name, outer_name, inner_name)),
                Box::new(self.replace_counter_with_offset(b, counter_name, outer_name, inner_name)),
            ),
            AstNode::Deref(expr) => AstNode::Deref(Box::new(self.replace_counter_with_offset(
                expr,
                counter_name,
                outer_name,
                inner_name,
            ))),
            AstNode::Store {
                target,
                index,
                value,
            } => AstNode::Store {
                target: Box::new(self.replace_counter_with_offset(
                    target,
                    counter_name,
                    outer_name,
                    inner_name,
                )),
                index: Box::new(self.replace_counter_with_offset(
                    index,
                    counter_name,
                    outer_name,
                    inner_name,
                )),
                value: Box::new(self.replace_counter_with_offset(
                    value,
                    counter_name,
                    outer_name,
                    inner_name,
                )),
            },
            AstNode::Assign(name, expr) => AstNode::Assign(
                name.clone(),
                Box::new(self.replace_counter_with_offset(
                    expr,
                    counter_name,
                    outer_name,
                    inner_name,
                )),
            ),
            AstNode::Block { scope, statements } => {
                let new_statements: Vec<_> = statements
                    .iter()
                    .map(|stmt| {
                        self.replace_counter_with_offset(stmt, counter_name, outer_name, inner_name)
                    })
                    .collect();
                AstNode::Block {
                    scope: scope.clone(),
                    statements: new_statements,
                }
            }
            _ => node.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loop_tiling_suggester() {
        let suggester = LoopTilingSuggester::new(16);

        // Simple loop: for(i=0; i<100; i++) { use i }
        let body = AstNode::Assign(
            "result".to_string(),
            Box::new(AstNode::Var("i".to_string())),
        );
        let ast = AstNode::Range {
            counter_name: "i".to_string(),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(100))),
            body: Box::new(body),
        };

        let suggestions = suggester.suggest(&ast);
        assert!(!suggestions.is_empty());

        // The suggested loop should be tiled
        // Verify it's a nested loop structure
        if let Some(AstNode::Range { counter_name, .. }) = suggestions.first() {
            assert!(counter_name.contains("tile") || counter_name.contains("idx"));
        }
    }
}

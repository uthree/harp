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
            ..
        } = node
        {
            // Skip if already tiled (avoid infinite tiling)
            if counter_name.contains("_tile")
                || counter_name.contains("_inner")
                || counter_name.contains("_idx")
            {
                return suggestions;
            }

            // Skip tiling for very small loops (less than tile size)
            // Allow tiling for loops >= tile_size for better cache behavior
            if let AstNode::Const(crate::ast::ConstLiteral::Isize(max_val)) = **max {
                if max_val < self.tile_size as isize {
                    return suggestions;
                }
            }
            // Create tiled loop structure with separate remainder loop:
            // main_max = max - max % tile_size
            // for(tile_start = 0; tile_start < main_max; tile_start += tile_size) {
            //     for(i = tile_start; i < tile_start + tile_size; i++) {
            //         body[i]
            //     }
            // }
            // // Remainder loop for last partial tile
            // for(i = main_max; i < max; i++) {
            //     body[i]
            // }

            let tile_size_node = AstNode::Const(ConstLiteral::Isize(self.tile_size as isize));
            let tile_start_name = format!("{}_tile_start", counter_name);
            let main_max_name = format!("{}_main_max", counter_name);

            // Calculate main_max = max - max % tile_size
            let main_max_value = AstNode::Add(
                max.clone(),
                Box::new(AstNode::Neg(Box::new(AstNode::Rem(
                    max.clone(),
                    Box::new(tile_size_node.clone()),
                )))),
            );

            // Inner tiled loop: for(i = tile_start; i < tile_start + tile_size; i++)
            let inner_tiled_loop = AstNode::Range {
                counter_name: counter_name.clone(),
                start: Box::new(AstNode::Var(tile_start_name.clone())),
                max: Box::new(AstNode::Add(
                    Box::new(AstNode::Var(tile_start_name.clone())),
                    Box::new(tile_size_node.clone()),
                )),
                step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                body: body.clone(),
            };

            // Main tiled loop: for(tile_start = 0; tile_start < main_max; tile_start += tile_size)
            let main_tiled_loop = AstNode::Range {
                counter_name: tile_start_name.clone(),
                start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
                max: Box::new(AstNode::Var(main_max_name.clone())),
                step: Box::new(tile_size_node),
                body: Box::new(inner_tiled_loop),
            };

            // Remainder loop: for(i = main_max; i < max; i++)
            let remainder_loop = AstNode::Range {
                counter_name: counter_name.clone(),
                start: Box::new(AstNode::Var(main_max_name.clone())),
                max: max.clone(),
                step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
                body: body.clone(),
            };

            // Combine main and remainder loops in a block
            let combined = AstNode::Block {
                scope: crate::ast::Scope {
                    declarations: vec![crate::ast::VariableDecl {
                        name: main_max_name.clone(),
                        dtype: crate::ast::DType::Isize,
                        constant: false,
                        size_expr: None,
                    }],
                },
                statements: vec![
                    AstNode::Assign(main_max_name, Box::new(main_max_value)),
                    main_tiled_loop,
                    remainder_loop,
                ],
            };

            suggestions.push(combined);
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
            start: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(0))),
            max: Box::new(AstNode::Const(ConstLiteral::Isize(100))),
            step: Box::new(AstNode::Const(crate::ast::ConstLiteral::Isize(1))),
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

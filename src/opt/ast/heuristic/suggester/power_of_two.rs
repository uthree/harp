use crate::ast::{AstNode, ConstLiteral};
use crate::opt::ast::heuristic::RewriteSuggester;

/// A suggester that converts multiplication by power-of-two constants to bit shifts.
/// This optimization is beneficial because bit shifts are generally faster than multiplication.
///
/// Examples:
/// - `x * 2` -> `x << 1`
/// - `x * 4` -> `x << 2`
/// - `x * 8` -> `x << 3`
/// - `2 * x` -> `x << 1` (commutative)
pub struct PowerOfTwoSuggester;

impl PowerOfTwoSuggester {
    /// Check if a value is a power of two and return its log2 if so
    fn log2_if_power_of_two(value: isize) -> Option<isize> {
        if value <= 0 {
            return None;
        }
        let unsigned = value as usize;
        // Check if exactly one bit is set (power of 2)
        if unsigned.count_ones() == 1 {
            Some(unsigned.trailing_zeros() as isize)
        } else {
            None
        }
    }
}

impl RewriteSuggester for PowerOfTwoSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Match multiplication nodes
        if let AstNode::Mul(left, right) = node {
            // Case 1: x * constant (where constant is power of 2)
            if let AstNode::Const(ConstLiteral::Isize(val)) = **right {
                if let Some(shift_amount) = Self::log2_if_power_of_two(val) {
                    // x * 2^n -> x << n
                    suggestions.push(AstNode::Shl(
                        left.clone(),
                        Box::new(AstNode::Const(ConstLiteral::Isize(shift_amount))),
                    ));
                }
            }

            // Case 2: constant * x (where constant is power of 2)
            if let AstNode::Const(ConstLiteral::Isize(val)) = **left {
                if let Some(shift_amount) = Self::log2_if_power_of_two(val) {
                    // 2^n * x -> x << n
                    suggestions.push(AstNode::Shl(
                        right.clone(),
                        Box::new(AstNode::Const(ConstLiteral::Isize(shift_amount))),
                    ));
                }
            }

            // Case 3: x * constant (where constant is Usize power of 2)
            if let AstNode::Const(ConstLiteral::Usize(val)) = **right {
                if val > 0 && val.count_ones() == 1 {
                    let shift_amount = val.trailing_zeros() as usize;
                    suggestions.push(AstNode::Shl(
                        left.clone(),
                        Box::new(AstNode::Const(ConstLiteral::Usize(shift_amount))),
                    ));
                }
            }

            // Case 4: constant * x (where constant is Usize power of 2)
            if let AstNode::Const(ConstLiteral::Usize(val)) = **left {
                if val > 0 && val.count_ones() == 1 {
                    let shift_amount = val.trailing_zeros() as usize;
                    suggestions.push(AstNode::Shl(
                        right.clone(),
                        Box::new(AstNode::Const(ConstLiteral::Usize(shift_amount))),
                    ));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_of_two_multiplication_isize() {
        let suggester = PowerOfTwoSuggester;

        // x * 2 -> x << 1
        let x = AstNode::Var("x".to_string());
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(2))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::Shl(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(1)))
        )));

        // x * 4 -> x << 2
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(4))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::Shl(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(2)))
        )));

        // x * 8 -> x << 3
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(8))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::Shl(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(3)))
        )));
    }

    #[test]
    fn test_commutative_power_of_two() {
        let suggester = PowerOfTwoSuggester;

        // 2 * x -> x << 1
        let x = AstNode::Var("x".to_string());
        let ast = AstNode::Mul(
            Box::new(AstNode::Const(ConstLiteral::Isize(2))),
            Box::new(x.clone()),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::Shl(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(1)))
        )));
    }

    #[test]
    fn test_usize_power_of_two() {
        let suggester = PowerOfTwoSuggester;

        // x * 16usize -> x << 4
        let x = AstNode::Var("x".to_string());
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Usize(16))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::Shl(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Usize(4)))
        )));
    }

    #[test]
    fn test_non_power_of_two_no_suggestion() {
        let suggester = PowerOfTwoSuggester;

        // x * 3 (not a power of 2, should not suggest shift)
        let x = AstNode::Var("x".to_string());
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(3))),
        );
        let suggestions = suggester.suggest(&ast);
        // Should have no suggestions for the top-level node
        // (only recursive suggestions if any children matched)
        assert!(suggestions.is_empty());

        // x * 5
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(5))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_negative_power_of_two_no_suggestion() {
        let suggester = PowerOfTwoSuggester;

        // x * -2 (negative, should not suggest shift)
        let x = AstNode::Var("x".to_string());
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(-2))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_large_power_of_two() {
        let suggester = PowerOfTwoSuggester;

        // x * 1024 -> x << 10
        let x = AstNode::Var("x".to_string());
        let ast = AstNode::Mul(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(1024))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::Shl(
            Box::new(x.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(10)))
        )));
    }

    #[test]
    fn test_nested_expression() {
        let suggester = PowerOfTwoSuggester;

        // (a + b) * 4 -> (a + b) << 2
        let a = AstNode::Var("a".to_string());
        let b = AstNode::Var("b".to_string());
        let sum = AstNode::Add(Box::new(a.clone()), Box::new(b.clone()));
        let ast = AstNode::Mul(
            Box::new(sum.clone()),
            Box::new(AstNode::Const(ConstLiteral::Isize(4))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::Shl(
            Box::new(sum),
            Box::new(AstNode::Const(ConstLiteral::Isize(2)))
        )));
    }
}

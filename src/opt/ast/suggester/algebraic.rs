use crate::ast::AstNode;
use crate::opt::ast::RewriteSuggester;

/// A suggester for distributive and associative laws
pub struct AlgebraicLawSuggester;

impl AlgebraicLawSuggester {
    /// Try to fold constants in addition
    fn try_fold_add(a: &AstNode, b: &AstNode) -> Option<AstNode> {
        use crate::ast::ConstLiteral;
        match (a, b) {
            (AstNode::Const(ConstLiteral::Isize(x)), AstNode::Const(ConstLiteral::Isize(y))) => {
                Some(AstNode::Const(ConstLiteral::Isize(x + y)))
            }
            (AstNode::Const(ConstLiteral::Usize(x)), AstNode::Const(ConstLiteral::Usize(y))) => {
                Some(AstNode::Const(ConstLiteral::Usize(x + y)))
            }
            (AstNode::Const(ConstLiteral::F32(x)), AstNode::Const(ConstLiteral::F32(y))) => {
                Some(AstNode::Const(ConstLiteral::F32(x + y)))
            }
            _ => None,
        }
    }

    /// Try to fold constants in multiplication
    fn try_fold_mul(a: &AstNode, b: &AstNode) -> Option<AstNode> {
        use crate::ast::ConstLiteral;
        match (a, b) {
            (AstNode::Const(ConstLiteral::Isize(x)), AstNode::Const(ConstLiteral::Isize(y))) => {
                Some(AstNode::Const(ConstLiteral::Isize(x * y)))
            }
            (AstNode::Const(ConstLiteral::Usize(x)), AstNode::Const(ConstLiteral::Usize(y))) => {
                Some(AstNode::Const(ConstLiteral::Usize(x * y)))
            }
            (AstNode::Const(ConstLiteral::F32(x)), AstNode::Const(ConstLiteral::F32(y))) => {
                Some(AstNode::Const(ConstLiteral::F32(x * y)))
            }
            _ => None,
        }
    }
}

impl RewriteSuggester for AlgebraicLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        // Constant folding for Add
        if let AstNode::Add(a, b) = node {
            if let Some(folded) = Self::try_fold_add(a, b) {
                suggestions.push(folded);
            }
        }

        // Constant folding for Mul
        if let AstNode::Mul(a, b) = node {
            if let Some(folded) = Self::try_fold_mul(a, b) {
                suggestions.push(folded);
            }
        }

        // Identity: x + 0 -> x
        if let AstNode::Add(a, b) = node {
            if matches!(**b, AstNode::Const(crate::ast::ConstLiteral::Isize(0))) {
                suggestions.push((**a).clone());
            }
            if matches!(**a, AstNode::Const(crate::ast::ConstLiteral::Isize(0))) {
                suggestions.push((**b).clone());
            }
        }

        // Identity: x * 1 -> x
        if let AstNode::Mul(a, b) = node {
            if matches!(**b, AstNode::Const(crate::ast::ConstLiteral::Isize(1))) {
                suggestions.push((**a).clone());
            }
            if matches!(**a, AstNode::Const(crate::ast::ConstLiteral::Isize(1))) {
                suggestions.push((**b).clone());
            }
        }

        // Identity: x * 0 -> 0
        if let AstNode::Mul(a, b) = node {
            if matches!(**b, AstNode::Const(crate::ast::ConstLiteral::Isize(0))) {
                suggestions.push(AstNode::from(0isize));
            }
            if matches!(**a, AstNode::Const(crate::ast::ConstLiteral::Isize(0))) {
                suggestions.push(AstNode::from(0isize));
            }
        }

        // Double negation: -(-x) -> x
        if let AstNode::Neg(inner) = node {
            if let AstNode::Neg(x) = &**inner {
                suggestions.push((**x).clone());
            }
            // Constant negation folding
            if let AstNode::Const(c) = &**inner {
                match c {
                    crate::ast::ConstLiteral::Isize(x) => {
                        suggestions.push(AstNode::Const(crate::ast::ConstLiteral::Isize(-x)));
                    }
                    crate::ast::ConstLiteral::F32(x) => {
                        suggestions.push(AstNode::Const(crate::ast::ConstLiteral::F32(-x)));
                    }
                    _ => {}
                }
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> AstNode {
        AstNode::Var(name.to_string())
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

    #[test]
    fn test_constant_folding() {
        let suggester = AlgebraicLawSuggester;

        // 2 + 3 -> 5
        let ast = AstNode::from(2isize) + AstNode::from(3isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(5isize)));

        // 2 * 3 -> 6
        let ast = AstNode::from(2isize) * AstNode::from(3isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(6isize)));
    }

    #[test]
    fn test_identity_elimination() {
        let suggester = AlgebraicLawSuggester;
        let x = var("x");

        // x + 0 -> x
        let ast = x.clone() + AstNode::from(0isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&x));

        // x * 1 -> x
        let ast = x.clone() * AstNode::from(1isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&x));

        // x * 0 -> 0
        let ast = x.clone() * AstNode::from(0isize);
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(0isize)));
    }
}

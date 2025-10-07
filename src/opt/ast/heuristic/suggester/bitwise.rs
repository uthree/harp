use crate::ast::AstNode;
use crate::ast_pattern;
use crate::opt::ast::heuristic::{RewriteSuggester, RuleBasedSuggester};

/// A suggester for bitwise operation laws
///
/// Laws:
/// - Commutative: a & b = b & a, a | b = b | a, a ^ b = b ^ a
/// - Associative: (a & b) & c = a & (b & c), (a | b) | c = a | (b | c), (a ^ b) ^ c = a ^ (b ^ c)
/// - Distributive: a & (b | c) = (a & b) | (a & c), a | (b & c) = (a | b) & (a | c)
/// - Identity: a & 0 = 0, a | 0 = a, a ^ 0 = a, a ^ a = 0
/// - Idempotent: a & a = a, a | a = a
/// - De Morgan's: ~(a & b) = ~a | ~b, ~(a | b) = ~a & ~b
/// - Double negation: ~~a = a
/// - Absorption: a & (a | b) = a, a | (a & b) = a
pub struct BitwiseLawSuggester {
    inner: RuleBasedSuggester,
}

impl Default for BitwiseLawSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl BitwiseLawSuggester {
    pub fn new() -> Self {
        let rules = vec![
            // Commutative laws
            ast_pattern!(|a, b| a & b => b.clone() & a.clone()),
            ast_pattern!(|a, b| a | b => b.clone() | a.clone()),
            ast_pattern!(|a, b| a ^ b => b.clone() ^ a.clone()),
            // Associative laws
            ast_pattern!(|a, b, c| (a & b) & c => a.clone() & (b.clone() & c.clone())),
            ast_pattern!(|a, b, c| a & (b & c) => (a.clone() & b.clone()) & c.clone()),
            ast_pattern!(|a, b, c| (a | b) | c => a.clone() | (b.clone() | c.clone())),
            ast_pattern!(|a, b, c| a | (b | c) => (a.clone() | b.clone()) | c.clone()),
            ast_pattern!(|a, b, c| (a ^ b) ^ c => a.clone() ^ (b.clone() ^ c.clone())),
            ast_pattern!(|a, b, c| a ^ (b ^ c) => (a.clone() ^ b.clone()) ^ c.clone()),
            // Distributive laws
            ast_pattern!(|a, b, c| a & (b | c) => (a.clone() & b.clone()) | (a.clone() & c.clone())),
            ast_pattern!(|a1, b, a2, c| (a1 & b) | (a2 & c), if a1 == a2 => a1.clone() & (b.clone() | c.clone())),
            ast_pattern!(|a, b, c| a | (b & c) => (a.clone() | b.clone()) & (a.clone() | c.clone())),
            ast_pattern!(|a1, b, a2, c| (a1 | b) & (a2 | c), if a1 == a2 => a1.clone() | (b.clone() & c.clone())),
            // Identity laws
            ast_pattern!(|a| a & AstNode::from(0isize) => AstNode::from(0isize)),
            ast_pattern!(|a| AstNode::from(0isize) & a => AstNode::from(0isize)),
            ast_pattern!(|a| a | AstNode::from(0isize) => a.clone()),
            ast_pattern!(|a| AstNode::from(0isize) | a => a.clone()),
            ast_pattern!(|a| a ^ AstNode::from(0isize) => a.clone()),
            ast_pattern!(|a| AstNode::from(0isize) ^ a => a.clone()),
            // Idempotent laws (using predicates because matching the same variable twice causes move issues)
            ast_pattern!(|a, b| a & b, if a == b => a.clone()),
            ast_pattern!(|a, b| a | b, if a == b => a.clone()),
            ast_pattern!(|a, b| a ^ b, if a == b => AstNode::from(0isize)),
            // De Morgan's laws
            ast_pattern!(|a, b| !(a & b) => !a.clone() | !b.clone()),
            ast_pattern!(|a, b| !a | !b => !(a.clone() & b.clone())),
            ast_pattern!(|a, b| !(a | b) => !a.clone() & !b.clone()),
            ast_pattern!(|a, b| !a & !b => !(a.clone() | b.clone())),
            // Double negation
            ast_pattern!(|a| !!a => a.clone()),
            // Absorption laws (using predicates to match same variable)
            ast_pattern!(|a1, a2, b| a1 & (a2 | b), if a1 == a2 => a1.clone()),
            ast_pattern!(|a1, a2, b| a1 | (a2 & b), if a1 == a2 => a1.clone()),
        ];

        Self {
            inner: RuleBasedSuggester::new(rules),
        }
    }
}

impl RewriteSuggester for BitwiseLawSuggester {
    fn suggest(&self, node: &AstNode) -> Vec<AstNode> {
        self.inner.suggest(node)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn var(name: &str) -> AstNode {
        AstNode::Var(name.to_string())
    }

    #[test]
    fn test_commutative() {
        let suggester = BitwiseLawSuggester::new();

        // a & b -> b & a
        let ast = AstNode::BitAnd(Box::new(var("a")), Box::new(var("b")));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::BitAnd(Box::new(var("b")), Box::new(var("a")))));

        // a | b -> b | a
        let ast = AstNode::BitOr(Box::new(var("a")), Box::new(var("b")));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::BitOr(Box::new(var("b")), Box::new(var("a")))));

        // a ^ b -> b ^ a
        let ast = AstNode::BitXor(Box::new(var("a")), Box::new(var("b")));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::BitXor(Box::new(var("b")), Box::new(var("a")))));
    }

    #[test]
    fn test_associative() {
        let suggester = BitwiseLawSuggester::new();

        // (a & b) & c -> a & (b & c)
        let ast = AstNode::BitAnd(
            Box::new(AstNode::BitAnd(Box::new(var("a")), Box::new(var("b")))),
            Box::new(var("c")),
        );
        let suggestions = suggester.suggest(&ast);
        let expected = AstNode::BitAnd(
            Box::new(var("a")),
            Box::new(AstNode::BitAnd(Box::new(var("b")), Box::new(var("c")))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_distributive() {
        let suggester = BitwiseLawSuggester::new();

        // a & (b | c) -> (a & b) | (a & c)
        let ast = AstNode::BitAnd(
            Box::new(var("a")),
            Box::new(AstNode::BitOr(Box::new(var("b")), Box::new(var("c")))),
        );
        let suggestions = suggester.suggest(&ast);
        let expected = AstNode::BitOr(
            Box::new(AstNode::BitAnd(Box::new(var("a")), Box::new(var("b")))),
            Box::new(AstNode::BitAnd(Box::new(var("a")), Box::new(var("c")))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_identity() {
        let suggester = BitwiseLawSuggester::new();

        // a & 0 -> 0
        let ast = AstNode::BitAnd(Box::new(var("a")), Box::new(AstNode::from(0isize)));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(0isize)));

        // a | 0 -> a
        let ast = AstNode::BitOr(Box::new(var("a")), Box::new(AstNode::from(0isize)));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("a")));

        // a ^ 0 -> a
        let ast = AstNode::BitXor(Box::new(var("a")), Box::new(AstNode::from(0isize)));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("a")));

        // a ^ a -> 0
        let ast = AstNode::BitXor(Box::new(var("a")), Box::new(var("a")));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&AstNode::from(0isize)));
    }

    #[test]
    fn test_idempotent() {
        let suggester = BitwiseLawSuggester::new();

        // a & a -> a
        let ast = AstNode::BitAnd(Box::new(var("a")), Box::new(var("a")));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("a")));

        // a | a -> a
        let ast = AstNode::BitOr(Box::new(var("a")), Box::new(var("a")));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("a")));
    }

    #[test]
    fn test_de_morgan() {
        let suggester = BitwiseLawSuggester::new();

        // ~(a & b) -> ~a | ~b
        let ast = AstNode::BitNot(Box::new(AstNode::BitAnd(
            Box::new(var("a")),
            Box::new(var("b")),
        )));
        let suggestions = suggester.suggest(&ast);
        let expected = AstNode::BitOr(
            Box::new(AstNode::BitNot(Box::new(var("a")))),
            Box::new(AstNode::BitNot(Box::new(var("b")))),
        );
        assert!(suggestions.contains(&expected));

        // ~(a | b) -> ~a & ~b
        let ast = AstNode::BitNot(Box::new(AstNode::BitOr(
            Box::new(var("a")),
            Box::new(var("b")),
        )));
        let suggestions = suggester.suggest(&ast);
        let expected = AstNode::BitAnd(
            Box::new(AstNode::BitNot(Box::new(var("a")))),
            Box::new(AstNode::BitNot(Box::new(var("b")))),
        );
        assert!(suggestions.contains(&expected));
    }

    #[test]
    fn test_double_negation() {
        let suggester = BitwiseLawSuggester::new();

        // ~~a -> a
        let ast = AstNode::BitNot(Box::new(AstNode::BitNot(Box::new(var("a")))));
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("a")));
    }

    #[test]
    fn test_absorption() {
        let suggester = BitwiseLawSuggester::new();

        // a & (a | b) -> a
        let ast = AstNode::BitAnd(
            Box::new(var("a")),
            Box::new(AstNode::BitOr(Box::new(var("a")), Box::new(var("b")))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("a")));

        // a | (a & b) -> a
        let ast = AstNode::BitOr(
            Box::new(var("a")),
            Box::new(AstNode::BitAnd(Box::new(var("a")), Box::new(var("b")))),
        );
        let suggestions = suggester.suggest(&ast);
        assert!(suggestions.contains(&var("a")));
    }
}

pub mod helper;
pub mod pattern;

mod function;
mod node;
mod ops;
mod range_builder;
mod transform;
mod types;
mod variable;

// Re-export all public types and structs
pub use function::Program;
pub use node::AstNode;
pub use range_builder::RangeBuilder;
pub use types::{ConstLiteral, DType};
pub use variable::{Scope, VariableDecl};

#[cfg(test)]
mod tests {
    use super::helper::*;
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(1.0f32, AstNode::Const(ConstLiteral::F32(1.0)))]
    #[case(42usize, AstNode::Const(ConstLiteral::Usize(42)))]
    #[case(-10isize, AstNode::Const(ConstLiteral::Isize(-10)))]
    fn test_from_numeric_literals(#[case] input: impl Into<AstNode>, #[case] expected: AstNode) {
        assert_eq!(input.into(), expected);
    }

    #[test]
    fn test_addition() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a + b;
        assert_eq!(
            expr,
            AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_subtraction() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a - b;
        assert_eq!(
            expr,
            AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Neg(Box::new(AstNode::Const(ConstLiteral::F32(
                    2.0
                )))))
            )
        );
    }

    #[test]
    fn test_multiplication() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a * b;
        assert_eq!(
            expr,
            AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_division() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a / b;
        assert_eq!(
            expr,
            AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Recip(Box::new(AstNode::Const(ConstLiteral::F32(
                    2.0
                )))))
            )
        );
    }

    #[test]
    fn test_remainder() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a % b;
        assert_eq!(
            expr,
            AstNode::Rem(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_negation() {
        let a = AstNode::Var("a".to_string());
        let expr = -a;
        assert_eq!(expr, AstNode::Neg(Box::new(AstNode::Var("a".to_string()))));
    }

    #[test]
    fn test_bitwise_ops() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2isize.into();

        // BitAnd
        let expr = a.clone() & b.clone();
        assert_eq!(
            expr,
            AstNode::BitAnd(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // BitOr
        let expr = a.clone() | b.clone();
        assert_eq!(
            expr,
            AstNode::BitOr(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // BitXor
        let expr = a.clone() ^ b.clone();
        assert_eq!(
            expr,
            AstNode::BitXor(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // Shl
        let expr = a.clone() << b.clone();
        assert_eq!(
            expr,
            AstNode::Shl(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // Shr
        let expr = a.clone() >> b.clone();
        assert_eq!(
            expr,
            AstNode::Shr(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // BitNot
        let expr = !a.clone();
        assert_eq!(
            expr,
            AstNode::BitNot(Box::new(AstNode::Var("a".to_string())))
        );
    }

    #[test]
    fn test_unary_ops() {
        let a = AstNode::Var("a".to_string());
        assert_eq!(
            a.clone().recip(),
            AstNode::Recip(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().sin(),
            AstNode::Sin(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().sqrt(),
            AstNode::Sqrt(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().exp2(),
            AstNode::Exp2(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.log2(),
            AstNode::Log2(Box::new(AstNode::Var("a".to_string())))
        );
    }

    #[test]
    fn test_complex_expression() {
        let a = AstNode::Var("a".to_string());
        let b = AstNode::Var("b".to_string());
        let c = 3.0f32;
        // -(a + b) * c
        let expr = -(a.clone() + b.clone()) * c;
        assert_eq!(
            expr,
            AstNode::Mul(
                Box::new(AstNode::Neg(Box::new(AstNode::Add(
                    Box::new(AstNode::Var("a".to_string())),
                    Box::new(AstNode::Var("b".to_string()))
                )))),
                Box::new(AstNode::Const(ConstLiteral::F32(3.0)))
            )
        );
    }

    #[test]
    fn test_replace_node() {
        let a = AstNode::Var("a".to_string());
        let target = AstNode::Const(ConstLiteral::F32(1.0));
        let replacement = AstNode::Const(ConstLiteral::F32(2.0));

        // a + 1.0 -> a + 2.0
        let expr = a.clone() + target.clone();
        let result = expr.replace_node(&target, replacement.clone());
        assert_eq!(result, a + replacement);
    }

    #[test]
    fn test_replace_if() {
        let a = AstNode::Var("a".to_string());
        let expr = a.clone() + AstNode::Const(ConstLiteral::F32(0.0));

        // Replace any addition with 0 with just the left operand
        let result = expr.replace_if(
            |node| matches!(node, AstNode::Add(_, r) if **r == AstNode::Const(ConstLiteral::F32(0.0))),
            |node| if let AstNode::Add(l, _) = node { *l } else { node }
        );

        assert_eq!(result, a);
    }

    #[test]
    fn test_helper_functions() {
        // Test var
        let v = var("x");
        assert_eq!(v, AstNode::Var("x".to_string()));

        // Test const_val
        let c = const_val(ConstLiteral::F32(1.0));
        assert_eq!(c, AstNode::Const(ConstLiteral::F32(1.0)));

        // Test assign
        let assign_node = assign("x", 1.0f32);
        assert_eq!(
            assign_node,
            AstNode::Assign(
                "x".to_string(),
                Box::new(AstNode::Const(ConstLiteral::F32(1.0)))
            )
        );

        // Test max
        let max_node = max(1.0f32, 2.0f32);
        assert_eq!(
            max_node,
            AstNode::Max(
                Box::new(AstNode::Const(ConstLiteral::F32(1.0))),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );

        // Test cast
        let cast_node = cast(DType::F32, 1isize);
        assert_eq!(
            cast_node,
            AstNode::Cast {
                dtype: DType::F32,
                expr: Box::new(AstNode::Const(ConstLiteral::Isize(1)))
            }
        );
    }

    #[test]
    fn test_range_builder() {
        // Test simple range with defaults
        let r1 = range("i", 10isize, var("x"));
        assert_eq!(
            r1,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: None,
            }
        );

        // Test range builder with custom start and step
        let r2 = range_builder("i", 100isize, var("x"))
            .start(5isize)
            .step(2isize)
            .build();
        assert_eq!(
            r2,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(5))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(100))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(2))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: None,
            }
        );

        // Test range builder with full unroll
        let r3 = range_builder("i", 10isize, var("x")).unroll().build();
        assert_eq!(
            r3,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: Some(0),
            }
        );

        // Test range builder with specific unroll factor
        let r4 = range_builder("i", 100isize, var("x")).unroll_by(4).build();
        assert_eq!(
            r4,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(100))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: Some(4),
            }
        );
    }

    #[test]
    fn test_block_helpers() {
        // Test block_with_statements
        let block_node = block_with_statements(vec![assign("x", 1.0f32), assign("y", 2.0f32)]);

        assert!(matches!(block_node, AstNode::Block { .. }));
        if let AstNode::Block { scope, statements } = block_node {
            assert_eq!(scope.declarations.len(), 0);
            assert_eq!(statements.len(), 2);
        }
    }

    #[test]
    fn test_other_helpers() {
        // Test store
        let store_node = store(var("arr"), 0isize, 1.0f32);
        assert!(matches!(store_node, AstNode::Store { .. }));

        // Test deref
        let deref_node = deref(var("ptr"));
        assert_eq!(
            deref_node,
            AstNode::Deref(Box::new(AstNode::Var("ptr".to_string())))
        );

        // Test call
        let call_node = call("foo", vec![var("x"), const_val(ConstLiteral::F32(1.0))]);
        assert!(matches!(call_node, AstNode::CallFunction { .. }));

        // Test rand, barrier, drop
        assert!(matches!(rand(), AstNode::Rand));
        assert!(matches!(barrier(), AstNode::Barrier));
        assert_eq!(drop("x"), AstNode::Drop("x".to_string()));
    }
}

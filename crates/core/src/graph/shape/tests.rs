use super::*;

#[test]
fn test_const_creation() {
    let expr = Expr::Const(42);
    assert_eq!(expr, Expr::Const(42));
}

#[test]
fn test_var_creation() {
    let expr = Expr::Var("x".to_string());
    assert_eq!(expr, Expr::Var("x".to_string()));
}

#[test]
fn test_from_integer() {
    let expr = Expr::from(10isize);
    assert_eq!(expr, Expr::Const(10));

    let expr = Expr::from(5usize);
    assert_eq!(expr, Expr::Const(5));
}

#[test]
fn test_from_str() {
    let expr = Expr::from("x");
    assert_eq!(expr, Expr::Var("x".to_string()));

    let expr = Expr::from("batch_size".to_string());
    assert_eq!(expr, Expr::Var("batch_size".to_string()));
}

#[test]
fn test_is_zero() {
    assert!(Expr::Const(0).is_zero());
    assert!(!Expr::Const(1).is_zero());
    assert!(!Expr::Var("x".to_string()).is_zero());
}

#[test]
fn test_is_one() {
    assert!(Expr::Const(1).is_one());
    assert!(!Expr::Const(0).is_one());
    assert!(!Expr::Var("x".to_string()).is_one());
}

#[test]
fn test_add_operator() {
    let a = Expr::Const(2);
    let b = Expr::Const(3);
    let sum = a + b;
    assert_eq!(
        sum,
        Expr::Add(Box::new(Expr::Const(2)), Box::new(Expr::Const(3)))
    );
}

#[test]
fn test_sub_operator() {
    let a = Expr::Const(5);
    let b = Expr::Const(3);
    let diff = a - b;
    assert_eq!(
        diff,
        Expr::Sub(Box::new(Expr::Const(5)), Box::new(Expr::Const(3)))
    );
}

#[test]
fn test_mul_operator() {
    let a = Expr::Const(4);
    let b = Expr::Const(5);
    let prod = a * b;
    assert_eq!(
        prod,
        Expr::Mul(Box::new(Expr::Const(4)), Box::new(Expr::Const(5)))
    );
}

#[test]
fn test_div_operator() {
    let a = Expr::Const(10);
    let b = Expr::Const(2);
    let quot = a / b;
    assert_eq!(
        quot,
        Expr::Div(Box::new(Expr::Const(10)), Box::new(Expr::Const(2)))
    );
}

#[test]
fn test_rem_operator() {
    let a = Expr::Const(10);
    let b = Expr::Const(3);
    let rem = a % b;
    assert_eq!(
        rem,
        Expr::Rem(Box::new(Expr::Const(10)), Box::new(Expr::Const(3)))
    );
}

#[test]
fn test_neg_operator() {
    let a = Expr::Const(5);
    let neg_a = -a;
    assert_eq!(
        neg_a,
        Expr::Sub(Box::new(Expr::Const(0)), Box::new(Expr::Const(5)))
    );
}

#[test]
fn test_simplify_add_zero() {
    // 0 + x = x
    let expr = Expr::Const(0) + Expr::Var("x".to_string());
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Var("x".to_string()));

    // x + 0 = x
    let expr = Expr::Var("x".to_string()) + Expr::Const(0);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Var("x".to_string()));
}

#[test]
fn test_simplify_add_const() {
    // 2 + 3 = 5
    let expr = Expr::Const(2) + Expr::Const(3);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(5));
}

#[test]
fn test_simplify_sub_zero() {
    // x - 0 = x
    let expr = Expr::Var("x".to_string()) - Expr::Const(0);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Var("x".to_string()));
}

#[test]
fn test_simplify_sub_self() {
    // x - x = 0
    let x = Expr::Var("x".to_string());
    let expr = x.clone() - x;
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(0));
}

#[test]
fn test_simplify_sub_const() {
    // 5 - 3 = 2
    let expr = Expr::Const(5) - Expr::Const(3);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(2));
}

#[test]
fn test_simplify_mul_zero() {
    // 0 * x = 0
    let expr = Expr::Const(0) * Expr::Var("x".to_string());
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(0));

    // x * 0 = 0
    let expr = Expr::Var("x".to_string()) * Expr::Const(0);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(0));
}

#[test]
fn test_simplify_mul_one() {
    // 1 * x = x
    let expr = Expr::Const(1) * Expr::Var("x".to_string());
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Var("x".to_string()));

    // x * 1 = x
    let expr = Expr::Var("x".to_string()) * Expr::Const(1);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Var("x".to_string()));
}

#[test]
fn test_simplify_mul_const() {
    // 3 * 4 = 12
    let expr = Expr::Const(3) * Expr::Const(4);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(12));
}

#[test]
fn test_simplify_div_one() {
    // x / 1 = x
    let expr = Expr::Var("x".to_string()) / Expr::Const(1);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Var("x".to_string()));
}

#[test]
fn test_simplify_div_self() {
    // x / x = 1
    let x = Expr::Var("x".to_string());
    let expr = x.clone() / x;
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(1));
}

#[test]
fn test_simplify_div_zero_numerator() {
    // 0 / x = 0
    let expr = Expr::Const(0) / Expr::Var("x".to_string());
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(0));
}

#[test]
#[should_panic(expected = "division by zero")]
fn test_simplify_div_zero_denominator() {
    // x / 0 should panic
    let expr = Expr::Var("x".to_string()) / Expr::Const(0);
    let _ = expr.simplify();
}

#[test]
fn test_simplify_div_const() {
    // 10 / 2 = 5
    let expr = Expr::Const(10) / Expr::Const(2);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(5));
}

#[test]
fn test_simplify_rem_one() {
    // x % 1 = 0
    let expr = Expr::Var("x".to_string()) % Expr::Const(1);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(0));
}

#[test]
fn test_simplify_rem_self() {
    // x % x = 0
    let x = Expr::Var("x".to_string());
    let expr = x.clone() % x;
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(0));
}

#[test]
fn test_simplify_rem_const() {
    // 10 % 3 = 1
    let expr = Expr::Const(10) % Expr::Const(3);
    let simplified = expr.simplify();
    assert_eq!(simplified, Expr::Const(1));
}

#[test]
fn test_simplify_complex_expr() {
    // (x + 0) * 1 = x
    let x = Expr::Var("x".to_string());
    let expr = (x.clone() + 0) * 1;
    let simplified = expr.simplify();
    assert_eq!(simplified, x);
}

#[test]
fn test_display_const() {
    let expr = Expr::Const(42);
    assert_eq!(format!("{}", expr), "42");
}

#[test]
fn test_display_var() {
    let expr = Expr::Var("x".to_string());
    assert_eq!(format!("{}", expr), "x");
}

#[test]
fn test_display_add() {
    let expr = Expr::Const(2) + Expr::Const(3);
    assert_eq!(format!("{}", expr), "2 + 3");
}

#[test]
fn test_display_sub() {
    let expr = Expr::Const(5) - Expr::Const(3);
    assert_eq!(format!("{}", expr), "5 - 3");
}

#[test]
fn test_display_mul() {
    let expr = Expr::Const(4) * Expr::Const(5);
    assert_eq!(format!("{}", expr), "4 * 5");
}

#[test]
fn test_display_complex() {
    // (2 + 3) * 4
    let expr = (Expr::Const(2) + Expr::Const(3)) * Expr::Const(4);
    assert_eq!(format!("{}", expr), "(2 + 3) * 4");
}

#[test]
fn test_assign_operators() {
    let mut expr = Expr::Const(5);
    expr += 3;
    assert_eq!(
        expr,
        Expr::Add(Box::new(Expr::Const(5)), Box::new(Expr::Const(3)))
    );

    let mut expr = Expr::Const(10);
    expr -= 3;
    assert_eq!(
        expr,
        Expr::Sub(Box::new(Expr::Const(10)), Box::new(Expr::Const(3)))
    );

    let mut expr = Expr::Const(4);
    expr *= 5;
    assert_eq!(
        expr,
        Expr::Mul(Box::new(Expr::Const(4)), Box::new(Expr::Const(5)))
    );

    let mut expr = Expr::Const(20);
    expr /= 4;
    assert_eq!(
        expr,
        Expr::Div(Box::new(Expr::Const(20)), Box::new(Expr::Const(4)))
    );

    let mut expr = Expr::Const(10);
    expr %= 3;
    assert_eq!(
        expr,
        Expr::Rem(Box::new(Expr::Const(10)), Box::new(Expr::Const(3)))
    );
}

#[test]
fn test_to_astnode_const() {
    use crate::ast::{AstNode, Literal};

    let expr = Expr::Const(42);
    let ast: AstNode = expr.into();

    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 42),
        _ => panic!("Expected Const node with Int(42)"),
    }
}

#[test]
fn test_to_astnode_add() {
    use crate::ast::{AstNode, Literal};

    // 定数の加算をテスト
    let expr = Expr::Const(2) + Expr::Const(3);
    let ast: AstNode = expr.into();

    // After simplify, this becomes Const(5)
    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 5),
        _ => panic!("Expected Const(5) after simplification"),
    }

    // 変数の加算もサポートされるようになった
    let a = Expr::Var("a".to_string());
    let b = Expr::Var("b".to_string());
    let expr = Expr::Add(Box::new(a), Box::new(b));
    let ast: AstNode = expr.into();

    // 変数を含む加算はAdd nodeとして変換される
    match ast {
        AstNode::Add(left, right) => match (*left, *right) {
            (AstNode::Var(name_a), AstNode::Var(name_b)) => {
                assert_eq!(name_a, "a");
                assert_eq!(name_b, "b");
            }
            _ => panic!("Expected Var nodes"),
        },
        _ => panic!("Expected Add node"),
    }
}

#[test]
fn test_to_astnode_sub() {
    use crate::ast::{AstNode, Literal};

    // After simplify: 5 - 3 = 2
    let expr = Expr::Const(5) - Expr::Const(3);
    let ast: AstNode = expr.into();

    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 2),
        _ => panic!("Expected Const(2) after simplification"),
    }
}

#[test]
fn test_to_astnode_mul() {
    use crate::ast::{AstNode, Literal};

    // After simplify: 4 * 5 = 20
    let expr = Expr::Const(4) * Expr::Const(5);
    let ast: AstNode = expr.into();

    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 20),
        _ => panic!("Expected Const(20) after simplification"),
    }
}

#[test]
fn test_to_astnode_div() {
    use crate::ast::{AstNode, Literal};

    // After simplify: 10 / 2 = 5
    let expr = Expr::Const(10) / Expr::Const(2);
    let ast: AstNode = expr.into();

    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 5),
        _ => panic!("Expected Const(5) after simplification"),
    }
}

#[test]
fn test_to_astnode_rem() {
    use crate::ast::{AstNode, Literal};

    // After simplify: 10 % 3 = 1
    let expr = Expr::Const(10) % Expr::Const(3);
    let ast: AstNode = expr.into();

    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 1),
        _ => panic!("Expected Const(1) after simplification"),
    }
}

#[test]
fn test_to_astnode_complex() {
    use crate::ast::{AstNode, Literal};

    // After simplify: (2 + 3) * 4 = 5 * 4 = 20
    let expr = (Expr::Const(2) + Expr::Const(3)) * Expr::Const(4);
    let ast: AstNode = expr.into();

    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 20),
        _ => panic!("Expected Const(20) after simplification"),
    }
}

#[test]
fn test_to_astnode_with_simplify() {
    use crate::ast::{AstNode, Literal};

    // 0 + 5 should simplify to 5 before conversion
    let expr = Expr::Const(0) + Expr::Const(5);
    let ast: AstNode = expr.into();

    match ast {
        AstNode::Const(Literal::I64(v)) => assert_eq!(v, 5),
        _ => panic!("Expected simplified Const(5)"),
    }
}

#[test]
fn test_to_astnode_var() {
    use crate::ast::AstNode;

    let expr = Expr::Var("x".to_string());
    let ast: AstNode = expr.into();

    // Varは正常に変換されるはず
    match ast {
        AstNode::Var(name) => assert_eq!(name, "x"),
        _ => panic!("Expected Var node"),
    }
}

// ===== From<char> tests =====

#[test]
fn test_from_char() {
    let expr = Expr::from('A');
    assert_eq!(expr, Expr::Var("A".to_string()));

    let expr = Expr::from('N');
    assert_eq!(expr, Expr::Var("N".to_string()));
}

// ===== shape! macro tests =====

#[test]
fn test_shape_macro_empty() {
    let shape: Vec<Expr> = shape![];
    assert!(shape.is_empty());
}

#[test]
fn test_shape_macro_all_const() {
    let shape = shape![2, 3, 4];
    assert_eq!(shape, vec![Expr::Const(2), Expr::Const(3), Expr::Const(4)]);
}

#[test]
fn test_shape_macro_all_var() {
    let shape = shape!['B', 'H', 'W', 'C'];
    assert_eq!(
        shape,
        vec![
            Expr::Var("B".to_string()),
            Expr::Var("H".to_string()),
            Expr::Var("W".to_string()),
            Expr::Var("C".to_string()),
        ]
    );
}

#[test]
fn test_shape_macro_mixed() {
    let shape = shape![2, 'N', 4];
    assert_eq!(
        shape,
        vec![Expr::Const(2), Expr::Var("N".to_string()), Expr::Const(4),]
    );
}

#[test]
fn test_shape_macro_trailing_comma() {
    let shape = shape![1, 2, 3,];
    assert_eq!(shape, vec![Expr::Const(1), Expr::Const(2), Expr::Const(3)]);
}

#[test]
fn test_shape_macro_single_element() {
    let shape = shape![10];
    assert_eq!(shape, vec![Expr::Const(10)]);

    let shape = shape!['X'];
    assert_eq!(shape, vec![Expr::Var("X".to_string())]);
}

#[test]
fn test_shape_macro_with_str() {
    // &str型も変数名として扱われる
    let shape = shape![2, "batch", 4];
    assert_eq!(
        shape,
        vec![
            Expr::Const(2),
            Expr::Var("batch".to_string()),
            Expr::Const(4),
        ]
    );

    // 複数文字の変数名
    let shape = shape!["batch_size", "seq_len", "hidden_dim"];
    assert_eq!(
        shape,
        vec![
            Expr::Var("batch_size".to_string()),
            Expr::Var("seq_len".to_string()),
            Expr::Var("hidden_dim".to_string()),
        ]
    );
}

#[test]
fn test_shape_macro_mixed_all_types() {
    // 数値、char、&strを全て混在
    let shape = shape![32, 'N', "hidden"];
    assert_eq!(
        shape,
        vec![
            Expr::Const(32),
            Expr::Var("N".to_string()),
            Expr::Var("hidden".to_string()),
        ]
    );
}

// =============================================================================
// Expr::evaluate tests
// =============================================================================

#[test]
fn test_evaluate_const() {
    let vars = std::collections::HashMap::new();
    assert_eq!(Expr::Const(42).evaluate(&vars), Ok(42));
    assert_eq!(Expr::Const(-10).evaluate(&vars), Ok(-10));
    assert_eq!(Expr::Const(0).evaluate(&vars), Ok(0));
}

#[test]
fn test_evaluate_var() {
    let mut vars = std::collections::HashMap::new();
    vars.insert("N".to_string(), 128);
    vars.insert("M".to_string(), 256);

    assert_eq!(Expr::Var("N".to_string()).evaluate(&vars), Ok(128));
    assert_eq!(Expr::Var("M".to_string()).evaluate(&vars), Ok(256));

    // Undefined variable should return error
    assert!(Expr::Var("undefined".to_string()).evaluate(&vars).is_err());
}

#[test]
fn test_evaluate_arithmetic() {
    let mut vars = std::collections::HashMap::new();
    vars.insert("x".to_string(), 10);
    vars.insert("y".to_string(), 3);

    // 10 + 3 = 13
    let add = Expr::Var("x".to_string()) + Expr::Var("y".to_string());
    assert_eq!(add.evaluate(&vars), Ok(13));

    // 10 - 3 = 7
    let sub = Expr::Var("x".to_string()) - Expr::Var("y".to_string());
    assert_eq!(sub.evaluate(&vars), Ok(7));

    // 10 * 3 = 30
    let mul = Expr::Var("x".to_string()) * Expr::Var("y".to_string());
    assert_eq!(mul.evaluate(&vars), Ok(30));

    // 10 / 3 = 3
    let div = Expr::Var("x".to_string()) / Expr::Var("y".to_string());
    assert_eq!(div.evaluate(&vars), Ok(3));

    // 10 % 3 = 1
    let rem = Expr::Var("x".to_string()) % Expr::Var("y".to_string());
    assert_eq!(rem.evaluate(&vars), Ok(1));
}

#[test]
fn test_evaluate_complex_expr() {
    let mut vars = std::collections::HashMap::new();
    vars.insert("batch".to_string(), 32);
    vars.insert("seq_len".to_string(), 128);
    vars.insert("hidden".to_string(), 512);

    // batch * seq_len * hidden = 32 * 128 * 512 = 2097152
    let expr = Expr::Var("batch".to_string())
        * Expr::Var("seq_len".to_string())
        * Expr::Var("hidden".to_string());
    assert_eq!(expr.evaluate(&vars), Ok(2097152));
}

#[test]
fn test_evaluate_division_by_zero() {
    let vars = std::collections::HashMap::new();

    let div = Expr::Const(10) / Expr::Const(0);
    assert!(div.evaluate(&vars).is_err());

    let rem = Expr::Const(10) % Expr::Const(0);
    assert!(rem.evaluate(&vars).is_err());
}

#[test]
fn test_evaluate_idx_error() {
    let vars = std::collections::HashMap::new();
    let idx = Expr::Idx(0);
    assert!(idx.evaluate(&vars).is_err());
}

#[test]
fn test_evaluate_usize() {
    let mut vars = std::collections::HashMap::new();
    vars.insert("N".to_string(), 128);

    assert_eq!(Expr::Const(42).evaluate_usize(&vars), Ok(42));
    assert_eq!(Expr::Var("N".to_string()).evaluate_usize(&vars), Ok(128));

    // Negative result should error
    assert!(Expr::Const(-1).evaluate_usize(&vars).is_err());
    assert!(
        (Expr::Const(0) - Expr::Const(10))
            .evaluate_usize(&vars)
            .is_err()
    );
}

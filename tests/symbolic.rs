use harp::shape::symbolic::Expr;

#[test]
fn test_expr_simplify_int_ops() {
    // Addition
    assert_eq!((Expr::from(1) + Expr::from(2)).simplify(), Expr::from(3));
    assert_eq!((Expr::from(5) + Expr::from(0)).simplify(), Expr::from(5));

    // Subtraction
    assert_eq!((Expr::from(5) - Expr::from(2)).simplify(), Expr::from(3));
    assert_eq!((Expr::from(5) - Expr::from(0)).simplify(), Expr::from(5));
    assert_eq!((Expr::from(5) - Expr::from(5)).simplify(), Expr::from(0));

    // Multiplication
    assert_eq!((Expr::from(2) * Expr::from(3)).simplify(), Expr::from(6));
    assert_eq!((Expr::from(5) * Expr::from(0)).simplify(), Expr::from(0));
    assert_eq!((Expr::from(5) * Expr::from(1)).simplify(), Expr::from(5));

    // Division
    assert_eq!((Expr::from(6) / Expr::from(3)).simplify(), Expr::from(2));
    assert_eq!((Expr::from(0) / Expr::from(5)).simplify(), Expr::from(0));
    assert_eq!((Expr::from(5) / Expr::from(1)).simplify(), Expr::from(5));

    // Remainder
    assert_eq!((Expr::from(7) % Expr::from(3)).simplify(), Expr::from(1));
    assert_eq!((Expr::from(0) % Expr::from(5)).simplify(), Expr::from(0));
    assert_eq!((Expr::from(7) % Expr::from(1)).simplify(), Expr::from(0));

    // Negation
    assert_eq!((-Expr::from(5)).simplify(), Expr::from(-5));
    assert_eq!((-(-Expr::from(5))).simplify(), Expr::from(5));
}

#[test]
fn test_expr_simplify_mixed_ops() {
    let x = Expr::Var("x".to_string());

    // Add with variable
    assert_eq!((x.clone() + Expr::from(0)).simplify(), x.clone());
    assert_eq!((Expr::from(0) + x.clone()).simplify(), x.clone());
    assert_eq!(
        (x.clone() + Expr::from(1)).simplify(),
        x.clone() + Expr::from(1)
    );

    // Mul with variable
    assert_eq!((x.clone() * Expr::from(0)).simplify(), Expr::from(0));
    assert_eq!((Expr::from(0) * x.clone()).simplify(), Expr::from(0));
    assert_eq!((x.clone() * Expr::from(1)).simplify(), x.clone());
    assert_eq!((Expr::from(1) * x.clone()).simplify(), x.clone());
    assert_eq!(
        (x.clone() * Expr::from(2)).simplify(),
        x.clone() * Expr::from(2)
    );

    // Sub with variable
    assert_eq!((x.clone() - Expr::from(0)).simplify(), x.clone());
    assert_eq!((x.clone() - x.clone()).simplify(), Expr::from(0));

    // Div with variable
    assert_eq!((Expr::from(0) / x.clone()).simplify(), Expr::from(0));
    assert_eq!((x.clone() / Expr::from(1)).simplify(), x.clone());
    assert_eq!((x.clone() / x.clone()).simplify(), Expr::from(1));
}

#[test]
fn test_expr_replace() {
    let x = Expr::Var("x".to_string());
    let y = Expr::Var("y".to_string());
    let expr = (x.clone() + Expr::from(1)) * y.clone();

    // Replace 'x' with '10'
    let replaced_expr = expr.clone().replace(&Expr::Var("x".to_string()), &Expr::from(10));
    assert_eq!(replaced_expr, (Expr::from(10) + Expr::from(1)) * y.clone());

    // Replace 'y' with 'x + 5'
    let replaced_expr_2 = expr
        .clone()
        .replace(&Expr::Var("y".to_string()), &(x.clone() + Expr::from(5)));
    assert_eq!(
        replaced_expr_2,
        (x.clone() + Expr::from(1)) * (x.clone() + Expr::from(5))
    );

    // Replace Index with a variable
    let index_expr = Expr::Index;
    let replaced_index = index_expr.clone().replace(&Expr::Index, &Expr::Var("idx0".to_string()));
    assert_eq!(replaced_index, Expr::Var("idx0".to_string()));

    // No replacement if variable not found
    let _z = Expr::Var("z".to_string());
    let replaced_expr_3 = expr.clone().replace(&Expr::Var("z".to_string()), &Expr::from(100));
    assert_eq!(replaced_expr_3, expr.clone());
}

#[test]
fn test_expr_from_traits() {
    assert_eq!(Expr::from(10), Expr::Int(10));
    assert_eq!(Expr::from(-5), Expr::Int(-5));
    assert_eq!(Expr::from("my_var"), Expr::Var("my_var".to_string()));
    assert_eq!(
        Expr::from("another_var".to_string()),
        Expr::Var("another_var".to_string())
    );
}
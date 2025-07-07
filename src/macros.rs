#[macro_export]
/// A convenient macro for creating a `Vec<Expr>` from a list of expressions.
///
/// This macro automatically converts integer literals and string literals
/// into `Expr::Int` and `Expr::Var` respectively, making it easier to define
/// symbolic shapes.
///
/// # Arguments
///
/// * `$x` - A comma-separated list of expressions, which can be integers or strings.
///
/// # Returns
///
/// A `Vec<Expr>` containing the converted expressions.
///
/// # Examples
///
/// ```
/// use harp::s;
/// use harp::shape::symbolic::Expr;
///
/// let shape_vec = s![2, "batch_size", 128];
/// assert_eq!(
///     shape_vec,
///     vec![
///         Expr::Int(2),
///         Expr::Var("batch_size".to_string()),
///         Expr::Int(128)
///     ]
/// );
/// ```
macro_rules! s {
    ($($x:expr),*) => {
        vec![
            $(
                $crate::shape::symbolic::Expr::from($x)
            ),*
        ]
    };
}

#[cfg(test)]
mod shape_macro_test {
    use crate::shape::symbolic::Expr;

    #[test]
    fn test_shape_macro() {
        let vec = s![2, "a", 4];
        assert_eq!(
            vec,
            vec![Expr::Int(2), Expr::Var("a".to_string()), Expr::Int(4)]
        );
    }
}

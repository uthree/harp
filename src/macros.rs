#[macro_export]
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

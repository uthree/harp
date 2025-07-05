use harp::ops::{
    Add, ElementwiseBinaryOperator, ElementwiseUnaryOperator, Exp2, Input, Log2, Mul, Operator,
    Recip, Rem, Sin, Sqrt,
};

#[test]
fn test_input_operator() {
    let _op = Input {};
    // Simply checking if it compiles and implements Operator trait
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Input {});
}

#[test]
fn test_add_operator() {
    let _op = Add {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Add {});
    fn check_binary_operator<T: ElementwiseBinaryOperator>(_: T) {}
    check_binary_operator(Add {});
}

#[test]
fn test_mul_operator() {
    let _op = Mul {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Mul {});
    fn check_binary_operator<T: ElementwiseBinaryOperator>(_: T) {}
    check_binary_operator(Mul {});
}

#[test]
fn test_recip_operator() {
    let _op = Recip {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Recip {});
    fn check_unary_operator<T: ElementwiseUnaryOperator>(_: T) {}
    check_unary_operator(Recip {});
}

#[test]
fn test_rem_operator() {
    let _op = Rem {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Rem {});
    fn check_binary_operator<T: ElementwiseBinaryOperator>(_: T) {}
    check_binary_operator(Rem {});
}

#[test]
fn test_sin_operator() {
    let _op = Sin {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Sin {});
    fn check_unary_operator<T: ElementwiseUnaryOperator>(_: T) {}
    check_unary_operator(Sin {});
}

#[test]
fn test_exp2_operator() {
    let _op = Exp2 {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Exp2 {});
    fn check_unary_operator<T: ElementwiseUnaryOperator>(_: T) {}
    check_unary_operator(Exp2 {});
}

#[test]
fn test_log2_operator() {
    let _op = Log2 {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Log2 {});
    fn check_unary_operator<T: ElementwiseUnaryOperator>(_: T) {}
    check_unary_operator(Log2 {});
}

#[test]
fn test_sqrt_operator() {
    let _op = Sqrt {};
    fn check_operator<T: Operator>(_: T) {}
    check_operator(Sqrt {});
    fn check_unary_operator<T: ElementwiseUnaryOperator>(_: T) {}
    check_unary_operator(Sqrt {});
}

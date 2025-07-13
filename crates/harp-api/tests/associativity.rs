use harp_ir::op::{AssociativeOp, Max, OpAdd, OpMul};

/// This is a helper function for static assertion.
/// It will only compile if the type `T` implements the `AssociativeOp` trait.
fn _is_associative<T: AssociativeOp>() {}

#[test]
fn test_associative_marker() {
    // This test serves as a static check to ensure that the
    // AssociativeOp trait is implemented for the correct operators.
    // If an operator that is not associative is marked as such,
    // this test will not fail, but it provides a clear declaration
    // of which operators are expected to be associative.
    _is_associative::<OpAdd>();
    _is_associative::<OpMul>();
    _is_associative::<Max>();

    // To manually verify that non-associative operators fail,
    // you could uncomment the following lines. They should cause a
    // compile-time error.
    // _is_associative::<OpSub>();
    // _is_associative::<OpDiv>();
}

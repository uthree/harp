use harp::shapetracker::{ShapeTracker, View};
use harp::uop::{DType, UOp};

#[test]
fn test_view_default_strides() {
    let shape = vec![2, 3, 4];
    let view = View::new(shape.clone(), None, None, None);
    assert_eq!(view.strides, vec![12, 4, 1]);
    assert!(view.contiguous);
}

#[test]
fn test_view_custom_strides() {
    let shape = vec![2, 3];
    let strides = Some(vec![1, 2]);
    let view = View::new(shape, strides, None, None);
    assert!(!view.contiguous);
}

#[test]
fn test_shapetracker_new() {
    let shape = vec![10, 20];
    let st = ShapeTracker::new(shape.clone());
    assert_eq!(st.views.len(), 1);
    assert_eq!(*st.shape(), shape);
    assert!(st.views[0].contiguous);
}

#[test]
fn test_shapetracker_reshape() {
    let st = ShapeTracker::new(vec![10, 20]);
    let new_st = st.reshape(vec![200]);
    assert_eq!(*new_st.shape(), vec![200]);
    assert!(new_st.views[0].contiguous);
}

#[test]
#[should_panic]
fn test_shapetracker_reshape_invalid() {
    let st = ShapeTracker::new(vec![10, 20]);
    st.reshape(vec![199]); // Panics because element count differs
}

#[test]
fn test_expr_node_simple_contiguous() {
    // Shape: [10, 20], Strides: [20, 1]
    let st = ShapeTracker::new(vec![10, 20]);
    let i = UOp::var("i", DType::U64);
    let expr = st.expr_node(&i);

    // Expected: (i / 1) % 20 * 1 + (i / 20) % 10 * 20
    // Simplified: i
    // The logic in expr_node is a bit complex, let's trace it.
    // rev_shape = [20, 10]
    // acc = 1
    // ret.push((i / 1) % 20) -> idx1
    // acc = 20
    // ret.push((i / 20) % 10) -> idx0
    // ret = [idx1, idx0]
    // rev_ret = [idx0, idx1]
    // expr_indices([idx0, idx1]) = idx0 * 20 + idx1 * 1
    // = ((i / 20) % 10) * 20 + ((i / 1) % 20)
    // This doesn't simplify to just `i`. Let's re-check tinygrad's logic.
    // Okay, the logic is `(idx // acc) % sh`.
    // Let's assume i = 25
    // idx1 = (25 / 1) % 20 = 5
    // idx0 = (25 / 20) % 10 = 1
    // expr = 1 * 20 + 5 = 25. It works.
    // The generated expression is correct, even if complex.
    // A good optimizer would simplify this down to `i`.
    // For now, let's just check against a manually constructed expected expression.
    let idx1 = (&i / &UOp::from(1u64)) % &UOp::from(20u64);
    let idx0 = (&i / &UOp::from(20u64)) % &UOp::from(10u64);
    let expected_expr = &idx0 * &UOp::from(20u64) + &idx1;

    assert_eq!(format!("{:?}", expr), format!("{:?}", expected_expr));
}

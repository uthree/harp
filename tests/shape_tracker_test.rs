use harp::shape::{symbolic::Expr, tracker::ShapeTracker};

#[test]
fn test_shape_tracker_full() {
    let dims = vec![2, 3, 4];
    let tracker = ShapeTracker::full(dims.iter().map(|&d| Expr::Int(d as isize)).collect());

    assert_eq!(tracker.max.len(), 3);
    assert_eq!(tracker.map.len(), 3);

    // Check max (dimensions)
    assert_eq!(tracker.max[0], Expr::Int(2));
    assert_eq!(tracker.max[1], Expr::Int(3));
    assert_eq!(tracker.max[2], Expr::Int(4));

    // Check map (strides - for contiguous, row-major)
    // The exact symbolic expressions might be complex, but we can check their structure
    // For [2, 3, 4], strides would be [12, 4, 1]
    // map should be something like [idx0 * 12, idx1 * 4, idx2 * 1]
    // This is a simplified check, actual symbolic expressions might be more involved.
    // We'll rely on the `simplify` method to make them canonical.
    assert_eq!(tracker.map[0].to_string(), "(idx * 12)");
    assert_eq!(tracker.map[1].to_string(), "(idx * 4)");
    assert_eq!(tracker.map[2].to_string(), "idx");
}

#[test]
fn test_shape_tracker_from_vec_usize() {
    let dims = vec![5, 6];
    let tracker: ShapeTracker = dims.into();

    assert_eq!(tracker.max.len(), 2);
    assert_eq!(tracker.map.len(), 2);

    assert_eq!(tracker.max[0], Expr::Int(5));
    assert_eq!(tracker.max[1], Expr::Int(6));

    assert_eq!(tracker.map[0].to_string(), "(idx * 6)");
    assert_eq!(tracker.map[1].to_string(), "idx");
}

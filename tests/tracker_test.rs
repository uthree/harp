use harp::graph::Graph;
use harp::shape::symbolic::Expr;
use harp::shape::tracker::ShapeTracker;

#[test]
fn test_shape_tracker_full() {
    let graph = Graph::new();
    let dims = vec![Expr::from(2), Expr::from(3), Expr::from(4)];
    let shape_tracker = ShapeTracker::full(graph.clone().downgrade(), dims.clone());

    // Test maxs
    assert_eq!(shape_tracker.max.len(), 3);
    assert_eq!(shape_tracker.max[0], Expr::from(2));
    assert_eq!(shape_tracker.max[1], Expr::from(3));
    assert_eq!(shape_tracker.max[2], Expr::from(4));

    // Test maps (simplified expressions)
    // The order of maps is reversed during creation, then reversed back.
    // So, for dims [2, 3, 4], the maps should correspond to:
    // Index * (3 * 4) = Index * 12
    // Index * 4
    // Index * 1
    assert_eq!(shape_tracker.map.len(), 3);
    assert_eq!(shape_tracker.map[0], Expr::Index * Expr::from(12));
    assert_eq!(shape_tracker.map[1], Expr::Index * Expr::from(4));
    assert_eq!(shape_tracker.map[2], Expr::Index);
}

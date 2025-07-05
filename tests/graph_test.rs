use harp::graph::Graph;
use harp::shape::symbolic::Expr;

#[test]
fn test_graph_new() {
    let graph = Graph::new();
    assert!(graph.data.borrow().input_nodes.is_empty());
    assert!(graph.data.borrow().output_nodes.is_empty());
}

#[test]
fn test_graph_input() {
    let mut graph = Graph::new();
    let shape = vec![Expr::from(10), Expr::from(20)];
    let _tensor = graph.input(shape.clone());

    assert_eq!(graph.data.borrow().input_nodes.len(), 1);
    assert_eq!(
        graph.data.borrow().input_nodes[0]
            .data
            .borrow()
            .shape_tracker
            .map,
        vec![Expr::Index * Expr::from(20), Expr::Index]
    );
}

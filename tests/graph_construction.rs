use harp::{
    graph::EdgeMetadata,
    operator::{Add, Exp2, Input, Mul},
    prelude::*,
};
use std::sync::{Arc, Mutex};

#[test]
fn test_simple_graph_construction() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into(), 1.into()]);

    // Inputs
    let a = Graph::new_input(graph.clone(), shape.clone());
    let b = Graph::new_input(graph.clone(), shape.clone());

    // Operation
    let c = &a + &b; // Add
    let d = &c * &a; // Mul
    let e = d.exp2(); // Exp2

    // --- Verification ---
    let g = graph.lock().unwrap();
    println!("{:?}", g);

    // Node count: 2 inputs + 3 ops = 5
    println!("Actual node count: {}", g.node_count());
    assert_eq!(g.node_count(), 5);
    // Edge count: (a,b)->add, (c,a)->mul, d->exp2 = 4
    assert_eq!(g.edge_count(), 5);

    // Check node types
    assert!(
        g.node_weight(a.node_index)
            .unwrap()
            .op()
            .as_any()
            .downcast_ref::<Input>()
            .is_some()
    );
    assert!(
        g.node_weight(b.node_index)
            .unwrap()
            .op()
            .as_any()
            .downcast_ref::<Input>()
            .is_some()
    );
    assert!(
        g.node_weight(c.node_index)
            .unwrap()
            .op()
            .as_any()
            .downcast_ref::<Add>()
            .is_some()
    );
    assert!(
        g.node_weight(d.node_index)
            .unwrap()
            .op()
            .as_any()
            .downcast_ref::<Mul>()
            .is_some()
    );
    assert!(
        g.node_weight(e.node_index)
            .unwrap()
            .op()
            .as_any()
            .downcast_ref::<Exp2>()
            .is_some()
    );

    // Check edges and argument order
    // c = a + b
    let mut c_parents = g.parents(c.node_index).collect::<Vec<_>>();
    c_parents.sort_by_key(|k| k.1); // Sort by argument index
    assert_eq!(c_parents.len(), 2);
    assert_eq!(
        c_parents[0],
        (
            a.node_index,
            EdgeMetadata {
                arg_index: 0,
                output_index: 0
            }
        )
    ); // a is 0th arg
    assert_eq!(
        c_parents[1],
        (
            b.node_index,
            EdgeMetadata {
                arg_index: 1,
                output_index: 0
            }
        )
    ); // b is 1st arg

    // d = c * a
    let mut d_parents = g.parents(d.node_index).collect::<Vec<_>>();
    d_parents.sort_by_key(|k| k.1);
    assert_eq!(d_parents.len(), 2);
    assert_eq!(
        d_parents[0],
        (
            c.node_index,
            EdgeMetadata {
                arg_index: 0,
                output_index: 0
            }
        )
    ); // c is 0th arg
    assert_eq!(
        d_parents[1],
        (
            a.node_index,
            EdgeMetadata {
                arg_index: 1,
                output_index: 0
            }
        )
    ); // a is 1st arg

    // e = d.exp2()
    let mut e_parents = g.parents(e.node_index).collect::<Vec<_>>();
    e_parents.sort_by_key(|k| k.1);
    assert_eq!(e_parents.len(), 1);
    assert_eq!(
        e_parents[0],
        (
            d.node_index,
            EdgeMetadata {
                arg_index: 0,
                output_index: 0
            }
        )
    ); // d is 0th arg
}

#[test]
fn test_to_dot_output() {
    let graph = Arc::new(Mutex::new(Graph::new()));
    let shape = ShapeTracker::full(vec![1.into(), 1.into()]);

    let a = Graph::new_input(graph.clone(), shape.clone());
    let b = Graph::new_input(graph.clone(), shape.clone());
    let c = &a + &b;
    let e = c.exp2();

    Graph::add_output_node(graph.clone(), &e);

    let g = graph.lock().unwrap();
    let dot_output = g.to_dot();

    // Basic checks for expected strings in the DOT output
    assert!(dot_output.contains("Input\nShape[(1, 1), (idx0, idx1)]"));
    assert!(dot_output.contains("Add\nShape[(1, 1), (idx0, idx1)]"));
    assert!(dot_output.contains("Exp2\nShape[(1, 1), (idx0, idx1)]"));
    assert!(dot_output.contains("peripheries=2")); // Output node style
    assert!(dot_output.contains("style=filled")); // Input node style
    assert!(dot_output.contains("fillcolor=lightgray")); // Input node style
    assert!(dot_output.contains("label = \"(0, 0)\"")); // Edge label
    assert!(dot_output.contains("label = \"(1, 0)\"")); // Edge label
}

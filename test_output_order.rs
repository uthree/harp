use harp::graph::{DType, Graph};

fn main() {
    // グラフ1: 出力順序 "a", "b"
    let mut graph1 = Graph::new();
    let input1 = graph1
        .input("x")
        .with_dtype(DType::F32)
        .with_shape(vec![10])
        .build();
    graph1.output("a", input1.clone());
    graph1.output("b", input1.clone());

    // グラフ2: 出力順序 "b", "a"
    let mut graph2 = Graph::new();
    let input2 = graph2
        .input("x")
        .with_dtype(DType::F32)
        .with_shape(vec![10])
        .build();
    graph2.output("b", input2.clone());
    graph2.output("a", input2.clone());

    let dot1 = graph1.to_dot();
    let dot2 = graph2.to_dot();

    println!("Graph 1 DOT length: {}", dot1.len());
    println!("Graph 2 DOT length: {}", dot2.len());
    println!("\nAre they equal? {}", dot1 == dot2);

    if dot1 != dot2 {
        println!("\n=== Graph 1 DOT ===");
        println!("{}", dot1);
        println!("\n=== Graph 2 DOT ===");
        println!("{}", dot2);
    } else {
        println!("\n✓ グラフは同一です");
    }
}

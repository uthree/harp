use harp::prelude::*;
use std::sync::{Arc, Mutex};

fn main() {
    // 1. 新しいグラフを作成
    let graph = Arc::new(Mutex::new(Graph::new()));

    // 2. 入力テンソルを定義
    let shape_a = ShapeTracker::full(vec![1.into(), 1.into()]);
    let a = Graph::new_input(graph.clone(), shape_a);

    let shape_b = ShapeTracker::full(vec![1.into(), 1.into()]);
    let b = Graph::new_input(graph.clone(), shape_b);

    // 3. 演算を適用してグラフを構築
    let c = &a + &b; // Add
    let d = &c * &a; // Mul
    let e = d.exp2(); // Exp2

    // 4. 出力ノードを指定
    graph.lock().unwrap().add_output(&e);

    // 5. 構築されたグラフをDOT形式で出力
    let g = graph.lock().unwrap();
    println!("{}", g.to_dot());
}

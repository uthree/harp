use harp::graph::{DType, Graph, GraphNode};

fn matmul(a: GraphNode, b: GraphNode) -> GraphNode {
    let a_shape = a.view.shape();
    let b_shape = b.view.shape();

    let m = a_shape[0].clone();
    let k_a = a_shape[1].clone();
    let n = b_shape[1].clone();

    let a_unsqueezed = a.view(a.view.clone().unsqueeze(2));
    let b_unsqueezed = b.view(b.view.clone().unsqueeze(0));

    let expanded_shape = vec![m.clone(), k_a.clone(), n.clone()];
    let a_expanded = a_unsqueezed.view(a_unsqueezed.view.clone().expand(expanded_shape.clone()));
    let b_expanded = b_unsqueezed.view(b_unsqueezed.view.clone().expand(expanded_shape));

    let product = a_expanded * b_expanded;
    product.reduce_sum(1)
}

fn main() {
    let mut graph = Graph::new();
    
    let m = 64;
    let k = 32;
    let n = 64;
    let p = 128;
    
    let a = graph.input("a").with_dtype(DType::F32).with_shape(vec![m, k]).build();
    let b = graph.input("b").with_dtype(DType::F32).with_shape(vec![k, n]).build();
    let c = graph.input("c").with_dtype(DType::F32).with_shape(vec![m, n]).build();
    let d = graph.input("d").with_dtype(DType::F32).with_shape(vec![n, p]).build();
    
    // スカラー定数は自動的にブロードキャストされる
    let scale_scalar = 2.0f32 * 3.0f32;

    let temp1 = matmul(a, b);
    let temp2 = temp1 + c;
    let temp3 = temp2 * scale_scalar;
    let result = matmul(temp3, d);
    
    graph.output("result", result);
    
    // グラフをDOT形式で出力
    println!("\nGraph structure:");
    println!("{}", graph.to_dot());
}

use harp::backends::CpuBackend;
use harp::dot::ToDot;
use harp::tensor::Tensor;
use std::sync::Arc;

fn main() {
    // 1. バックエンドを作成します
    let backend = Arc::new(CpuBackend::new());

    // 2. 2つの入力テンソルをVecから作成します。
    let t1 = Tensor::from_vec(vec![2.0f32; 200], vec![10, 20], backend.clone());
    let t2 = Tensor::from_vec(vec![3.0f32; 200], vec![10, 20], backend.clone());

    // 3. テンソルを乗算します。
    let t3 = &t1 * &t2;

    // 4. 構築されたグラフをDOT形式で出力します。
    let dot_graph = t3.to_dot();
    println!("{}", dot_graph);

    println!("\n// 上記のDOTグラフをコピーして、Graphvizなどのツールで視覚化できます。");
}

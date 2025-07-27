use harp::prelude::*;
use ndarray::ArrayD;

fn main() {
    // 1. 2つの入力テンソルを作成します。
    let t1: Tensor = ArrayD::from_elem(vec![10, 20], 3.0f32).into();
    let t2: Tensor = ArrayD::from_elem(vec![10, 20], 4.0f32).into();

    // 2. テンソルを乗算します。
    let t3 = &t1 * &t2;

    // 3. 構築されたグラフをDOT形式で出力します。
    let dot_graph = t3.to_dot();
    println!("{}", dot_graph);

    println!("\n// 上記のDOTグラフをコピーして、Graphvizなどのツールで視覚化できます。");
}

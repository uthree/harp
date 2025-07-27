use harp::prelude::*;
use ndarray::ArrayD;

fn main() {
    // 1. 2つの入力テンソルを作成します。
    //    これらは計算グラフの葉（leaf）ノードになります。
    let t1: Tensor = ArrayD::from_elem(vec![10, 20], 1.0f32).into();
    let t2: Tensor = ArrayD::from_elem(vec![10, 20], 2.0f32).into();

    // 2. テンソルを加算します。
    //    これにより、t1とt2を子ノードとして持つ新しい計算グラフが構築されます。
    let t3 = &t1 + &t2;

    // 3. 構築されたグラフをDOT形式で出力します。
    let dot_graph = t3.to_dot();
    println!("{}", dot_graph);

    println!("\n// 上記のDOTグラフをコピーして、Graphvizなどのツールで視覚化できます。");
    println!(
        "// 例えば、https://dreampuf.github.io/GraphvizOnline/ などのオンラインビューアが便利です。"
    );
}

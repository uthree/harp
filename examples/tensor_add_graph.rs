use harp::prelude::*;
use std::rc::Rc;

fn main() {
    // 1. バックエンドを作成します
    let backend = Rc::new(ClangBackend::new().expect("Failed to create ClangBackend"));

    // 2. 2つの入力テンソルを作成します。
    //    これらは計算グラフの葉（leaf）ノードになります。
    let t1 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(vec![10, 20]),
        DType::F32,
        backend.clone(),
    );

    let t2 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(vec![10, 20]),
        DType::F32,
        backend.clone(),
    );

    // 3. テンソルを加算します。
    //    これにより、t1とt2を子ノードとして持つ新しい計算グラフが構築されます。
    let t3 = &t1 + &t2;

    // 4. 構築されたグラフをDOT形式で出力します。
    let dot_graph = t3.to_dot();
    println!("{}", dot_graph);

    println!("\n// 上記のDOTグラフをコピーして、Graphvizなどのツールで視覚化できます。");
    println!("// 例えば、https://dreampuf.github.io/GraphvizOnline/ などのオンラインビューアが便利です。");
}

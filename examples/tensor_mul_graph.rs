use harp::prelude::*;
use harp::dtype::IntoDType;
use std::rc::Rc;

fn main() {
    // 1. バックエンドを作成します
    let backend = Rc::new(ClangBackend::new().expect("Failed to create ClangBackend"));

    // 2. 2つの入力テンソルを作成します。
    let t1: Tensor<f32> = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(vec![10, 20]),
        f32::into_dtype(),
        backend.clone(),
    );

    let t2: Tensor<f32> = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(vec![10, 20]),
        f32::into_dtype(),
        backend.clone(),
    );

    // 3. テンソルを乗算します。
    let t3 = &t1 * &t2;

    // 4. 構築されたグラフをDOT形式で出力します。
    let dot_graph = t3.to_dot();
    println!("{}", dot_graph);

    println!("\n// 上記のDOTグラフをコピーして、Graphvizなどのツールで視覚化できます。");
}
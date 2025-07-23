use harp::backends::GccBackend;
use harp::dot::ToDot;
use harp::dtype::DType;
use harp::shapetracker::ShapeTracker;
use harp::tensor::{Tensor, TensorOp};
use std::sync::Arc;

fn main() {
    // 1. バックエンドを作成します
    let backend = Arc::new(GccBackend::new());

    // 2. 2つの入力テンソルを作成します。
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

    // 3. テンソルを乗算します。
    let t3 = &t1 * &t2;

    // 4. 構築されたグラフをDOT形式で出力します。
    let dot_graph = t3.to_dot();
    println!("{}", dot_graph);

    println!("\n// 上記のDOTグラフをコピーして、Graphvizなどのツールで視覚化できます。");
}

use harp::prelude::*;
use std::rc::Rc;

fn main() {
    let backend = Rc::new(ClangBackend::new().expect("Failed to create ClangBackend"));

    // 元となるテンソルを作成
    let t1 = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(vec![10, 20]),
        DType::F32,
        backend.clone(),
    );

    println!("--- Original Tensor Graph ---");
    println!("{}", t1.to_dot());

    // テンソルをreshape
    let t2 = t1.reshape(vec![2, 5, 20]);

    println!("\n--- Reshaped Tensor Graph ---");
    println!("{}", t2.to_dot());

    println!("\n// Reshape操作はShapeTrackerを変更するだけで、");
    println!("// 計算グラフの構造（opやsrc）は変更しないことがわかります。");
}

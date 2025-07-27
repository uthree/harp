use harp::prelude::*;
use ndarray::ArrayD;

fn main() {
    // 元となるテンソルを作成
    let t1: Tensor = ArrayD::from_elem(vec![10, 20], 1.0f32).into();

    println!("--- Original Tensor Graph ---");
    println!("{}", t1.to_dot());

    // テンソルをreshape
    let t2 = t1.reshape(vec![2, 5, 20]);

    println!("\n--- Reshaped Tensor Graph ---");
    println!("{}", t2.to_dot());

    println!("\n// Reshape操作はShapeTrackerを変更するだけで、");
    println!("// 計算グラフの構造（opやsrc）は変更しないことがわかります。");
}

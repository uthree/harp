use harp::backends::CpuBackend;
use harp::dot::ToDot;
use harp::tensor::Tensor;
use std::sync::Arc;

fn main() {
    let backend = Arc::new(CpuBackend::new());

    // 元となるテンソルを作成
    let t1 = Tensor::from_vec(vec![0.0f32; 200], vec![10, 20], backend.clone());

    println!("--- Original Tensor Graph ---");
    println!("{}", t1.to_dot());

    // テンソルをreshape
    let t2 = t1.reshape(vec![2, 5, 20]);

    println!("\n--- Reshaped Tensor Graph ---");
    println!("{}", t2.to_dot());

    println!("\n// Reshape操作はShapeTrackerを変更するだけで、");
    println!("// 計算グラフの構造（opやsrc）は変更しないことがわかります。");
}

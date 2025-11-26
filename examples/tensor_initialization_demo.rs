//! Tensor初期化APIのデモ
//!
//! Tensor::zeros(), Tensor::ones(), Tensor::full() の使用例

use harp_autograd::Tensor;

fn main() {
    env_logger::init();

    println!("=== Tensor Initialization Demo ===\n");

    // 1. zeros() - ゼロで埋められたテンソル
    println!("1. Tensor::zeros():");
    let zeros = Tensor::zeros(vec![2, 3]);
    println!("  Shape: {:?}", zeros.data.view.shape());
    println!("  DType: {:?}", zeros.data.dtype);
    println!("  Requires grad: {}", zeros.requires_grad());
    println!();

    // 2. ones() - 1で埋められたテンソル
    println!("2. Tensor::ones():");
    let ones = Tensor::ones(vec![3, 4]);
    println!("  Shape: {:?}", ones.data.view.shape());
    println!("  DType: {:?}", ones.data.dtype);
    println!("  Requires grad: {}", ones.requires_grad());
    println!();

    // 3. full() - 指定した値で埋められたテンソル
    println!("3. Tensor::full():");
    let fives = Tensor::full(vec![2, 2], 5.0);
    println!("  Shape: {:?}", fives.data.view.shape());
    println!("  DType: {:?}", fives.data.dtype);
    println!("  Value: 5.0");
    println!();

    // 4. 演算の例
    println!("4. Operations with initialized tensors:");
    let a = Tensor::zeros(vec![3]);
    let b = Tensor::ones(vec![3]);
    let c = &a + &b; // 0 + 1 = 1
    println!("  zeros([3]) + ones([3])");
    println!("  Result shape: {:?}", c.data.view.shape());
    println!();

    let x = Tensor::full(vec![2, 3], 2.0);
    let y = Tensor::full(vec![2, 3], 3.0);
    let z = &x * &y; // 2 * 3 = 6
    println!("  full([2, 3], 2.0) * full([2, 3], 3.0)");
    println!("  Result shape: {:?}", z.data.view.shape());
    println!();

    // 5. ブロードキャスティングの例
    println!("5. Broadcasting example:");
    let scalar = Tensor::full(vec![], 10.0); // スカラー
    let vector = Tensor::ones(vec![5]);
    let broadcasted = &scalar + &vector; // 10 + [1, 1, 1, 1, 1] = [11, 11, 11, 11, 11]
    println!("  scalar + vector([5])");
    println!("  Result shape: {:?}", broadcasted.data.view.shape());
    println!();

    // 6. Graphモジュールを使わない初期化の利点
    println!("6. Benefits of direct initialization:");
    println!("  - No need to create a Graph instance");
    println!("  - No need to specify dtype explicitly (defaults to F32)");
    println!("  - More concise and PyTorch-like API");
    println!("  - Perfect for quick prototyping and testing");
    println!();

    println!("=== Demo Complete ===");
}

//! Tensorのデバイス管理デモ
//!
//! PyTorch風のデバイス指定と遅延評価の組み合わせを示します

use harp::autograd::Tensor;
use harp::backend::Device;

fn main() {
    env_logger::init();

    println!("=== Tensor Device Management Demo ===\n");

    // 1. デフォルトデバイスで作成
    println!("1. Default device:");
    let x = Tensor::zeros(vec![3, 4]);
    println!("  Device: {}", x.device());
    println!("  Available: {}", x.device().is_available());
    println!();

    // 2. デバイス指定で作成
    println!("2. Creating tensors on specific devices:");
    let x_cpu = Tensor::ones_on(vec![2, 3], Device::cpu());
    println!("  CPU tensor device: {}", x_cpu.device());

    #[cfg(target_os = "macos")]
    {
        let x_metal = Tensor::ones_on(vec![2, 3], Device::metal(0));
        println!("  Metal tensor device: {}", x_metal.device());
    }

    let x_opencl = Tensor::ones_on(vec![2, 3], Device::opencl(0));
    if x_opencl.device().is_available() {
        println!("  OpenCL tensor device: {}", x_opencl.device());
    } else {
        println!("  OpenCL not available");
    }
    println!();

    // 3. .to()でデバイス変更
    println!("3. Moving tensors between devices:");
    let tensor = Tensor::full(vec![4, 5], 3.14);
    println!("  Original device: {}", tensor.device());

    let tensor_cpu = tensor.to(Device::cpu());
    println!("  After .to(cpu): {}", tensor_cpu.device());

    // 元のテンソルは変更されない
    println!("  Original still: {}", tensor.device());
    println!();

    // 4. 演算時のデバイス継承
    println!("4. Device inheritance in operations:");
    let a = Tensor::zeros_on(vec![3], Device::cpu());
    let b = Tensor::ones_on(vec![3], Device::cpu());

    let c = &a + &b;
    println!("  a.device() = {}", a.device());
    println!("  b.device() = {}", b.device());
    println!("  (a + b).device() = {}", c.device());

    let d = &c * 2.0;
    println!("  (c * 2.0).device() = {}", d.device());
    println!();

    // 5. 遅延評価とデバイス
    println!("5. Lazy evaluation with device:");
    let x = Tensor::ones(vec![5]);
    let y = &x * 2.0 + 1.0; // 計算グラフが構築されるだけ

    println!("  Computation graph built (not executed yet)");
    println!("  y.device() = {}", y.device());
    println!("  This device will be used when y.realize() is called");
    println!();

    // 6. デバイス情報のDebug表示
    println!("6. Debug output:");
    let tensor = Tensor::zeros_on(vec![2, 2], Device::cpu());
    println!("  {:?}", tensor);
    println!();

    // 7. 実用例：モデルの訓練を想定
    println!("7. Practical example (training loop concept):");
    let device = Device::auto_select();
    println!("  Selected device: {}", device);

    // パラメータをデバイスに配置
    let weights = Tensor::ones_on(vec![10, 5], device);
    let bias = Tensor::zeros_on(vec![5], device);

    println!("  weights.device() = {}", weights.device());
    println!("  bias.device() = {}", bias.device());

    // 入力もデバイスに配置（biasと同じ形状）
    let input = Tensor::full_on(vec![5], 0.5, device);
    println!("  input.device() = {}", input.device());

    // 演算（forward pass）
    let output = &input + &bias; // 実際の計算は realize() 時に実行される
    println!("  output.device() = {}", output.device());
    println!("  All tensors are on the same device!");
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Key points:");
    println!("  - Tensors hold device information (like PyTorch)");
    println!("  - Computation is lazy (like tinygrad)");
    println!("  - Device is inherited in operations");
    println!("  - Use .to(device) to move between devices");
    println!("  - Use .realize() to execute on the tensor's device");
}

//! Autograd realize() デモ
//!
//! tinygradスタイルのrealize()メソッドの使用例

use harp::backend::Device;
use harp::graph::{DType, Graph};
use harp_autograd::Tensor;

fn main() {
    env_logger::init();

    println!("=== Autograd realize() Demo ===\n");

    // 1. Graphの構築
    println!("1. Building computation graph:");
    let mut graph = Graph::new();
    let x = Tensor::from_graph_node(graph.input("x", DType::F32, vec![5]), true);

    // y = 2 * x + 3
    let y = &x * 2.0 + 3.0;
    println!("  Expression: y = 2 * x + 3");
    println!("  Input shape: {:?}", x.data.view.shape());
    println!("  Output shape: {:?}", y.data.view.shape());
    println!();

    // 2. 自動デバイス選択
    println!("2. Automatic device selection:");
    let device = Device::auto_select();
    println!("  Selected device: {}", device);
    println!("  Available: {}", device.is_available());
    println!();

    // 3. realize()の説明（実装は未完成）
    println!("3. realize() method (concept):");
    println!("  The realize() method would:");
    println!("  - Automatically select the best device (GPU > CPU)");
    println!("  - Compile the computation graph");
    println!("  - Execute on the selected device");
    println!("  - Return results as Vec<f32>");
    println!();

    println!("Example usage (conceptual):");
    println!("  ```");
    println!("  let mut inputs = HashMap::new();");
    println!("  inputs.insert(\"x\".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);");
    println!("  let result = y.realize_with(inputs)?;");
    println!("  // result = [5.0, 7.0, 9.0, 11.0, 13.0]");
    println!("  ```");
    println!();

    // 4. デバイスごとのPipeline取得（実際に動作）
    println!("4. Device pipeline management:");
    let devices = vec![
        Device::cpu(),
        #[cfg(target_os = "macos")]
        Device::metal(0),
        Device::opencl(0),
    ];

    for device in &devices {
        if device.is_available() {
            match device.get_pipeline() {
                Ok(pipeline) => {
                    println!("  ✓ {} - Pipeline ready", device);

                    // グラフのコンパイル（実行なし）
                    let mut g = Graph::new();
                    let a = g.input("a", DType::F32, vec![10]);
                    let b = g.input("b", DType::F32, vec![10]);
                    let c = a + b;
                    g.output("result", c);

                    match pipeline.borrow_mut().compile_graph(g) {
                        Ok(_) => println!("    - Compilation successful"),
                        Err(e) => println!("    - Compilation failed: {}", e),
                    }
                }
                Err(e) => println!("  ✗ {} - Pipeline error: {}", device, e),
            }
        } else {
            println!("  - {} - Not available", device);
        }
    }
    println!();

    // 5. 勾配計算との統合（コンセプト）
    println!("5. Integration with autograd:");
    println!("  forward:");
    println!("    result = y.realize_with(inputs)?;");
    println!();
    println!("  backward:");
    println!("    loss.backward();");
    println!("    grad = x.grad().unwrap();");
    println!("    grad_values = grad.realize_with(inputs)?;");
    println!();

    println!("=== Demo Complete ===");
    println!();
    println!("Note: Full realize() implementation requires:");
    println!("  - Kernel execution API");
    println!("  - Buffer management");
    println!("  - Input/output data handling");
    println!("  - Dynamic shape resolution");
}

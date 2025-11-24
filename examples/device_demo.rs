//! デバイスAPI使用例
//!
//! PyTorchの`torch.device`に相当する機能のデモンストレーション

use harp::backend::Device;
use harp::graph::{DType, Graph};
use std::rc::Rc;

fn main() {
    env_logger::init();

    println!("=== Harp Device API Demo ===\n");

    // 1. デバイスの作成と選択
    println!("1. デバイスの作成:");
    let cpu = Device::cpu();
    println!("  CPU device: {}", cpu);

    #[cfg(target_os = "macos")]
    {
        let metal = Device::metal(0);
        println!("  Metal device: {}", metal);
    }

    let opencl = Device::opencl(0);
    println!("  OpenCL device: {}", opencl);

    // 文字列からパース
    let device: Device = "cpu:0".parse().unwrap();
    println!("  Parsed from string 'cpu:0': {}", device);

    println!();

    // 2. 自動デバイス選択
    println!("2. 自動デバイス選択:");
    let auto_device = Device::auto_select();
    println!("  Auto-selected device: {}", auto_device);
    println!("  Available: {}", auto_device.is_available());
    println!();

    // 3. Pipelineの取得と共有
    println!("3. Pipeline共有のデモ:");
    let device1 = Device::cpu();
    let pipeline1 = device1.get_pipeline().unwrap();
    println!("  First pipeline acquired");

    let device2 = Device::cpu();
    let pipeline2 = device2.get_pipeline().unwrap();
    println!("  Second pipeline acquired");

    // 同じPipelineインスタンスが共有されていることを確認
    if Rc::ptr_eq(&pipeline1, &pipeline2) {
        println!("  ✓ Pipelines are shared (same instance)");
    } else {
        println!("  ✗ Pipelines are NOT shared");
    }
    println!();

    // 4. グラフのコンパイルとキャッシュ
    println!("4. カーネルキャッシュのデモ:");

    // グラフを作成
    let mut graph = Graph::new();
    let a = graph
        .input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![100])
        .build();
    let b = graph
        .input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![100])
        .build();
    let c = &a + &b;
    let d = &c * 2.0;
    graph.output("result", d);

    // コンパイルしてキャッシュ
    let key = "vector_add_mul".to_string();
    println!("  Compiling graph with key '{}'...", key);

    let result = pipeline1.borrow_mut().compile_and_cache(key.clone(), graph);
    match result {
        Ok(()) => println!("  ✓ Compilation successful and cached"),
        Err(e) => println!("  ✗ Compilation failed: {}", e),
    }

    // キャッシュから取得
    if pipeline1.borrow().has_cached_kernel(&key) {
        println!("  ✓ Kernel found in cache");
    } else {
        println!("  ✗ Kernel NOT found in cache");
    }

    // 同じデバイスから再度Pipelineを取得してもキャッシュが共有されていることを確認
    let pipeline3 = Device::cpu().get_pipeline().unwrap();
    if pipeline3.borrow().has_cached_kernel(&key) {
        println!("  ✓ Cache is shared across pipeline instances");
    } else {
        println!("  ✗ Cache is NOT shared");
    }
    println!();

    // 5. 複数デバイスの管理
    println!("5. 複数デバイスの管理:");
    let devices = vec![Device::cpu(), Device::opencl(0)];

    #[cfg(target_os = "macos")]
    let devices = {
        let mut d = devices;
        d.push(Device::metal(0));
        d
    };

    for device in &devices {
        if device.is_available() {
            match device.get_pipeline() {
                Ok(_) => println!("  ✓ {} pipeline ready", device),
                Err(e) => println!("  ✗ {} failed: {}", device, e),
            }
        } else {
            println!("  - {} not available", device);
        }
    }

    println!("\n=== Demo Complete ===");
}

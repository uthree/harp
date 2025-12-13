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

    #[cfg(target_os = "macos")]
    {
        let metal = Device::metal(0);
        println!("  Metal device: {}", metal);
    }

    let opencl = Device::opencl(0);
    println!("  OpenCL device: {}", opencl);

    // 文字列からパース
    let device: Device = "opencl:0".parse().unwrap();
    println!("  Parsed from string 'opencl:0': {}", device);

    println!();

    // 2. 自動デバイス選択
    println!("2. 自動デバイス選択:");
    let auto_device = Device::auto_select();
    println!("  Auto-selected device: {}", auto_device);
    println!("  Available: {}", auto_device.is_available());
    println!();

    // 3. Pipelineの取得と共有
    println!("3. Pipeline共有のデモ:");
    let device1 = Device::opencl(0);
    let pipeline1 = device1.get_pipeline().unwrap();
    println!("  First pipeline acquired");

    let device2 = Device::opencl(0);
    let pipeline2 = device2.get_pipeline().unwrap();
    println!("  Second pipeline acquired");

    // 同じPipelineインスタンスが共有されていることを確認
    if Rc::ptr_eq(&pipeline1, &pipeline2) {
        println!("  [OK] Pipelines are shared (same instance)");
    } else {
        println!("  [NG] Pipelines are NOT shared");
    }
    println!();

    // 4. グラフのコンパイルとキャッシュ
    println!("4. カーネルキャッシュのデモ:");

    // グラフを作成
    let mut graph = Graph::new();
    let a = graph.input("a", DType::F32, vec![100]);
    let b = graph.input("b", DType::F32, vec![100]);
    let c = &a + &b;
    let d = &c * 2.0;
    graph.output("result", d);

    // コンパイルしてキャッシュ
    let key = "vector_add_mul".to_string();
    println!("  Compiling graph with key '{}'...", key);

    let result = pipeline1.borrow_mut().compile_and_cache(key.clone(), graph);
    match result {
        Ok(()) => println!("  [OK] Compilation successful and cached"),
        Err(e) => println!("  [NG] Compilation failed: {}", e),
    }

    // キャッシュから取得
    if pipeline1.borrow().has_cached_kernel(&key) {
        println!("  [OK] Kernel found in cache");
    } else {
        println!("  [NG] Kernel NOT found in cache");
    }

    // 同じデバイスから再度Pipelineを取得してもキャッシュが共有されていることを確認
    let pipeline3 = Device::opencl(0).get_pipeline().unwrap();
    if pipeline3.borrow().has_cached_kernel(&key) {
        println!("  [OK] Cache is shared across pipeline instances");
    } else {
        println!("  [NG] Cache is NOT shared");
    }
    println!();

    // 5. 複数デバイスの管理
    println!("5. 複数デバイスの管理:");
    let mut devices = vec![Device::opencl(0)];

    #[cfg(target_os = "macos")]
    {
        devices.push(Device::metal(0));
    }

    for device in &devices {
        if device.is_available() {
            match device.get_pipeline() {
                Ok(_) => println!("  [OK] {} pipeline ready", device),
                Err(e) => println!("  [NG] {} failed: {}", device, e),
            }
        } else {
            println!("  [-] {} not available", device);
        }
    }

    println!("\n=== Demo Complete ===");
}

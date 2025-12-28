//! Fusion test - demonstrates operator fusion behavior
//!
//! Run with:
//! ```
//! RUST_LOG=harp::tensor=debug cargo run --features opencl --example fusion_test
//! ```

use harp::backend::{HarpDevice, set_device};
use harp::tensor::{Dim2, Tensor};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let device = HarpDevice::auto().expect("No available device");
    println!("Using device: {:?}", device.kind());
    set_device(device);

    // Test 1: Intermediate variable goes out of scope (true fusion)
    println!("=== Test 1: Chain with scoped intermediate (TRUE FUSION) ===");
    println!("Expected: intermediate c goes out of scope -> FUSED (strong_count=1)");
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::full([2, 3], 2.0);
    let d = Tensor::<f32, Dim2>::full([2, 3], 3.0);
    let e = {
        let c = &a + &b; // c goes out of scope after this block
        &c + &d
    };
    e.realize().unwrap();
    println!("Result: {:?}\n", e.data().unwrap());

    // Test 2: Intermediate variable stays in scope (no fusion due to strong_count)
    println!("=== Test 2: Chain with variable in scope (NO FUSION) ===");
    println!("Expected: c stays in scope -> strong_count=2 -> REALIZED");
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::full([2, 3], 2.0);
    let c = &a + &b; // c stays in scope
    let d = Tensor::<f32, Dim2>::full([2, 3], 3.0);
    let e = &c + &d;
    e.realize().unwrap();
    println!("Result: {:?}\n", e.data().unwrap());

    // Test 3: Branching (definitely no fusion)
    println!("=== Test 3: Branching (NO FUSION - multiple uses) ===");
    println!("Expected: c used in both e and g -> strong_count=3 -> REALIZED");
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::full([2, 3], 2.0);
    let c = &a + &b;
    let d = Tensor::<f32, Dim2>::full([2, 3], 3.0);
    let e = &c + &d;
    let f = Tensor::<f32, Dim2>::full([2, 3], 4.0);
    let g = &c + &f;
    e.realize().unwrap();
    g.realize().unwrap();
    println!("Result e: {:?}", e.data().unwrap());
    println!("Result g: {:?}", g.data().unwrap());
}

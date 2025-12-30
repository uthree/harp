//! Cross-backend consistency tests
//!
//! These tests verify that different backends produce the same results.
//! Run with: `cargo test --features "metal,opencl"` (macOS only)

#![cfg(all(feature = "metal", feature = "opencl", target_os = "macos"))]

mod common;

use common::vec_approx_eq;
use harp::backend::global::{DeviceKind, set_default_device};
use harp::metal::MetalDevice;
use harp::opencl::OpenCLDevice;
use harp::tensor::{DimDyn, Tensor};

#[test]
fn test_cross_backend_consistency() {
    // Setup Metal
    let metal_device = match MetalDevice::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    // Setup OpenCL
    let opencl_device = match OpenCLDevice::new() {
        Ok(d) => d,
        Err(_) => return,
    };

    // Create input data
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    // Test on Metal
    set_default_device(metal_device.clone(), DeviceKind::Metal);
    let a_metal = Tensor::<f32, DimDyn>::from_data(input_data.clone(), vec![2, 4]);
    let b_metal = Tensor::<f32, DimDyn>::from_data(vec![1.0; 8], vec![2, 4]);
    let c_metal = (&a_metal + &b_metal) * 2.0;
    c_metal.realize().expect("Metal execution failed");
    let metal_result = c_metal.data().expect("Metal data missing");

    // Test on OpenCL
    set_default_device(opencl_device.clone(), DeviceKind::OpenCL);
    let a_opencl = Tensor::<f32, DimDyn>::from_data(input_data.clone(), vec![2, 4]);
    let b_opencl = Tensor::<f32, DimDyn>::from_data(vec![1.0; 8], vec![2, 4]);
    let c_opencl = (&a_opencl + &b_opencl) * 2.0;
    c_opencl.realize().expect("OpenCL execution failed");
    let opencl_result = c_opencl.data().expect("OpenCL data missing");

    // Compare results
    assert!(
        vec_approx_eq(&metal_result, &opencl_result),
        "Cross-backend mismatch:\nMetal:  {:?}\nOpenCL: {:?}",
        metal_result,
        opencl_result
    );
}

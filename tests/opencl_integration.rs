//! OpenCL Backend Integration Tests
//!
//! Run with: `cargo test --features opencl`

#![cfg(feature = "opencl")]

mod common;

use common::{approx_eq, vec_approx_eq};
use harp::backend::global::{DeviceKind, set_default_device};
use harp::opencl::OpenCLDevice;
use harp::tensor::{DimDyn, Sqrt, Tensor};

fn setup_opencl() -> bool {
    match OpenCLDevice::new() {
        Ok(device) => {
            set_default_device(device, DeviceKind::OpenCL);
            true
        }
        Err(e) => {
            eprintln!("OpenCL device not available: {:?}", e);
            false
        }
    }
}

#[test]
fn test_opencl_const_fill() {
    if !setup_opencl() {
        return;
    }

    let t = Tensor::<f32, DimDyn>::full_dyn(&[2, 3], 2.5);
    let result = t.realize();

    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = t.data().expect("No data after realize");
    assert_eq!(data.len(), 6);
    assert!(data.iter().all(|&x| approx_eq(x, 2.5)));
}

#[test]
fn test_opencl_add() {
    if !setup_opencl() {
        return;
    }

    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::<f32, DimDyn>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
    let c = &a + &b;

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    let expected = vec![6.0, 8.0, 10.0, 12.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Add mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_mul() {
    if !setup_opencl() {
        return;
    }

    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::<f32, DimDyn>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
    let c = &a * &b;

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    let expected = vec![2.0, 6.0, 12.0, 20.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Mul mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_scalar_add() {
    if !setup_opencl() {
        return;
    }

    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let c = &a + 10.0;

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    let expected = vec![11.0, 12.0, 13.0, 14.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Scalar add mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_neg() {
    if !setup_opencl() {
        return;
    }

    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, -2.0, 3.0, -4.0], vec![4]);
    let c = -&a;

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    let expected = vec![-1.0, 2.0, -3.0, 4.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Neg mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_sqrt() {
    if !setup_opencl() {
        return;
    }

    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 4.0, 9.0, 16.0], vec![4]);
    let c = a.sqrt();

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    let expected = vec![1.0, 2.0, 3.0, 4.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Sqrt mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_fused_ops() {
    if !setup_opencl() {
        return;
    }

    // Test fused operation: (a + b) * c
    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
    let b = Tensor::<f32, DimDyn>::from_data(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
    let c = Tensor::<f32, DimDyn>::from_data(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
    let result_tensor = (&a + &b) * &c;

    let result = result_tensor.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = result_tensor.data().expect("No data after realize");
    // (1+1)*2=4, (2+1)*2=6, (3+1)*2=8, (4+1)*2=10
    let expected = vec![4.0, 6.0, 8.0, 10.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Fused ops mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_reduce_sum() {
    if !setup_opencl() {
        return;
    }

    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let sum = a.sum(1);

    let result = sum.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = sum.data().expect("No data after realize");
    // Row 0: 1+2+3=6, Row 1: 4+5+6=15
    let expected = vec![6.0, 15.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Reduce sum mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

// ========================================================================
// Complex Graph Tests
// ========================================================================

#[test]
fn test_opencl_matmul() {
    if !setup_opencl() {
        return;
    }

    // 2x3 @ 3x2 -> 2x2
    // A = [[1, 2, 3], [4, 5, 6]]
    // B = [[1, 2], [3, 4], [5, 6]]
    // C = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
    //   = [[22, 28], [49, 64]]
    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
    let c = a.matmul(&b);

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    let expected = vec![22.0, 28.0, 49.0, 64.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Matmul mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_contiguous_with_ops() {
    if !setup_opencl() {
        return;
    }

    // Test: transpose -> contiguous -> add
    // This tests that contiguous correctly materializes the transposed view
    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    // a = [[1, 2, 3], [4, 5, 6]]
    // a.T = [[1, 4], [2, 5], [3, 6]]
    let a_t = a.transpose().contiguous();
    let b = Tensor::<f32, DimDyn>::from_data(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
    let c = &a_t + &b;

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    // [[1+10, 4+20], [2+30, 5+40], [3+50, 6+60]] = [[11, 24], [32, 45], [53, 66]]
    let expected = vec![11.0, 24.0, 32.0, 45.0, 53.0, 66.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Contiguous + ops mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_deep_fusion_chain() {
    if !setup_opencl() {
        return;
    }

    // Test: ((a + b) * c - d) / e
    // Tests deep fusion of multiple elementwise operations
    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::<f32, DimDyn>::from_data(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);
    let c = Tensor::<f32, DimDyn>::from_data(vec![3.0, 3.0, 3.0, 3.0], vec![2, 2]);
    let d = Tensor::<f32, DimDyn>::from_data(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    let e = Tensor::<f32, DimDyn>::from_data(vec![2.0, 2.0, 2.0, 2.0], vec![2, 2]);

    // ((1+2)*3-1)/2 = (9-1)/2 = 4
    // ((2+2)*3-1)/2 = (12-1)/2 = 5.5
    // ((3+2)*3-1)/2 = (15-1)/2 = 7
    // ((4+2)*3-1)/2 = (18-1)/2 = 8.5
    let result_tensor = ((&a + &b) * &c - &d) / &e;

    let result = result_tensor.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = result_tensor.data().expect("No data after realize");
    let expected = vec![4.0, 5.5, 7.0, 8.5];
    assert!(
        vec_approx_eq(&data, &expected),
        "Deep fusion chain mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_transpose_reduce() {
    if !setup_opencl() {
        return;
    }

    // Test: transpose -> reduce_sum
    // This tests reducing on transposed axes
    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    // a = [[1, 2, 3], [4, 5, 6]]
    // a.T = [[1, 4], [2, 5], [3, 6]]
    // Sum along axis 1 of transposed: [1+4, 2+5, 3+6] = [5, 7, 9]
    let a_t = a.transpose();
    let sum = a_t.sum(1);

    let result = sum.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = sum.data().expect("No data after realize");
    let expected = vec![5.0, 7.0, 9.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Transpose + reduce mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_reshape_ops() {
    if !setup_opencl() {
        return;
    }

    // Test: reshape -> ops -> reshape
    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let a_flat = a.reshape_dyn(&[6]);
    let b = Tensor::<f32, DimDyn>::from_data(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![6]);
    let c = &a_flat + &b;
    let c_reshaped = c.reshape_dyn(&[3, 2]);

    let result = c_reshaped.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c_reshaped.data().expect("No data after realize");
    // [2, 3, 4, 5, 6, 7] reshaped to [3, 2]
    let expected = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Reshape ops mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_reduce_max() {
    if !setup_opencl() {
        return;
    }

    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], vec![2, 3]);
    let max = a.max(1);

    let result = max.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = max.data().expect("No data after realize");
    // Row 0: max(1, 5, 3) = 5, Row 1: max(2, 8, 4) = 8
    let expected = vec![5.0, 8.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Reduce max mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

#[test]
fn test_opencl_broadcast_add() {
    if !setup_opencl() {
        return;
    }

    // Test broadcasting: [2, 3] + [3] -> [2, 3]
    let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let b = Tensor::<f32, DimDyn>::from_data(vec![10.0, 20.0, 30.0], vec![3]);
    let c = &a + &b;

    let result = c.realize();
    assert!(result.is_ok(), "realize() failed: {:?}", result.err());

    let data = c.data().expect("No data after realize");
    // [[1+10, 2+20, 3+30], [4+10, 5+20, 6+30]] = [[11, 22, 33], [14, 25, 36]]
    let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
    assert!(
        vec_approx_eq(&data, &expected),
        "Broadcast add mismatch: {:?} vs {:?}",
        data,
        expected
    );
}

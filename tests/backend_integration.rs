//! Backend Integration Tests
//!
//! These tests verify that the full computation pipeline produces correct results
//! on each backend (Metal, OpenCL).
//!
//! Run with:
//! - `cargo test --features metal` for Metal backend (macOS only)
//! - `cargo test --features opencl` for OpenCL backend

// Test Utilities - used by feature-specific test modules
#[allow(dead_code)]
const EPSILON: f32 = 1e-5;

#[allow(dead_code)]
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

#[allow(dead_code)]
fn vec_approx_eq(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y))
}

// ============================================================================
// Metal Backend Tests
// ============================================================================

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_tests {
    use super::*;
    use harp::backend::metal::MetalDevice;

    fn setup_metal() -> bool {
        match MetalDevice::new() {
            Ok(device) => {
                set_default_device(device, DeviceKind::Metal);
                true
            }
            Err(e) => {
                eprintln!("Metal device not available: {:?}", e);
                false
            }
        }
    }

    #[test]
    fn test_metal_const_fill() {
        if !setup_metal() {
            return;
        }

        let t = Tensor::<f32, DimDyn>::full_dyn(&[2, 3], 3.14);
        let result = t.realize();

        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        assert_eq!(data.len(), 6);
        assert!(data.iter().all(|&x| approx_eq(x, 3.14)));
    }

    #[test]
    fn test_metal_add() {
        if !setup_metal() {
            return;
        }

        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f32, DimDyn>::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = &a + &b;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![6.0, 8.0, 10.0, 12.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Add mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_mul() {
        if !setup_metal() {
            return;
        }

        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::<f32, DimDyn>::from_data(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]);
        let c = &a * &b;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![2.0, 6.0, 12.0, 20.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Mul mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_scalar_add() {
        if !setup_metal() {
            return;
        }

        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let c = &a + 10.0;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![11.0, 12.0, 13.0, 14.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Scalar add mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_neg() {
        if !setup_metal() {
            return;
        }

        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, -2.0, 3.0, -4.0], vec![4]);
        let c = -&a;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![-1.0, 2.0, -3.0, 4.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Neg mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_sqrt() {
        if !setup_metal() {
            return;
        }

        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 4.0, 9.0, 16.0], vec![4]);
        let c = a.sqrt();

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![1.0, 2.0, 3.0, 4.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Sqrt mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_fused_ops() {
        if !setup_metal() {
            return;
        }

        // Test fused operation: (a + b) * c
        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = Tensor::<f32, DimDyn>::from_data(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let c = Tensor::<f32, DimDyn>::from_data(vec![2.0, 2.0, 2.0, 2.0], vec![4]);
        let result_tensor = (&a + &b) * &c;

        let result = result_tensor.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
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
    fn test_metal_reduce_sum() {
        if !setup_metal() {
            return;
        }

        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let sum = a.sum(1);

        let result = sum.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        // Row 0: 1+2+3=6, Row 1: 4+5+6=15
        let expected = vec![6.0, 15.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Reduce sum mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_with_ndarray() {
        if !setup_metal() {
            return;
        }

        use harp::tensor::Dim2;

        // Create tensors from ndarray
        let arr_a: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let arr_b: Array2<f32> = array![[5.0, 6.0], [7.0, 8.0]];

        let a = Tensor::<f32, Dim2>::from_ndarray(&arr_a);
        let b = Tensor::<f32, Dim2>::from_ndarray(&arr_b);
        let c = &a + &b;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        // Convert back to ndarray and verify
        let output: Array2<f32> = result.unwrap().to_ndarray().expect("No data after realize");
        let expected: Array2<f32> = array![[6.0, 8.0], [10.0, 12.0]];

        assert_eq!(output.shape(), expected.shape());
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b), "Mismatch: {} vs {}", a, b);
        }
    }

    // ========================================================================
    // Complex Graph Tests
    // ========================================================================

    #[test]
    fn test_metal_matmul() {
        if !setup_metal() {
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

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![22.0, 28.0, 49.0, 64.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Matmul mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_contiguous_with_ops() {
        if !setup_metal() {
            return;
        }

        // Test: transpose -> contiguous -> add
        // This tests that contiguous correctly materializes the transposed view
        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        // a = [[1, 2, 3], [4, 5, 6]]
        // a.T = [[1, 4], [2, 5], [3, 6]]
        let a_t = a.transpose().contiguous();
        let b =
            Tensor::<f32, DimDyn>::from_data(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
        let c = &a_t + &b;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
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
    fn test_metal_deep_fusion_chain() {
        if !setup_metal() {
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

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![4.0, 5.5, 7.0, 8.5];
        assert!(
            vec_approx_eq(&data, &expected),
            "Deep fusion chain mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_transpose_reduce() {
        if !setup_metal() {
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

        let data = result.unwrap().data().expect("No data after realize");
        let expected = vec![5.0, 7.0, 9.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Transpose + reduce mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_metal_reshape_ops() {
        if !setup_metal() {
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

        let data = result.unwrap().data().expect("No data after realize");
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
    fn test_metal_reduce_max() {
        if !setup_metal() {
            return;
        }

        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], vec![2, 3]);
        let max = a.max(1);

        let result = max.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
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
    fn test_metal_broadcast_add() {
        if !setup_metal() {
            return;
        }

        // Test broadcasting: [2, 3] + [3] -> [2, 3]
        let a = Tensor::<f32, DimDyn>::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::<f32, DimDyn>::from_data(vec![10.0, 20.0, 30.0], vec![3]);
        let c = &a + &b;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        // [[1+10, 2+20, 3+30], [4+10, 5+20, 6+30]] = [[11, 22, 33], [14, 25, 36]]
        let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Broadcast add mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }
}

// ============================================================================
// OpenCL Backend Tests
// ============================================================================

#[cfg(feature = "opencl")]
mod opencl_tests {
    use super::*;
    use harp::backend::opencl::OpenCLDevice;

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

        let t = Tensor::<f32, DimDyn>::full_dyn(&[2, 3], 3.14);
        let result = t.realize();

        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
        assert_eq!(data.len(), 6);
        assert!(data.iter().all(|&x| approx_eq(x, 3.14)));
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
        // Row 0: 1+2+3=6, Row 1: 4+5+6=15
        let expected = vec![6.0, 15.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Reduce sum mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }

    #[test]
    fn test_opencl_with_ndarray() {
        if !setup_opencl() {
            return;
        }

        use harp::tensor::Dim2;

        // Create tensors from ndarray
        let arr_a: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let arr_b: Array2<f32> = array![[5.0, 6.0], [7.0, 8.0]];

        let a = Tensor::<f32, Dim2>::from_ndarray(&arr_a);
        let b = Tensor::<f32, Dim2>::from_ndarray(&arr_b);
        let c = &a + &b;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        // Convert back to ndarray and verify
        let output: Array2<f32> = result.unwrap().to_ndarray().expect("No data after realize");
        let expected: Array2<f32> = array![[6.0, 8.0], [10.0, 12.0]];

        assert_eq!(output.shape(), expected.shape());
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!(approx_eq(*a, *b), "Mismatch: {} vs {}", a, b);
        }
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

        let data = result.unwrap().data().expect("No data after realize");
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
        let b =
            Tensor::<f32, DimDyn>::from_data(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], vec![3, 2]);
        let c = &a_t + &b;

        let result = c.realize();
        assert!(result.is_ok(), "realize() failed: {:?}", result.err());

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
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

        let data = result.unwrap().data().expect("No data after realize");
        // [[1+10, 2+20, 3+30], [4+10, 5+20, 6+30]] = [[11, 22, 33], [14, 25, 36]]
        let expected = vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        assert!(
            vec_approx_eq(&data, &expected),
            "Broadcast add mismatch: {:?} vs {:?}",
            data,
            expected
        );
    }
}

// ============================================================================
// Cross-backend consistency tests (when both are available)
// ============================================================================

#[cfg(all(feature = "metal", feature = "opencl", target_os = "macos"))]
mod cross_backend_tests {
    use super::*;
    use harp::backend::metal::MetalDevice;
    use harp::backend::opencl::OpenCLDevice;

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
        let metal_result = c_metal
            .realize()
            .ok()
            .and_then(|t| t.data())
            .expect("Metal execution failed");

        // Test on OpenCL
        set_default_device(opencl_device.clone(), DeviceKind::OpenCL);
        let a_opencl = Tensor::<f32, DimDyn>::from_data(input_data.clone(), vec![2, 4]);
        let b_opencl = Tensor::<f32, DimDyn>::from_data(vec![1.0; 8], vec![2, 4]);
        let c_opencl = (&a_opencl + &b_opencl) * 2.0;
        let opencl_result = c_opencl
            .realize()
            .ok()
            .and_then(|t| t.data())
            .expect("OpenCL execution failed");

        // Compare results
        assert!(
            vec_approx_eq(&metal_result, &opencl_result),
            "Cross-backend mismatch:\nMetal:  {:?}\nOpenCL: {:?}",
            metal_result,
            opencl_result
        );
    }
}

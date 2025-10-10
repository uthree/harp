#[cfg(feature = "backend-c")]
mod tests {
    use harp::tensor::{Tensor, Tensor1};

    #[test]
    fn test_tensor_add() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3], "c");

        let c = a + b;

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_sub() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![5.0, 7.0, 9.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");

        let c = a - b;

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_mul() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![5.0, 6.0, 7.0], &[3], "c");

        let c = a * b;

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_div() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![2.0, 4.0, 5.0], &[3], "c");

        let c = a / b;

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_neg() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, -2.0, 3.0], &[3], "c");

        let c = -a;

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_recip() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![2.0, 4.0, 5.0], &[3], "c");

        let c = a.recip();

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_sin() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3], "c");

        let c = a.sin();

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_sqrt() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![4.0, 9.0, 16.0], &[3], "c");

        let c = a.sqrt();

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_log2() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![2.0, 4.0, 8.0], &[3], "c");

        let c = a.log2();

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_exp2() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");

        let c = a.exp2();

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_ln() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.718, 7.389], &[3], "c");

        let c = a.ln();

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_exp() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[3], "c");

        let c = a.exp();

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_max() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 5.0, 3.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![2.0, 4.0, 6.0], &[3], "c");

        let c = a.max(b);

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_tensor_pow() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![2.0, 2.0, 2.0], &[3], "c");

        let c = a.pow(b);

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_chained_operations() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3], "c");
        let b2: Tensor1<f32> = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3], "c");

        // (a + b) * b2
        let c = (a + b) * b2;

        // Shape should be preserved
        assert_eq!(c.shape(), &[3]);
        assert_eq!(c.ndim(), 1);
    }

    #[test]
    fn test_requires_grad() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c").enable_grad();

        assert!(a.is_requires_grad());
    }

    #[test]
    fn test_grad_methods() {
        let mut a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c").enable_grad();

        // Initially no gradient
        assert!(a.grad().is_none());

        // After backward pass, gradient should be set
        a.backward();
        assert!(a.grad().is_some());

        // Test zero_grad
        a.zero_grad();
        assert!(a.grad().is_none());
    }

    #[test]
    fn test_complex_expression() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");
        let b: Tensor1<f32> = Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3], "c");
        let c: Tensor1<f32> = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3], "c");

        // (a * b + c).sqrt()
        let result = (a * b + c).sqrt();

        assert_eq!(result.shape(), &[3]);
    }

    #[test]
    fn test_tensor_sum() {
        use harp::tensor::{Tensor2, TensorDyn};

        let a: Tensor2<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], "c");

        // Sum along axis 0 (result shape: [3])
        let sum_axis0: TensorDyn<f32> = a.clone().sum(0);
        assert_eq!(sum_axis0.shape(), &[3]);
        assert_eq!(sum_axis0.ndim(), 1);

        // Sum along axis 1 (result shape: [2])
        let sum_axis1: TensorDyn<f32> = a.sum(1);
        assert_eq!(sum_axis1.shape(), &[2]);
        assert_eq!(sum_axis1.ndim(), 1);
    }

    #[test]
    fn test_tensor_product() {
        use harp::tensor::{Tensor2, TensorDyn};

        let a: Tensor2<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], "c");

        // Product along axis 0
        let prod_axis0: TensorDyn<f32> = a.clone().product(0);
        assert_eq!(prod_axis0.shape(), &[3]);

        // Product along axis 1
        let prod_axis1: TensorDyn<f32> = a.product(1);
        assert_eq!(prod_axis1.shape(), &[2]);
    }

    #[test]
    fn test_tensor_reduce_max() {
        use harp::tensor::{Tensor2, TensorDyn};

        let a: Tensor2<f32> = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], "c");

        // Max along axis 0
        let max_axis0: TensorDyn<f32> = a.clone().reduce_max(0);
        assert_eq!(max_axis0.shape(), &[3]);

        // Max along axis 1
        let max_axis1: TensorDyn<f32> = a.reduce_max(1);
        assert_eq!(max_axis1.shape(), &[2]);
    }

    #[test]
    fn test_tensor_cumsum() {
        use harp::tensor::Tensor2;

        let a: Tensor2<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], "c");

        // Cumsum along axis 0 (shape preserved: [2, 3])
        let cumsum_axis0 = a.clone().cumsum(0);
        assert_eq!(cumsum_axis0.shape(), &[2, 3]);
        assert_eq!(cumsum_axis0.ndim(), 2);

        // Cumsum along axis 1 (shape preserved: [2, 3])
        let cumsum_axis1 = a.cumsum(1);
        assert_eq!(cumsum_axis1.shape(), &[2, 3]);
        assert_eq!(cumsum_axis1.ndim(), 2);
    }

    #[test]
    fn test_tensor_cumprod() {
        use harp::tensor::Tensor2;

        let a: Tensor2<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 2.0, 2.0, 2.0], &[2, 3], "c");

        // Cumprod along axis 0 (shape preserved)
        let cumprod_axis0 = a.clone().cumprod(0);
        assert_eq!(cumprod_axis0.shape(), &[2, 3]);

        // Cumprod along axis 1 (shape preserved)
        let cumprod_axis1 = a.cumprod(1);
        assert_eq!(cumprod_axis1.shape(), &[2, 3]);
    }

    #[test]
    fn test_tensor_cummax() {
        use harp::tensor::Tensor2;

        let a: Tensor2<f32> = Tensor::from_vec(vec![1.0, 5.0, 3.0, 2.0, 1.0, 4.0], &[2, 3], "c");

        // Cummax along axis 0 (shape preserved)
        let cummax_axis0 = a.clone().cummax(0);
        assert_eq!(cummax_axis0.shape(), &[2, 3]);

        // Cummax along axis 1 (shape preserved)
        let cummax_axis1 = a.cummax(1);
        assert_eq!(cummax_axis1.shape(), &[2, 3]);
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn test_sum_axis_out_of_bounds() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");
        let _ = a.sum(1); // Should panic, 1D tensor only has axis 0
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn test_cumsum_axis_out_of_bounds() {
        let a: Tensor1<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");
        let _ = a.cumsum(1); // Should panic, 1D tensor only has axis 0
    }

    #[test]
    fn test_chained_reduce_operations() {
        use harp::tensor::{Tensor2, TensorDyn};

        let a: Tensor2<f32> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], "c");

        // Sum along axis 1, then use the result
        let sum_result: TensorDyn<f32> = a.sum(1);
        assert_eq!(sum_result.shape(), &[2]);
    }
}

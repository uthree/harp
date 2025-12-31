//! Tests for movement primitive operations

use crate::tensor::ops::PadValue;
use crate::tensor::{Dim2, Dim3, Dim4, Dim5, Dim6, Dim8, DimDyn, Tensor};

#[test]
fn test_squeeze() {
    let a = Tensor::<f32, DimDyn>::ones_dyn(&[1, 2, 3]);
    let b = a.squeeze(0);
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
fn test_unsqueeze() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.unsqueeze(0);
    assert_eq!(b.shape(), &[1, 2, 3]);
}

#[test]
fn test_reshape() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.reshape([6]);
    assert_eq!(b.shape(), &[6]);
}

#[test]
fn test_reshape_dyn() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.reshape_dyn(&[3, 2]);
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
fn test_permute() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.permute(&[1, 0]);
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
fn test_transpose() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.transpose();
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
fn test_expand() {
    let a = Tensor::<f32, Dim2>::ones([1, 3]);
    let b = a.expand(&[4, 3]);
    assert_eq!(b.shape(), &[4, 3]);
}

#[test]
fn test_flatten() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.flatten();
    assert_eq!(b.shape(), &[6]);
}

#[test]
fn test_contiguous() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.contiguous();
    assert_eq!(b.shape(), &[2, 3]);
}

// f64 tests
#[test]
fn test_unsqueeze_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]);
    let b = a.unsqueeze(0);
    assert_eq!(b.shape(), &[1, 2, 3]);
}

#[test]
fn test_reshape_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]);
    let b = a.reshape([6]);
    assert_eq!(b.shape(), &[6]);
}

#[test]
fn test_permute_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]);
    let b = a.permute(&[1, 0]);
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
fn test_transpose_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]);
    let b = a.transpose();
    assert_eq!(b.shape(), &[3, 2]);
}

#[test]
fn test_expand_f64() {
    let a = Tensor::<f64, Dim2>::ones([1, 3]);
    let b = a.expand(&[4, 3]);
    assert_eq!(b.shape(), &[4, 3]);
}

#[test]
fn test_flatten_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]);
    let b = a.flatten();
    assert_eq!(b.shape(), &[6]);
}

#[test]
fn test_contiguous_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]);
    let b = a.contiguous();
    assert_eq!(b.shape(), &[2, 3]);
}

// Pad tests
#[test]
fn test_pad_shape() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.pad(&[(1, 2), (0, 1)], PadValue::Zero);
    assert_eq!(b.shape(), &[5, 4]); // [2+1+2, 3+0+1]
}

#[test]
fn test_pad_zero_shape() {
    let a = Tensor::<f32, Dim2>::ones([3, 4]);
    let b = a.pad_zero(&[(2, 1), (1, 3)]);
    assert_eq!(b.shape(), &[6, 8]); // [3+2+1, 4+1+3]
}

#[test]
fn test_pad_no_padding() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = a.pad(&[(0, 0), (0, 0)], PadValue::Zero);
    assert_eq!(b.shape(), &[2, 3]); // No change
}

#[test]
fn test_pad_1d() {
    use crate::tensor::Dim1;
    let a = Tensor::<f32, Dim1>::ones([5]);
    let b = a.pad(&[(2, 3)], PadValue::One);
    assert_eq!(b.shape(), &[10]); // 5+2+3
}

// Type-safe dimension tests
#[test]
fn test_squeeze_type_safe() {
    use crate::tensor::Dim1;

    // Dim3 -> Dim2 (squeeze one dimension)
    let a = Tensor::<f32, Dim3>::ones([2, 1, 3]);
    let b: Tensor<f32, Dim2> = a.squeeze(1);
    assert_eq!(b.shape(), &[2, 3]);

    // Dim2 -> Dim1 (squeeze one dimension)
    let c = Tensor::<f32, Dim2>::ones([1, 5]);
    let d: Tensor<f32, Dim1> = c.squeeze(0);
    assert_eq!(d.shape(), &[5]);
}

#[test]
fn test_unsqueeze_type_safe() {
    use crate::tensor::Dim1;

    // Dim2 -> Dim3 (unsqueeze adds one dimension)
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b: Tensor<f32, Dim3> = a.unsqueeze(0);
    assert_eq!(b.shape(), &[1, 2, 3]);

    // Dim1 -> Dim2 (unsqueeze adds one dimension)
    let c = Tensor::<f32, Dim1>::ones([5]);
    let d: Tensor<f32, Dim2> = c.unsqueeze(1);
    assert_eq!(d.shape(), &[5, 1]);
}

#[test]
fn test_pad_type_safe() {
    // Dim2 -> Dim2 (pad preserves dimension)
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b: Tensor<f32, Dim2> = a.pad(&[(1, 1), (2, 2)], PadValue::Zero);
    assert_eq!(b.shape(), &[4, 7]);

    // Dim3 -> Dim3 (pad preserves dimension)
    let c = Tensor::<f32, Dim3>::ones([2, 3, 4]);
    let d: Tensor<f32, Dim3> = c.pad_zero(&[(0, 1), (1, 0), (2, 2)]);
    assert_eq!(d.shape(), &[3, 4, 8]);
}

#[test]
fn test_chained_squeeze_unsqueeze() {
    use crate::tensor::Dim1;

    // Dim2 -> Dim3 -> Dim2
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b: Tensor<f32, Dim3> = a.unsqueeze(0);
    let c: Tensor<f32, Dim2> = b.squeeze(0);
    assert_eq!(c.shape(), &[2, 3]);

    // Dim1 -> Dim2 -> Dim1
    let d = Tensor::<f32, Dim1>::ones([5]);
    let e: Tensor<f32, Dim2> = d.unsqueeze(0);
    let f: Tensor<f32, Dim1> = e.squeeze(0);
    assert_eq!(f.shape(), &[5]);
}

// Slice tests
#[test]
fn test_slice_basic() {
    let a = Tensor::<f32, Dim2>::ones([4, 5]);
    let b = a.slice(&[(1, 3), (2, 5)]);
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
fn test_slice_full_range() {
    let a = Tensor::<f32, Dim2>::ones([4, 5]);
    let b = a.slice(&[(0, 4), (0, 5)]);
    assert_eq!(b.shape(), &[4, 5]);
}

#[test]
fn test_slice_single_element() {
    use crate::tensor::Dim1;
    let a = Tensor::<f32, Dim1>::ones([10]);
    let b = a.slice(&[(5, 6)]);
    assert_eq!(b.shape(), &[1]);
}

#[test]
fn test_slice_3d() {
    let a = Tensor::<f32, Dim3>::ones([4, 5, 6]);
    let b = a.slice(&[(1, 3), (0, 5), (2, 4)]);
    assert_eq!(b.shape(), &[2, 5, 2]);
}

#[test]
fn test_slice_type_safe() {
    // Slice preserves dimension type
    let a = Tensor::<f32, Dim2>::ones([4, 5]);
    let b: Tensor<f32, Dim2> = a.slice(&[(1, 3), (2, 5)]);
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
fn test_slice_f64() {
    let a = Tensor::<f64, Dim2>::ones([4, 5]);
    let b = a.slice(&[(0, 2), (1, 4)]);
    assert_eq!(b.shape(), &[2, 3]);
}

#[test]
#[should_panic(expected = "must be less than end")]
fn test_slice_invalid_range() {
    let a = Tensor::<f32, Dim2>::ones([4, 5]);
    let _ = a.slice(&[(3, 1), (0, 5)]); // start > end
}

#[test]
#[should_panic(expected = "exceeds dimension size")]
fn test_slice_out_of_bounds() {
    let a = Tensor::<f32, Dim2>::ones([4, 5]);
    let _ = a.slice(&[(0, 4), (0, 10)]); // end > dim size
}

// Concat tests
#[test]
fn test_concat_axis0() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::ones([4, 3]);
    let c = Tensor::concat(&[&a, &b], 0);
    assert_eq!(c.shape(), &[6, 3]);
}

#[test]
fn test_concat_axis1() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::ones([2, 5]);
    let c = Tensor::concat(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 8]);
}

#[test]
fn test_concat_multiple() {
    let a = Tensor::<f32, Dim2>::ones([1, 3]);
    let b = Tensor::<f32, Dim2>::ones([2, 3]);
    let c = Tensor::<f32, Dim2>::ones([3, 3]);
    let d = Tensor::concat(&[&a, &b, &c], 0);
    assert_eq!(d.shape(), &[6, 3]);
}

#[test]
fn test_concat_3d() {
    let a = Tensor::<f32, Dim3>::ones([2, 3, 4]);
    let b = Tensor::<f32, Dim3>::ones([2, 5, 4]);
    let c = Tensor::concat(&[&a, &b], 1);
    assert_eq!(c.shape(), &[2, 8, 4]);
}

#[test]
fn test_concat_type_safe() {
    // Concat preserves dimension type
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::ones([4, 3]);
    let c: Tensor<f32, Dim2> = Tensor::concat(&[&a, &b], 0);
    assert_eq!(c.shape(), &[6, 3]);
}

#[test]
fn test_concat_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]);
    let b = Tensor::<f64, Dim2>::ones([4, 3]);
    let c = Tensor::concat(&[&a, &b], 0);
    assert_eq!(c.shape(), &[6, 3]);
}

#[test]
#[should_panic(expected = "Cannot concatenate empty")]
fn test_concat_empty() {
    let empty: &[&Tensor<f32, Dim2>] = &[];
    let _ = Tensor::concat(empty, 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_concat_invalid_axis() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::ones([4, 3]);
    let _ = Tensor::concat(&[&a, &b], 5);
}

#[test]
#[should_panic(expected = "same shape on non-axis")]
fn test_concat_shape_mismatch() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]);
    let b = Tensor::<f32, Dim2>::ones([4, 5]); // Different non-axis dimension
    let _ = Tensor::concat(&[&a, &b], 0);
}

// ========================================================================
// Gradient tests for pad, slice, concat
// ========================================================================

#[test]
fn test_pad_backward_shape() {
    // Test that PadBackward produces correct gradient shape
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let padded = a.pad(&[(1, 2), (0, 1)], PadValue::Zero);

    assert!(padded.requires_grad());
    assert_eq!(padded.shape(), &[5, 4]); // [2+1+2, 3+0+1]

    // Backward
    padded.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
}

#[test]
fn test_pad_backward_f64() {
    let a = Tensor::<f64, Dim2>::ones([3, 4]).set_requires_grad(true);
    let padded = a.pad_zero(&[(2, 1), (1, 3)]);

    assert!(padded.requires_grad());
    padded.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[3, 4]);
}

#[test]
fn test_slice_backward_shape() {
    // Test that SliceBackward produces correct gradient shape
    let a = Tensor::<f32, Dim2>::ones([4, 5]).set_requires_grad(true);
    let sliced = a.slice(&[(1, 3), (2, 5)]);

    assert!(sliced.requires_grad());
    assert_eq!(sliced.shape(), &[2, 3]);

    // Backward
    sliced.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[4, 5]); // Same as original input
}

#[test]
fn test_slice_backward_f64() {
    let a = Tensor::<f64, Dim2>::ones([6, 8]).set_requires_grad(true);
    let sliced = a.slice(&[(1, 4), (2, 6)]);

    assert!(sliced.requires_grad());
    sliced.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[6, 8]);
}

#[test]
fn test_concat_backward_shape() {
    // Test that ConcatBackward produces correct gradient shapes
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let b = Tensor::<f32, Dim2>::ones([4, 3]).set_requires_grad(true);
    let c = Tensor::concat(&[&a, &b], 0);

    assert!(c.requires_grad());
    assert_eq!(c.shape(), &[6, 3]);

    // Backward
    c.backward();

    let grad_a = a.grad().expect("a should have gradient");
    let grad_b = b.grad().expect("b should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]); // Same as original a
    assert_eq!(grad_b.shape(), &[4, 3]); // Same as original b
}

#[test]
fn test_concat_backward_axis1() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let b = Tensor::<f32, Dim2>::ones([2, 5]).set_requires_grad(true);
    let c = Tensor::concat(&[&a, &b], 1);

    assert_eq!(c.shape(), &[2, 8]);
    c.backward();

    let grad_a = a.grad().expect("a should have gradient");
    let grad_b = b.grad().expect("b should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[2, 5]);
}

#[test]
fn test_concat_backward_multiple() {
    let a = Tensor::<f32, Dim2>::ones([1, 3]).set_requires_grad(true);
    let b = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let c = Tensor::<f32, Dim2>::ones([3, 3]).set_requires_grad(true);
    let d = Tensor::concat(&[&a, &b, &c], 0);

    assert_eq!(d.shape(), &[6, 3]);
    d.backward();

    let grad_a = a.grad().expect("a should have gradient");
    let grad_b = b.grad().expect("b should have gradient");
    let grad_c = c.grad().expect("c should have gradient");
    assert_eq!(grad_a.shape(), &[1, 3]);
    assert_eq!(grad_b.shape(), &[2, 3]);
    assert_eq!(grad_c.shape(), &[3, 3]);
}

#[test]
fn test_concat_backward_f64() {
    let a = Tensor::<f64, Dim2>::ones([2, 3]).set_requires_grad(true);
    let b = Tensor::<f64, Dim2>::ones([4, 3]).set_requires_grad(true);
    let c = Tensor::concat(&[&a, &b], 0);

    c.backward();

    let grad_a = a.grad().expect("a should have gradient");
    let grad_b = b.grad().expect("b should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]);
    assert_eq!(grad_b.shape(), &[4, 3]);
}

#[test]
fn test_pad_slice_inverse_backward() {
    // pad and slice are inverses for gradients
    // If we pad then slice back, gradient should flow correctly
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let padded = a.pad_zero(&[(1, 1), (2, 2)]);
    let sliced = padded.slice(&[(1, 3), (2, 5)]); // Slice back to original shape

    assert_eq!(sliced.shape(), &[2, 3]);
    sliced.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]);
}

#[test]
fn test_no_grad_propagation() {
    // When input doesn't require grad, output shouldn't either
    let a = Tensor::<f32, Dim2>::ones([2, 3]); // No requires_grad
    let padded = a.pad_zero(&[(1, 1), (0, 0)]);
    assert!(!padded.requires_grad());

    let b = Tensor::<f32, Dim2>::ones([4, 5]);
    let sliced = b.slice(&[(1, 3), (2, 5)]);
    assert!(!sliced.requires_grad());

    let c = Tensor::<f32, Dim2>::ones([2, 3]);
    let d = Tensor::<f32, Dim2>::ones([4, 3]);
    let concated = Tensor::concat(&[&c, &d], 0);
    assert!(!concated.requires_grad());
}

// ========================================================================
// Gradient tests for view operations (squeeze, unsqueeze, reshape, expand, etc.)
// ========================================================================

#[test]
fn test_squeeze_backward_shape() {
    let a = Tensor::<f32, Dim3>::ones([2, 1, 3]).set_requires_grad(true);
    let squeezed = a.squeeze(1);

    assert!(squeezed.requires_grad());
    assert_eq!(squeezed.shape(), &[2, 3]);

    squeezed.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 1, 3]); // Same as original input
}

#[test]
fn test_unsqueeze_backward_shape() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let unsqueezed = a.unsqueeze(1);

    assert!(unsqueezed.requires_grad());
    assert_eq!(unsqueezed.shape(), &[2, 1, 3]);

    unsqueezed.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
}

#[test]
fn test_reshape_backward_shape() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let reshaped = a.reshape([3, 2]);

    assert!(reshaped.requires_grad());
    assert_eq!(reshaped.shape(), &[3, 2]);

    reshaped.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
}

#[test]
fn test_reshape_dyn_backward_shape() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let reshaped = a.reshape_dyn(&[6]);

    assert!(reshaped.requires_grad());
    assert_eq!(reshaped.shape(), &[6]);

    reshaped.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
}

#[test]
fn test_expand_backward_shape() {
    let a = Tensor::<f32, Dim2>::ones([1, 3]).set_requires_grad(true);
    let expanded = a.expand(&[4, 3]);

    assert!(expanded.requires_grad());
    assert_eq!(expanded.shape(), &[4, 3]);

    expanded.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[1, 3]); // Same as original input
}

#[test]
fn test_permute_backward_shape() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let permuted = a.permute(&[1, 0]);

    assert!(permuted.requires_grad());
    assert_eq!(permuted.shape(), &[3, 2]);

    permuted.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
}

#[test]
fn test_transpose_backward_shape() {
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let transposed = a.transpose();

    assert!(transposed.requires_grad());
    assert_eq!(transposed.shape(), &[3, 2]);

    transposed.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]); // Same as original input
}

#[test]
fn test_chained_view_ops_backward() {
    // Test that gradients flow through chained view operations
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let b = a.unsqueeze(0); // [1, 2, 3]
    let c = b.expand(&[4, 2, 3]); // [4, 2, 3]
    let d = c.reshape_dyn(&[8, 3]); // [8, 3]

    assert!(d.requires_grad());
    d.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]);
}

#[test]
fn test_squeeze_unsqueeze_inverse_backward() {
    // squeeze and unsqueeze are inverses
    let a = Tensor::<f32, Dim2>::ones([2, 3]).set_requires_grad(true);
    let unsqueezed = a.unsqueeze(1); // [2, 1, 3]
    let squeezed = unsqueezed.squeeze(1); // [2, 3]

    assert_eq!(squeezed.shape(), &[2, 3]);
    squeezed.backward();

    let grad_a = a.grad().expect("a should have gradient");
    assert_eq!(grad_a.shape(), &[2, 3]);
}

#[test]
fn test_view_ops_no_grad_propagation() {
    // When input doesn't require grad, output shouldn't either
    let a = Tensor::<f32, Dim2>::ones([2, 3]); // No requires_grad

    let squeezed = a.unsqueeze(0).squeeze(0);
    assert!(!squeezed.requires_grad());

    let reshaped = a.reshape([6]);
    assert!(!reshaped.requires_grad());

    let expanded = Tensor::<f32, Dim2>::ones([1, 3]).expand(&[4, 3]);
    assert!(!expanded.requires_grad());

    let permuted = a.permute(&[1, 0]);
    assert!(!permuted.requires_grad());

    let transposed = a.transpose();
    assert!(!transposed.requires_grad());
}

// unfold tests

#[test]
fn test_unfold1d_shape() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 10]);
    let unfolded = input.unfold1d(3, 1);
    assert_eq!(unfolded.shape(), &[2, 3, 8, 3]);
}

#[test]
fn test_unfold1d_with_stride() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 10]);
    let unfolded = input.unfold1d(3, 2);
    assert_eq!(unfolded.shape(), &[2, 3, 4, 3]);
}

#[test]
fn test_unfold2d_shape() {
    let input = Tensor::<f32, Dim4>::ones([2, 3, 28, 28]);
    let unfolded = input.unfold2d((3, 3), (1, 1));
    assert_eq!(unfolded.shape(), &[2, 3, 26, 26, 3, 3]);
}

#[test]
fn test_unfold2d_with_stride() {
    let input = Tensor::<f32, Dim4>::ones([2, 3, 28, 28]);
    let unfolded = input.unfold2d((3, 3), (2, 2));
    assert_eq!(unfolded.shape(), &[2, 3, 13, 13, 3, 3]);
}

#[test]
fn test_unfold3d_shape() {
    let input = Tensor::<f32, Dim5>::ones([2, 3, 16, 16, 16]);
    let unfolded = input.unfold3d((3, 3, 3), (1, 1, 1));
    assert_eq!(unfolded.shape(), &[2, 3, 14, 14, 14, 3, 3, 3]);
}

// ============================================================================
// Fold tests
// ============================================================================

#[test]
fn test_fold1d_shape() {
    // [N, C, out_L, k] -> [N, C, L]
    let input = Tensor::<f32, Dim4>::ones([2, 3, 8, 3]);
    let folded = input.fold1d(10, 1); // L = (out_L - 1) * stride + k = 7*1 + 3 = 10
    assert_eq!(folded.shape(), &[2, 3, 10]);
}

#[test]
fn test_fold1d_non_overlapping() {
    // stride == k: simple reshape
    let input = Tensor::<f32, Dim4>::ones([2, 3, 4, 3]);
    let folded = input.fold1d(12, 3); // L = (4-1)*3 + 3 = 12
    assert_eq!(folded.shape(), &[2, 3, 12]);
}

#[test]
fn test_fold2d_shape() {
    // [N, C, out_H, out_W, kH, kW] -> [N, C, H, W]
    let input = Tensor::<f32, Dim6>::ones([2, 3, 6, 6, 3, 3]);
    let folded = input.fold2d((8, 8), (1, 1)); // H,W = (6-1)*1 + 3 = 8
    assert_eq!(folded.shape(), &[2, 3, 8, 8]);
}

#[test]
fn test_fold2d_non_overlapping() {
    // stride == kernel: simple permute + reshape
    let input = Tensor::<f32, Dim6>::ones([2, 3, 4, 4, 2, 2]);
    let folded = input.fold2d((8, 8), (2, 2)); // H,W = (4-1)*2 + 2 = 8
    assert_eq!(folded.shape(), &[2, 3, 8, 8]);
}

#[test]
fn test_fold3d_shape() {
    // [N, C, out_H, out_W, out_D, kH, kW, kD] -> [N, C, H, W, D]
    let input = Tensor::<f32, Dim8>::ones([2, 3, 4, 4, 4, 2, 2, 2]);
    let folded = input.fold3d((5, 5, 5), (1, 1, 1)); // H,W,D = (4-1)*1 + 2 = 5
    assert_eq!(folded.shape(), &[2, 3, 5, 5, 5]);
}

#[test]
fn test_fold3d_non_overlapping() {
    // stride == kernel: simple permute + reshape
    let input = Tensor::<f32, Dim8>::ones([2, 3, 2, 2, 2, 3, 3, 3]);
    let folded = input.fold3d((6, 6, 6), (3, 3, 3)); // H,W,D = (2-1)*3 + 3 = 6
    assert_eq!(folded.shape(), &[2, 3, 6, 6, 6]);
}

// ============================================================================
// Unfold gradient tests (unfold -> fold for backward)
// ============================================================================

#[test]
fn test_unfold1d_backward_shape() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 10]).set_requires_grad(true);
    let unfolded = input.unfold1d(3, 1);
    assert!(unfolded.requires_grad());
    assert_eq!(unfolded.shape(), &[2, 3, 8, 3]);

    // Backward
    unfolded.backward();

    let grad = input.grad().expect("input should have gradient");
    assert_eq!(grad.shape(), &[2, 3, 10]); // Same as original input
}

#[test]
fn test_unfold2d_backward_shape() {
    let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 4]).set_requires_grad(true);
    let unfolded = input.unfold2d((2, 2), (1, 1));
    assert!(unfolded.requires_grad());
    assert_eq!(unfolded.shape(), &[1, 1, 3, 3, 2, 2]);

    // Backward
    unfolded.backward();

    let grad = input.grad().expect("input should have gradient");
    assert_eq!(grad.shape(), &[1, 1, 4, 4]); // Same as original input
}

#[test]
fn test_unfold2d_backward_non_overlapping() {
    // stride == kernel_size
    let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 4]).set_requires_grad(true);
    let unfolded = input.unfold2d((2, 2), (2, 2));
    assert!(unfolded.requires_grad());
    assert_eq!(unfolded.shape(), &[1, 1, 2, 2, 2, 2]);

    // Backward
    unfolded.backward();

    let grad = input.grad().expect("input should have gradient");
    assert_eq!(grad.shape(), &[1, 1, 4, 4]);
}

#[test]
fn test_unfold_fold_round_trip() {
    // unfold -> fold should give back the same shape (but different values due to accumulation)
    let input = Tensor::<f32, Dim4>::ones([2, 3, 4, 4]).set_requires_grad(true);
    let unfolded = input.unfold2d((2, 2), (1, 1)); // [2, 3, 3, 3, 2, 2]
    let folded = unfolded.fold2d((4, 4), (1, 1)); // Back to [2, 3, 4, 4]
    assert_eq!(folded.shape(), &[2, 3, 4, 4]);
}

#[test]
fn test_unfold_no_grad_propagation() {
    // When input doesn't require grad, output shouldn't either
    let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 4]); // No requires_grad
    let unfolded = input.unfold2d((2, 2), (1, 1));
    assert!(!unfolded.requires_grad());
}

// ============================================================================
// Dilated Unfold/Fold tests
// ============================================================================

#[test]
fn test_unfold1d_dilated_shape() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 10]);
    // dilation=2: effective kernel size = (3-1)*2+1 = 5
    // output length = (10-5)/1+1 = 6
    let unfolded = input.unfold1d_dilated(3, 1, 2);
    assert_eq!(unfolded.shape(), &[2, 3, 6, 3]);
}

#[test]
fn test_unfold1d_dilated_with_stride() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 14]);
    // dilation=2: effective kernel size = (3-1)*2+1 = 5
    // output length = (14-5)/2+1 = 5
    let unfolded = input.unfold1d_dilated(3, 2, 2);
    assert_eq!(unfolded.shape(), &[2, 3, 5, 3]);
}

#[test]
fn test_unfold2d_dilated_shape() {
    let input = Tensor::<f32, Dim4>::ones([2, 3, 10, 10]);
    // dilation=(2,2): effective kernel size = (3-1)*2+1 = 5
    // output size = (10-5)/1+1 = 6
    let unfolded = input.unfold2d_dilated((3, 3), (1, 1), (2, 2));
    assert_eq!(unfolded.shape(), &[2, 3, 6, 6, 3, 3]);
}

#[test]
fn test_unfold2d_dilated_with_stride() {
    let input = Tensor::<f32, Dim4>::ones([2, 3, 15, 15]);
    // dilation=(2,2): effective kernel size = (3-1)*2+1 = 5
    // output size = (15-5)/2+1 = 6
    let unfolded = input.unfold2d_dilated((3, 3), (2, 2), (2, 2));
    assert_eq!(unfolded.shape(), &[2, 3, 6, 6, 3, 3]);
}

#[test]
fn test_unfold3d_dilated_shape() {
    let input = Tensor::<f32, Dim5>::ones([2, 3, 10, 10, 10]);
    // dilation=(2,2,2): effective kernel size = (3-1)*2+1 = 5
    // output size = (10-5)/1+1 = 6
    let unfolded = input.unfold3d_dilated((3, 3, 3), (1, 1, 1), (2, 2, 2));
    assert_eq!(unfolded.shape(), &[2, 3, 6, 6, 6, 3, 3, 3]);
}

#[test]
fn test_fold1d_dilated_shape() {
    // [N, C, out_L, k] -> [N, C, L]
    // effective kernel size = (3-1)*2+1 = 5
    // L = (out_L - 1) * stride + effective_k = (6-1)*1 + 5 = 10
    let input = Tensor::<f32, Dim4>::ones([2, 3, 6, 3]);
    let folded = input.fold1d_dilated(10, 1, 2);
    assert_eq!(folded.shape(), &[2, 3, 10]);
}

#[test]
fn test_fold2d_dilated_shape() {
    // [N, C, out_H, out_W, kH, kW] -> [N, C, H, W]
    // effective kernel size = (3-1)*2+1 = 5
    // H,W = (out_H - 1) * stride + effective_k = (6-1)*1 + 5 = 10
    let input = Tensor::<f32, Dim6>::ones([2, 3, 6, 6, 3, 3]);
    let folded = input.fold2d_dilated((10, 10), (1, 1), (2, 2));
    assert_eq!(folded.shape(), &[2, 3, 10, 10]);
}

#[test]
fn test_fold3d_dilated_shape() {
    // [N, C, out_H, out_W, out_D, kH, kW, kD] -> [N, C, H, W, D]
    // effective kernel size = (3-1)*2+1 = 5
    // H,W,D = (out - 1) * stride + effective_k = (6-1)*1 + 5 = 10
    let input = Tensor::<f32, Dim8>::ones([2, 3, 6, 6, 6, 3, 3, 3]);
    let folded = input.fold3d_dilated((10, 10, 10), (1, 1, 1), (2, 2, 2));
    assert_eq!(folded.shape(), &[2, 3, 10, 10, 10]);
}

#[test]
fn test_unfold1d_dilated_backward_shape() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 10]).set_requires_grad(true);
    let unfolded = input.unfold1d_dilated(3, 1, 2);
    assert!(unfolded.requires_grad());
    assert_eq!(unfolded.shape(), &[2, 3, 6, 3]);

    unfolded.backward();

    let grad = input.grad().expect("input should have gradient");
    assert_eq!(grad.shape(), &[2, 3, 10]); // Same as original input
}

#[test]
fn test_unfold2d_dilated_backward_shape() {
    let input = Tensor::<f32, Dim4>::ones([1, 1, 8, 8]).set_requires_grad(true);
    let unfolded = input.unfold2d_dilated((3, 3), (1, 1), (2, 2));
    assert!(unfolded.requires_grad());
    // effective kernel size = (3-1)*2+1 = 5
    // output size = (8-5)/1+1 = 4
    assert_eq!(unfolded.shape(), &[1, 1, 4, 4, 3, 3]);

    unfolded.backward();

    let grad = input.grad().expect("input should have gradient");
    assert_eq!(grad.shape(), &[1, 1, 8, 8]); // Same as original input
}

#[test]
fn test_unfold3d_dilated_backward_shape() {
    let input = Tensor::<f32, Dim5>::ones([1, 1, 8, 8, 8]).set_requires_grad(true);
    let unfolded = input.unfold3d_dilated((2, 2, 2), (1, 1, 1), (2, 2, 2));
    assert!(unfolded.requires_grad());
    // effective kernel size = (2-1)*2+1 = 3
    // output size = (8-3)/1+1 = 6
    assert_eq!(unfolded.shape(), &[1, 1, 6, 6, 6, 2, 2, 2]);

    unfolded.backward();

    let grad = input.grad().expect("input should have gradient");
    assert_eq!(grad.shape(), &[1, 1, 8, 8, 8]); // Same as original input
}

#[test]
fn test_unfold_fold_dilated_round_trip() {
    // unfold -> fold with dilation should give back the same shape
    let input = Tensor::<f32, Dim4>::ones([2, 3, 10, 10]).set_requires_grad(true);
    // effective kernel size = (3-1)*2+1 = 5, output = (10-5)/1+1 = 6
    let unfolded = input.unfold2d_dilated((3, 3), (1, 1), (2, 2)); // [2, 3, 6, 6, 3, 3]
    let folded = unfolded.fold2d_dilated((10, 10), (1, 1), (2, 2)); // Back to [2, 3, 10, 10]
    assert_eq!(folded.shape(), &[2, 3, 10, 10]);
}

// ============================================================================
// Interleave tests
// ============================================================================

#[test]
fn test_interleave_1d_stride2() {
    use crate::tensor::Dim1;
    // [3] with stride=2 on axis 0 -> [5] (1 + (3-1) * 2 = 5)
    let a = Tensor::<f32, Dim1>::ones([3]);
    let b = a.interleave(&[0], &[2]);
    assert_eq!(b.shape(), &[5]);
}

#[test]
fn test_interleave_2d_stride2() {
    // [2, 3, 4, 5] with stride=(2, 3) on axes (2, 3)
    // new_h = 1 + (4-1) * 2 = 7
    // new_w = 1 + (5-1) * 3 = 13
    let a = Tensor::<f32, Dim4>::ones([2, 3, 4, 5]);
    let b = a.interleave(&[2, 3], &[2, 3]);
    assert_eq!(b.shape(), &[2, 3, 7, 13]);
}

#[test]
fn test_interleave_no_change_stride1() {
    let a = Tensor::<f32, Dim2>::ones([3, 4]);
    let b = a.interleave(&[0, 1], &[1, 1]);
    assert_eq!(b.shape(), &[3, 4]);
}

#[test]
fn test_interleave_single_element() {
    use crate::tensor::Dim1;
    // Single element: [1] with stride=2 -> [1] (1 + (1-1) * 2 = 1)
    let a = Tensor::<f32, Dim1>::ones([1]);
    let b = a.interleave(&[0], &[2]);
    assert_eq!(b.shape(), &[1]);
}

// ============================================================================
// Fold with stride > 1 tests
// ============================================================================

#[test]
fn test_fold1d_stride2() {
    // [N, C, out_L, k] -> [N, C, L]
    // k=2, out_L=3, stride=2
    // L = (out_L - 1) * stride + k = (3-1)*2 + 2 = 6
    let input = Tensor::<f32, Dim4>::ones([1, 1, 3, 2]);
    let folded = input.fold1d(6, 2);
    assert_eq!(folded.shape(), &[1, 1, 6]);
}

#[test]
fn test_fold2d_stride2() {
    // [N, C, out_H, out_W, kH, kW] -> [N, C, H, W]
    // k=(2,2), out=(3,3), stride=(2,2)
    // H,W = (3-1)*2 + 2 = 6
    let input = Tensor::<f32, Dim6>::ones([1, 1, 3, 3, 2, 2]);
    let folded = input.fold2d((6, 6), (2, 2));
    assert_eq!(folded.shape(), &[1, 1, 6, 6]);
}

#[test]
fn test_fold3d_stride2() {
    // [N, C, out_H, out_W, out_D, kH, kW, kD] -> [N, C, H, W, D]
    // k=(2,2,2), out=(2,2,2), stride=(2,2,2)
    // H,W,D = (2-1)*2 + 2 = 4
    let input = Tensor::<f32, Dim8>::ones([1, 1, 2, 2, 2, 2, 2, 2]);
    let folded = input.fold3d((4, 4, 4), (2, 2, 2));
    assert_eq!(folded.shape(), &[1, 1, 4, 4, 4]);
}

#[test]
fn test_fold2d_stride2_dilation2() {
    // k=(2,2), dilation=(2,2) -> effective_k = 3
    // out=(2,2), stride=(2,2)
    // H,W = (2-1)*2 + 3 = 5
    let input = Tensor::<f32, Dim6>::ones([1, 1, 2, 2, 2, 2]);
    let folded = input.fold2d_dilated((5, 5), (2, 2), (2, 2));
    assert_eq!(folded.shape(), &[1, 1, 5, 5]);
}

#[test]
fn test_unfold_fold_stride2_round_trip() {
    // unfold -> fold with stride > 1 should give back the same shape
    let input = Tensor::<f32, Dim4>::ones([1, 1, 8, 8]).set_requires_grad(true);
    // k=2, stride=2 -> out = (8-2)/2+1 = 4
    let unfolded = input.unfold2d((2, 2), (2, 2)); // [1, 1, 4, 4, 2, 2]
    assert_eq!(unfolded.shape(), &[1, 1, 4, 4, 2, 2]);
    // fold back: (4-1)*2 + 2 = 8
    let folded = unfolded.fold2d((8, 8), (2, 2)); // [1, 1, 8, 8]
    assert_eq!(folded.shape(), &[1, 1, 8, 8]);
}

#[test]
fn test_unfold_fold_stride2_backward() {
    // Test gradient flow with stride > 1
    let input = Tensor::<f32, Dim4>::ones([1, 1, 6, 6]).set_requires_grad(true);
    // k=2, stride=2 -> out = (6-2)/2+1 = 3
    let unfolded = input.unfold2d((2, 2), (2, 2)); // [1, 1, 3, 3, 2, 2]
    assert!(unfolded.requires_grad());

    unfolded.backward();

    let grad = input.grad().expect("input should have gradient");
    assert_eq!(grad.shape(), &[1, 1, 6, 6]);
}

// ============================================================================
// Gather tests
// ============================================================================

#[test]
fn test_gather_1d_shape() {
    use crate::tensor::Dim1;
    let input = Tensor::<f32, Dim1>::ones([4]);
    let index = Tensor::<i64, Dim1>::zeros([3]);
    let output = input.gather(0, &index);
    assert_eq!(output.shape(), &[3]);
}

#[test]
fn test_gather_2d_dim0_shape() {
    let input = Tensor::<f32, Dim2>::ones([4, 5]);
    let index = Tensor::<i64, Dim2>::zeros([3, 5]);
    let output = input.gather(0, &index);
    assert_eq!(output.shape(), &[3, 5]);
}

#[test]
fn test_gather_2d_dim1_shape() {
    let input = Tensor::<f32, Dim2>::ones([4, 5]);
    let index = Tensor::<i64, Dim2>::zeros([4, 3]);
    let output = input.gather(1, &index);
    assert_eq!(output.shape(), &[4, 3]);
}

#[test]
fn test_gather_3d_dim0_shape() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 4]);
    let index = Tensor::<i64, Dim3>::zeros([5, 3, 4]);
    let output = input.gather(0, &index);
    assert_eq!(output.shape(), &[5, 3, 4]);
}

#[test]
fn test_gather_3d_dim1_shape() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 4]);
    let index = Tensor::<i64, Dim3>::zeros([2, 5, 4]);
    let output = input.gather(1, &index);
    assert_eq!(output.shape(), &[2, 5, 4]);
}

#[test]
fn test_gather_3d_dim2_shape() {
    let input = Tensor::<f32, Dim3>::ones([2, 3, 4]);
    let index = Tensor::<i64, Dim3>::zeros([2, 3, 5]);
    let output = input.gather(2, &index);
    assert_eq!(output.shape(), &[2, 3, 5]);
}

#[test]
fn test_gather_type_safe() {
    // Gather preserves dimension type
    let input = Tensor::<f32, Dim2>::ones([4, 5]);
    let index = Tensor::<i64, Dim2>::zeros([3, 5]);
    let output: Tensor<f32, Dim2> = input.gather(0, &index);
    assert_eq!(output.shape(), &[3, 5]);
}

#[test]
fn test_gather_f64() {
    let input = Tensor::<f64, Dim2>::ones([4, 5]);
    let index = Tensor::<i64, Dim2>::zeros([3, 5]);
    let output = input.gather(0, &index);
    assert_eq!(output.shape(), &[3, 5]);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_gather_invalid_dim() {
    let input = Tensor::<f32, Dim2>::ones([4, 5]);
    let index = Tensor::<i64, Dim2>::zeros([3, 5]);
    let _ = input.gather(2, &index); // dim 2 out of bounds for 2D tensor
}

#[test]
#[should_panic(expected = "must match")]
fn test_gather_shape_mismatch() {
    let input = Tensor::<f32, Dim2>::ones([4, 5]);
    let index = Tensor::<i64, Dim2>::zeros([3, 6]); // dim 1 mismatch: 5 vs 6
    let _ = input.gather(0, &index);
}

#[test]
fn test_gather_no_grad_propagation() {
    // Gather does not support gradient tracking
    let input = Tensor::<f32, Dim2>::ones([4, 5]); // No requires_grad
    let index = Tensor::<i64, Dim2>::zeros([3, 5]);
    let output = input.gather(0, &index);
    assert!(!output.requires_grad());
}

// ============================================================================
// Scatter-Add Tests
// ============================================================================

#[test]
fn test_scatter_add_1d_shape() {
    let target = Tensor::<f32, DimDyn>::zeros_dyn(&[4]);
    let index = Tensor::<i64, DimDyn>::zeros_dyn(&[3]);
    let src = Tensor::<f32, DimDyn>::ones_dyn(&[3]);

    let result = target.scatter_add(0, &index, &src);
    assert_eq!(result.shape(), &[4]);
}

#[test]
fn test_scatter_add_2d_dim0_shape() {
    let target = Tensor::<f32, DimDyn>::zeros_dyn(&[4, 3]);
    let index = Tensor::<i64, DimDyn>::zeros_dyn(&[2, 3]);
    let src = Tensor::<f32, DimDyn>::ones_dyn(&[2, 3]);

    let result = target.scatter_add(0, &index, &src);
    assert_eq!(result.shape(), &[4, 3]);
}

#[test]
fn test_scatter_add_2d_dim1_shape() {
    let target = Tensor::<f32, DimDyn>::zeros_dyn(&[3, 4]);
    let index = Tensor::<i64, DimDyn>::zeros_dyn(&[3, 2]);
    let src = Tensor::<f32, DimDyn>::ones_dyn(&[3, 2]);

    let result = target.scatter_add(1, &index, &src);
    assert_eq!(result.shape(), &[3, 4]);
}

// ============================================================================
// Gather with Gradient Tests
// ============================================================================

#[test]
fn test_gather_autograd_shape() {
    use crate::tensor::Dim1;

    let input = Tensor::<f32, Dim1>::ones([4]).set_requires_grad(true);
    let index = Tensor::<i64, Dim1>::zeros([4]);
    let output = input.gather(0, &index);

    assert_eq!(output.shape(), &[4]);
    assert!(output.requires_grad());
}

#[test]
fn test_gather_autograd_2d() {
    let input = Tensor::<f32, Dim2>::ones([4, 5]).set_requires_grad(true);
    let index = Tensor::<i64, Dim2>::zeros([3, 5]);
    let output = input.gather(0, &index);

    assert_eq!(output.shape(), &[3, 5]);
    assert!(output.requires_grad());
}

#[test]
fn test_gather_no_requires_grad() {
    use crate::tensor::Dim1;

    let input = Tensor::<f32, Dim1>::ones([4]); // No requires_grad
    let index = Tensor::<i64, Dim1>::zeros([4]);
    let output = input.gather(0, &index);

    assert_eq!(output.shape(), &[4]);
    assert!(!output.requires_grad());
}

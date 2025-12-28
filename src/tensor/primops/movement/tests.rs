//! Tests for movement primitive operations

use crate::tensor::ops::PadValue;
use crate::tensor::{Dim2, Dim3, Dim4, Dim5, DimDyn, Tensor};

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

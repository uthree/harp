//! Backward (gradient) operations for movement primitives

use std::marker::PhantomData;

use crate::tensor::{DimDyn, Dimension, FloatDType, GradFn, GradFnTyped, Tensor};

// ============================================================================
// Movement Gradients (type-safe over Dimension)
// ============================================================================

/// Gradient for Pad: y = pad(x, padding)
/// ∂L/∂x = slice(∂L/∂y, padding位置)
pub struct PadBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    padding: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> PadBackward<T, D> {
    pub fn new(input: Tensor<T, D>, padding: Vec<(usize, usize)>) -> Self {
        Self { input, padding }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T> for PadBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // padding から slice範囲を計算: (before, before + original_dim)
        let ranges: Vec<(usize, usize)> = self
            .padding
            .iter()
            .zip(self.input.shape())
            .map(|(&(before, _), &dim)| (before, before + dim))
            .collect();
        vec![grad_output.slice(&ranges)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone().into_dyn()]
    }

    fn name(&self) -> &'static str {
        "PadBackward"
    }
}

/// Gradient for Slice: y = slice(x, ranges)
/// ∂L/∂x = pad_zero(∂L/∂y, 適切なpadding)
pub struct SliceBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    ranges: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> SliceBackward<T, D> {
    pub fn new(input: Tensor<T, D>, ranges: Vec<(usize, usize)>) -> Self {
        Self { input, ranges }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T> for SliceBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // ranges から padding を計算: (start, original_dim - end)
        let padding: Vec<(usize, usize)> = self
            .ranges
            .iter()
            .zip(self.input.shape())
            .map(|(&(start, end), &dim)| (start, dim - end))
            .collect();
        vec![grad_output.pad_zero(&padding)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone().into_dyn()]
    }

    fn name(&self) -> &'static str {
        "SliceBackward"
    }
}

/// Gradient for Concat: y = concat([a, b, ...], axis)
/// ∂L/∂a, ∂L/∂b, ... = split(∂L/∂y, axis, sizes)
pub struct ConcatBackward<T: FloatDType, D: Dimension> {
    inputs: Vec<Tensor<T, D>>,
    axis: usize,
}

impl<T: FloatDType, D: Dimension> ConcatBackward<T, D> {
    pub fn new(inputs: Vec<Tensor<T, D>>, axis: usize) -> Self {
        Self { inputs, axis }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T> for ConcatBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        let mut grads = Vec::new();
        let mut offset = 0;
        for input in &self.inputs {
            let size = input.shape()[self.axis];
            let ranges: Vec<(usize, usize)> = grad_output
                .shape()
                .iter()
                .enumerate()
                .map(|(i, &dim)| {
                    if i == self.axis {
                        (offset, offset + size)
                    } else {
                        (0, dim)
                    }
                })
                .collect();
            grads.push(grad_output.slice(&ranges));
            offset += size;
        }
        grads
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        self.inputs.iter().map(|t| t.clone().into_dyn()).collect()
    }

    fn name(&self) -> &'static str {
        "ConcatBackward"
    }
}

// ============================================================================
// View Operation Gradients
// ============================================================================

/// Gradient for Squeeze: y = squeeze(x, dim)
/// ∂L/∂x = unsqueeze(∂L/∂y, dim)
pub struct SqueezeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    dim: usize,
}

impl<T: FloatDType> SqueezeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, dim: usize) -> Self {
        Self { input, dim }
    }
}

impl<T: FloatDType> GradFn<T> for SqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // unsqueeze the gradient back at the squeezed dimension
        vec![grad_output.unsqueeze(self.dim)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "SqueezeBackward"
    }
}

/// Gradient for Unsqueeze: y = unsqueeze(x, dim)
/// ∂L/∂x = squeeze(∂L/∂y, dim)
pub struct UnsqueezeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    dim: usize,
}

impl<T: FloatDType> UnsqueezeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, dim: usize) -> Self {
        Self { input, dim }
    }
}

impl<T: FloatDType> GradFn<T> for UnsqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // squeeze the gradient back at the unsqueezed dimension
        vec![grad_output.squeeze(self.dim)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }
}

/// Gradient for Reshape: y = reshape(x, new_shape)
/// ∂L/∂x = reshape(∂L/∂y, original_shape)
pub struct ReshapeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType> ReshapeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

impl<T: FloatDType> GradFn<T> for ReshapeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // reshape gradient back to original shape
        vec![grad_output.reshape_dyn(&self.original_shape)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

/// Gradient for Expand: y = expand(x, new_shape)
/// ∂L/∂x = sum(∂L/∂y) over broadcast dimensions
pub struct ExpandBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType> ExpandBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

impl<T: FloatDType> GradFn<T> for ExpandBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Sum gradients over dimensions that were broadcast
        // A dimension was broadcast if original_shape[i] == 1 and grad_output.shape()[i] > 1
        let grad_shape = grad_output.shape();
        let mut result = grad_output.clone();

        // Find axes to reduce (where original was 1 but grad is larger)
        // Process in reverse order to preserve indices
        for i in (0..self.original_shape.len()).rev() {
            if self.original_shape[i] == 1 && grad_shape[i] > 1 {
                // Sum over this dimension (keepdim=true by using sum then unsqueeze)
                result = result.sum(i).unsqueeze(i);
            }
        }

        vec![result]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}

/// Gradient for Permute: y = permute(x, axes)
/// ∂L/∂x = permute(∂L/∂y, inverse_axes)
pub struct PermuteBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    axes: Vec<usize>,
}

impl<T: FloatDType> PermuteBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>, axes: Vec<usize>) -> Self {
        Self { input, axes }
    }
}

impl<T: FloatDType> GradFn<T> for PermuteBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Compute inverse permutation
        let mut inverse_axes = vec![0; self.axes.len()];
        for (i, &axis) in self.axes.iter().enumerate() {
            inverse_axes[axis] = i;
        }
        vec![grad_output.permute(&inverse_axes)]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "PermuteBackward"
    }
}

/// Gradient for Transpose: y = transpose(x)
/// ∂L/∂x = transpose(∂L/∂y)
pub struct TransposeBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
}

impl<T: FloatDType> TransposeBackward<T> {
    pub fn new(input: Tensor<T, DimDyn>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType> GradFn<T> for TransposeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // Transpose is its own inverse
        vec![grad_output.transpose()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

// ============================================================================
// Unfold Gradients (uses fold as inverse)
// ============================================================================

/// Gradient for Unfold1d: y = unfold1d(x, size, stride, dilation)
/// ∂L/∂x = fold1d(∂L/∂y, output_size, stride, dilation)
pub struct Unfold1dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output_size: usize,
    stride: usize,
    dilation: usize,
}

impl<T: FloatDType> Unfold1dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        output_size: usize,
        _kernel_size: usize,
        stride: usize,
        dilation: usize,
    ) -> Self {
        Self {
            input,
            output_size,
            stride,
            dilation,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Unfold1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // fold1d_dilated is the inverse of unfold1d_dilated
        let grad_4d = grad_output.into_dim4();
        let folded = grad_4d.fold1d_dilated(self.output_size, self.stride, self.dilation);
        vec![folded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Unfold1dBackward"
    }
}

/// Gradient for Unfold2d: y = unfold2d(x, sizes, strides, dilations)
/// ∂L/∂x = fold2d(∂L/∂y, output_size, strides, dilations)
pub struct Unfold2dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output_size: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl<T: FloatDType> Unfold2dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        output_size: (usize, usize),
        _kernel_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Self {
        Self {
            input,
            output_size,
            strides,
            dilations,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Unfold2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // fold2d_dilated is the inverse of unfold2d_dilated
        let grad_6d = grad_output.into_dim6();
        let folded = grad_6d.fold2d_dilated(self.output_size, self.strides, self.dilations);
        vec![folded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Unfold2dBackward"
    }
}

/// Gradient for Unfold3d: y = unfold3d(x, sizes, strides, dilations)
/// ∂L/∂x = fold3d(∂L/∂y, output_size, strides, dilations)
pub struct Unfold3dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    dilations: (usize, usize, usize),
}

impl<T: FloatDType> Unfold3dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        output_size: (usize, usize, usize),
        _kernel_size: (usize, usize, usize),
        strides: (usize, usize, usize),
        dilations: (usize, usize, usize),
    ) -> Self {
        Self {
            input,
            output_size,
            strides,
            dilations,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Unfold3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // fold3d_dilated is the inverse of unfold3d_dilated
        let grad_8d = grad_output.into_dim8();
        let folded = grad_8d.fold3d_dilated(self.output_size, self.strides, self.dilations);
        vec![folded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Unfold3dBackward"
    }
}

// ============================================================================
// Fold Gradients (uses unfold as inverse)
// ============================================================================

/// Gradient for Fold1d: y = fold1d(x, output_size, stride, dilation)
/// ∂L/∂x = unfold1d(∂L/∂y, kernel_size, stride, dilation)
pub struct Fold1dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
}

impl<T: FloatDType> Fold1dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
    ) -> Self {
        Self {
            input,
            kernel_size,
            stride,
            dilation,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Fold1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // unfold1d_dilated is the inverse of fold1d_dilated
        let grad_3d = grad_output.into_dim3();
        let unfolded = grad_3d.unfold1d_dilated(self.kernel_size, self.stride, self.dilation);
        vec![unfolded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Fold1dBackward"
    }
}

/// Gradient for Fold2d: y = fold2d(x, output_size, strides, dilations)
/// ∂L/∂x = unfold2d(∂L/∂y, kernel_size, strides, dilations)
pub struct Fold2dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl<T: FloatDType> Fold2dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        kernel_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Self {
        Self {
            input,
            kernel_size,
            strides,
            dilations,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Fold2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // unfold2d_dilated is the inverse of fold2d_dilated
        let grad_4d = grad_output.into_dim4();
        let unfolded = grad_4d.unfold2d_dilated(self.kernel_size, self.strides, self.dilations);
        vec![unfolded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Fold2dBackward"
    }
}

/// Gradient for Fold3d: y = fold3d(x, output_size, strides, dilations)
/// ∂L/∂x = unfold3d(∂L/∂y, kernel_size, strides, dilations)
pub struct Fold3dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    kernel_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    dilations: (usize, usize, usize),
}

impl<T: FloatDType> Fold3dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        kernel_size: (usize, usize, usize),
        strides: (usize, usize, usize),
        dilations: (usize, usize, usize),
    ) -> Self {
        Self {
            input,
            kernel_size,
            strides,
            dilations,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Fold3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // unfold3d_dilated is the inverse of fold3d_dilated
        let grad_5d = grad_output.into_dim5();
        let unfolded = grad_5d.unfold3d_dilated(self.kernel_size, self.strides, self.dilations);
        vec![unfolded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Fold3dBackward"
    }
}

// ============================================================================
// Typed Backward Structs (new system with static dimension typing)
// ============================================================================

/// Typed gradient for Squeeze: y = squeeze(x, dim)
/// Input is D, output is D::Smaller
pub struct SqueezeBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    dim: usize,
}

impl<T: FloatDType, D: Dimension> SqueezeBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, dim: usize) -> Self {
        Self { input, dim }
    }
}

// Output is D::Smaller, so we implement GradFnTyped for that dimension
impl<T: FloatDType, D: Dimension> GradFnTyped<T, D::Smaller> for SqueezeBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Smaller>) {
        if self.input.requires_grad_typed() {
            // Unsqueeze grad from D::Smaller to D
            // Use DimDyn intermediate for type flexibility
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_unsqueezed = grad_dyn.unsqueeze(self.dim);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_unsqueezed.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "SqueezeBackwardTyped"
    }
}

/// Typed gradient for Unsqueeze: y = unsqueeze(x, dim)
/// Input is D, output is D::Larger
pub struct UnsqueezeBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    dim: usize,
}

impl<T: FloatDType, D: Dimension> UnsqueezeBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, dim: usize) -> Self {
        Self { input, dim }
    }
}

// Output is D::Larger, so we implement GradFnTyped for that dimension
impl<T: FloatDType, D: Dimension> GradFnTyped<T, D::Larger> for UnsqueezeBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Larger>) {
        if self.input.requires_grad_typed() {
            // Squeeze grad from D::Larger to D
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_squeezed = grad_dyn.squeeze(self.dim);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_squeezed.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackwardTyped"
    }
}

/// Typed gradient for Pad: y = pad(x, padding)
/// Same dimension, D -> D
pub struct PadBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    padding: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> PadBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, padding: Vec<(usize, usize)>) -> Self {
        Self { input, padding }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for PadBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad_typed() {
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let ranges: Vec<(usize, usize)> = self
                .padding
                .iter()
                .zip(self.input.shape())
                .map(|(&(before, _), &dim)| (before, before + dim))
                .collect();
            let grad_sliced = grad_dyn.slice(&ranges);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_sliced.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "PadBackwardTyped"
    }
}

/// Typed gradient for Slice: y = slice(x, ranges)
/// Same dimension, D -> D
pub struct SliceBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    ranges: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> SliceBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, ranges: Vec<(usize, usize)>) -> Self {
        Self { input, ranges }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for SliceBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad_typed() {
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let padding: Vec<(usize, usize)> = self
                .ranges
                .iter()
                .zip(self.input.shape())
                .map(|(&(start, end), &dim)| (start, dim - end))
                .collect();
            let grad_padded = grad_dyn.pad_zero(&padding);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_padded.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "SliceBackwardTyped"
    }
}

/// Typed gradient for Reshape: y = reshape(x, new_shape)
/// Dimension can change, but we treat output as DIn, input as DOut
pub struct ReshapeBackwardTyped<T: FloatDType, DIn: Dimension, DOut: Dimension> {
    input: Tensor<T, DIn>,
    _output_dim: PhantomData<DOut>,
}

impl<T: FloatDType, DIn: Dimension, DOut: Dimension> ReshapeBackwardTyped<T, DIn, DOut> {
    pub fn new(input: Tensor<T, DIn>) -> Self {
        Self {
            input,
            _output_dim: PhantomData,
        }
    }
}

impl<T: FloatDType, DIn: Dimension, DOut: Dimension> GradFnTyped<T, DOut>
    for ReshapeBackwardTyped<T, DIn, DOut>
{
    fn backward(&self, grad_output: &Tensor<T, DOut>) {
        if self.input.requires_grad_typed() {
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_reshaped = grad_dyn.reshape_dyn(self.input.shape());
            let grad_input: Tensor<T, DIn> = Tensor {
                inner: grad_reshaped.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ReshapeBackwardTyped"
    }
}

/// Typed gradient for Permute: y = permute(x, axes)
/// Input D -> Output DimDyn
pub struct PermuteBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    axes: Vec<usize>,
}

impl<T: FloatDType, D: Dimension> PermuteBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, axes: Vec<usize>) -> Self {
        Self { input, axes }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, DimDyn> for PermuteBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad_typed() {
            // Compute inverse permutation
            let mut inverse = vec![0; self.axes.len()];
            for (i, &ax) in self.axes.iter().enumerate() {
                inverse[ax] = i;
            }
            let grad_permuted = grad_output.permute(&inverse);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_permuted.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "PermuteBackwardTyped"
    }
}

/// Typed gradient for Transpose: y = transpose(x)
/// Same dimension, D -> D (swaps last two axes)
pub struct TransposeBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

/// Typed gradient for Concat: z = concat([a, b, ...], axis)
/// ∂L/∂a = slice(∂L/∂z, a's range), etc.
pub struct ConcatBackwardTyped<T: FloatDType, D: Dimension> {
    inputs: Vec<Tensor<T, D>>,
    axis: usize,
}

impl<T: FloatDType, D: Dimension> ConcatBackwardTyped<T, D> {
    pub fn new(inputs: Vec<Tensor<T, D>>, axis: usize) -> Self {
        Self { inputs, axis }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, D> for ConcatBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        let mut offset = 0;
        for input in &self.inputs {
            if input.requires_grad_typed() {
                let size = input.shape()[self.axis];
                let ranges: Vec<(usize, usize)> = grad_output
                    .shape()
                    .iter()
                    .enumerate()
                    .map(|(i, &dim)| {
                        if i == self.axis {
                            (offset, offset + size)
                        } else {
                            (0, dim)
                        }
                    })
                    .collect();
                let grad_slice = grad_output.slice(&ranges);
                input.backward_with_typed(grad_slice);
            }
            offset += input.shape()[self.axis];
        }
    }

    fn name(&self) -> &'static str {
        "ConcatBackwardTyped"
    }
}

/// Typed gradient for Expand: y = expand(x, new_shape)
/// ∂L/∂x = sum(∂L/∂y) over broadcast dimensions
pub struct ExpandBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType, D: Dimension> ExpandBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

// Expand returns DimDyn, so we implement GradFnTyped for DimDyn
impl<T: FloatDType, D: Dimension> GradFnTyped<T, DimDyn> for ExpandBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad_typed() {
            // Sum gradients over dimensions that were broadcast
            // A dimension was broadcast if original_shape[i] == 1 and grad_output.shape()[i] > 1
            let grad_shape = grad_output.shape();
            let mut result = grad_output.clone();

            // Find axes to reduce (where original was 1 but grad is larger)
            // Process in reverse order to preserve indices
            for i in (0..self.original_shape.len()).rev() {
                if self.original_shape[i] == 1 && grad_shape[i] > 1 {
                    // Sum over this dimension (keepdim=true by using sum then unsqueeze)
                    result = result.sum(i).unsqueeze(i);
                }
            }

            let grad_input: Tensor<T, D> = Tensor {
                inner: result.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ExpandBackwardTyped"
    }
}

/// Typed gradient for ReshapeDyn: y = reshape_dyn(x, new_shape)
/// ∂L/∂x = reshape(∂L/∂y, original_shape)
pub struct ReshapeDynBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType, D: Dimension> ReshapeDynBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

// reshape_dyn returns DimDyn
impl<T: FloatDType, D: Dimension> GradFnTyped<T, DimDyn> for ReshapeDynBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad_typed() {
            let grad_reshaped = grad_output.reshape_dyn(&self.original_shape);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_reshaped.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ReshapeDynBackwardTyped"
    }
}

impl<T: FloatDType, D: Dimension> TransposeBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, DimDyn> for TransposeBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad_typed() {
            let grad_transposed = grad_output.transpose();
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_transposed.inner,
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "TransposeBackwardTyped"
    }
}

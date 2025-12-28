//! Backward (gradient) operations for movement primitives

use crate::tensor::{DimDyn, Dimension, FloatDType, GradFn, Tensor};

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

/// Gradient for Unfold1d: y = unfold1d(x, size, stride)
/// ∂L/∂x = fold1d(∂L/∂y, output_size, stride)
pub struct Unfold1dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output_size: usize,
    stride: usize,
}

impl<T: FloatDType> Unfold1dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        output_size: usize,
        _kernel_size: usize,
        stride: usize,
    ) -> Self {
        Self {
            input,
            output_size,
            stride,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Unfold1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // fold1d is the inverse of unfold1d
        let grad_4d = grad_output.into_dim4();
        let folded = grad_4d.fold1d(self.output_size, self.stride);
        vec![folded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Unfold1dBackward"
    }
}

/// Gradient for Unfold2d: y = unfold2d(x, sizes, strides)
/// ∂L/∂x = fold2d(∂L/∂y, output_size, strides)
pub struct Unfold2dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output_size: (usize, usize),
    strides: (usize, usize),
}

impl<T: FloatDType> Unfold2dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        output_size: (usize, usize),
        _kernel_size: (usize, usize),
        strides: (usize, usize),
    ) -> Self {
        Self {
            input,
            output_size,
            strides,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Unfold2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // fold2d is the inverse of unfold2d
        let grad_6d = grad_output.into_dim6();
        let folded = grad_6d.fold2d(self.output_size, self.strides);
        vec![folded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Unfold2dBackward"
    }
}

/// Gradient for Unfold3d: y = unfold3d(x, sizes, strides)
/// ∂L/∂x = fold3d(∂L/∂y, output_size, strides)
pub struct Unfold3dBackward<T: FloatDType> {
    input: Tensor<T, DimDyn>,
    output_size: (usize, usize, usize),
    strides: (usize, usize, usize),
}

impl<T: FloatDType> Unfold3dBackward<T> {
    pub fn new(
        input: Tensor<T, DimDyn>,
        output_size: (usize, usize, usize),
        _kernel_size: (usize, usize, usize),
        strides: (usize, usize, usize),
    ) -> Self {
        Self {
            input,
            output_size,
            strides,
        }
    }
}

impl<T: FloatDType> GradFn<T> for Unfold3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) -> Vec<Tensor<T, DimDyn>> {
        // fold3d is the inverse of unfold3d
        let grad_8d = grad_output.into_dim8();
        let folded = grad_8d.fold3d(self.output_size, self.strides);
        vec![folded.into_dyn()]
    }

    fn inputs(&self) -> Vec<Tensor<T, DimDyn>> {
        vec![self.input.clone()]
    }

    fn name(&self) -> &'static str {
        "Unfold3dBackward"
    }
}

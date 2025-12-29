//! Backward (gradient) operations for movement primitives

use std::marker::PhantomData;

use crate::tensor::{
    Dim3, Dim4, Dim5, Dim6, Dim8, DimDyn, Dimension, FloatDType, GradFnTyped, Tensor,
};

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

// ============================================================================
// Typed Unfold/Fold Backward Structs
// ============================================================================

/// Typed gradient for Unfold1d: y = unfold1d(x, size, stride, dilation)
/// Input: Dim3, Output: Dim4
/// ∂L/∂x = fold1d(∂L/∂y, output_size, stride, dilation)
pub struct Unfold1dBackwardTyped<T: FloatDType> {
    input: Tensor<T, Dim3>,
    output_size: usize,
    stride: usize,
    dilation: usize,
}

impl<T: FloatDType> Unfold1dBackwardTyped<T> {
    pub fn new(input: Tensor<T, Dim3>, output_size: usize, stride: usize, dilation: usize) -> Self {
        Self {
            input,
            output_size,
            stride,
            dilation,
        }
    }
}

impl<T: FloatDType> GradFnTyped<T, Dim4> for Unfold1dBackwardTyped<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim4>) {
        if self.input.requires_grad_typed() {
            let folded = grad_output.fold1d_dilated(self.output_size, self.stride, self.dilation);
            self.input.backward_with_typed(folded);
        }
    }

    fn name(&self) -> &'static str {
        "Unfold1dBackwardTyped"
    }
}

/// Typed gradient for Unfold2d: y = unfold2d(x, sizes, strides, dilations)
/// Input: Dim4, Output: Dim6
/// ∂L/∂x = fold2d(∂L/∂y, output_size, strides, dilations)
pub struct Unfold2dBackwardTyped<T: FloatDType> {
    input: Tensor<T, Dim4>,
    output_size: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl<T: FloatDType> Unfold2dBackwardTyped<T> {
    pub fn new(
        input: Tensor<T, Dim4>,
        output_size: (usize, usize),
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

impl<T: FloatDType> GradFnTyped<T, Dim6> for Unfold2dBackwardTyped<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim6>) {
        if self.input.requires_grad_typed() {
            let folded = grad_output.fold2d_dilated(self.output_size, self.strides, self.dilations);
            self.input.backward_with_typed(folded);
        }
    }

    fn name(&self) -> &'static str {
        "Unfold2dBackwardTyped"
    }
}

/// Typed gradient for Unfold3d: y = unfold3d(x, sizes, strides, dilations)
/// Input: Dim5, Output: Dim8
/// ∂L/∂x = fold3d(∂L/∂y, output_size, strides, dilations)
pub struct Unfold3dBackwardTyped<T: FloatDType> {
    input: Tensor<T, Dim5>,
    output_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    dilations: (usize, usize, usize),
}

impl<T: FloatDType> Unfold3dBackwardTyped<T> {
    pub fn new(
        input: Tensor<T, Dim5>,
        output_size: (usize, usize, usize),
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

impl<T: FloatDType> GradFnTyped<T, Dim8> for Unfold3dBackwardTyped<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim8>) {
        if self.input.requires_grad_typed() {
            let folded = grad_output.fold3d_dilated(self.output_size, self.strides, self.dilations);
            self.input.backward_with_typed(folded);
        }
    }

    fn name(&self) -> &'static str {
        "Unfold3dBackwardTyped"
    }
}

/// Typed gradient for Fold1d: y = fold1d(x, output_size, stride, dilation)
/// Input: Dim4, Output: Dim3
/// ∂L/∂x = unfold1d(∂L/∂y, kernel_size, stride, dilation)
pub struct Fold1dBackwardTyped<T: FloatDType> {
    input: Tensor<T, Dim4>,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
}

impl<T: FloatDType> Fold1dBackwardTyped<T> {
    pub fn new(input: Tensor<T, Dim4>, kernel_size: usize, stride: usize, dilation: usize) -> Self {
        Self {
            input,
            kernel_size,
            stride,
            dilation,
        }
    }
}

impl<T: FloatDType> GradFnTyped<T, Dim3> for Fold1dBackwardTyped<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim3>) {
        if self.input.requires_grad_typed() {
            let unfolded =
                grad_output.unfold1d_dilated(self.kernel_size, self.stride, self.dilation);
            self.input.backward_with_typed(unfolded);
        }
    }

    fn name(&self) -> &'static str {
        "Fold1dBackwardTyped"
    }
}

/// Typed gradient for Fold2d: y = fold2d(x, output_size, strides, dilations)
/// Input: Dim6, Output: Dim4
/// ∂L/∂x = unfold2d(∂L/∂y, kernel_size, strides, dilations)
pub struct Fold2dBackwardTyped<T: FloatDType> {
    input: Tensor<T, Dim6>,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl<T: FloatDType> Fold2dBackwardTyped<T> {
    pub fn new(
        input: Tensor<T, Dim6>,
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

impl<T: FloatDType> GradFnTyped<T, Dim4> for Fold2dBackwardTyped<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim4>) {
        if self.input.requires_grad_typed() {
            let unfolded =
                grad_output.unfold2d_dilated(self.kernel_size, self.strides, self.dilations);
            self.input.backward_with_typed(unfolded);
        }
    }

    fn name(&self) -> &'static str {
        "Fold2dBackwardTyped"
    }
}

/// Typed gradient for Fold3d: y = fold3d(x, output_size, strides, dilations)
/// Input: Dim8, Output: Dim5
/// ∂L/∂x = unfold3d(∂L/∂y, kernel_size, strides, dilations)
pub struct Fold3dBackwardTyped<T: FloatDType> {
    input: Tensor<T, Dim8>,
    kernel_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    dilations: (usize, usize, usize),
}

impl<T: FloatDType> Fold3dBackwardTyped<T> {
    pub fn new(
        input: Tensor<T, Dim8>,
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

impl<T: FloatDType> GradFnTyped<T, Dim5> for Fold3dBackwardTyped<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim5>) {
        if self.input.requires_grad_typed() {
            let unfolded =
                grad_output.unfold3d_dilated(self.kernel_size, self.strides, self.dilations);
            self.input.backward_with_typed(unfolded);
        }
    }

    fn name(&self) -> &'static str {
        "Fold3dBackwardTyped"
    }
}

// ============================================================================
// Dimension Conversion Backward Structs
// ============================================================================

/// Typed gradient for into_dyn: y = x.into_dyn()
/// Input D -> Output DimDyn
/// This is a no-op that just changes the static dimension type.
pub struct IntoDynBackwardTyped<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> IntoDynBackwardTyped<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFnTyped<T, DimDyn> for IntoDynBackwardTyped<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad_typed() {
            // Convert gradient from DimDyn back to D
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_output.inner.clone(),
                autograd_typed: None,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with_typed(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "IntoDynBackwardTyped"
    }
}

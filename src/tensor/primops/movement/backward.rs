//! Backward (gradient) operations for movement primitives

use std::marker::PhantomData;

use crate::tensor::{Dim3, Dim4, Dim5, Dim6, Dim8, DimDyn, Dimension, FloatDType, GradFn, Tensor};

// ============================================================================
// Typed Backward Structs (new system with static dimension typing)
// ============================================================================

/// Typed gradient for Squeeze: y = squeeze(x, dim)
/// Input is D, output is D::Smaller
pub struct SqueezeBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    dim: usize,
}

impl<T: FloatDType, D: Dimension> SqueezeBackward<T, D> {
    pub fn new(input: Tensor<T, D>, dim: usize) -> Self {
        Self { input, dim }
    }
}

// Output is D::Smaller, so we implement GradFn for that dimension
impl<T: FloatDType, D: Dimension> GradFn<T, D::Smaller> for SqueezeBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Smaller>) {
        if self.input.requires_grad() {
            // Unsqueeze grad from D::Smaller to D
            // Use DimDyn intermediate for type flexibility
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_unsqueezed = grad_dyn.unsqueeze(self.dim);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_unsqueezed.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "SqueezeBackward"
    }
}

/// Typed gradient for Unsqueeze: y = unsqueeze(x, dim)
/// Input is D, output is D::Larger
pub struct UnsqueezeBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    dim: usize,
}

impl<T: FloatDType, D: Dimension> UnsqueezeBackward<T, D> {
    pub fn new(input: Tensor<T, D>, dim: usize) -> Self {
        Self { input, dim }
    }
}

// Output is D::Larger, so we implement GradFn for that dimension
impl<T: FloatDType, D: Dimension> GradFn<T, D::Larger> for UnsqueezeBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D::Larger>) {
        if self.input.requires_grad() {
            // Squeeze grad from D::Larger to D
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_squeezed = grad_dyn.squeeze(self.dim);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_squeezed.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }
}

/// Typed gradient for Pad: y = pad(x, padding)
/// Same dimension, D -> D
pub struct PadBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    padding: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> PadBackward<T, D> {
    pub fn new(input: Tensor<T, D>, padding: Vec<(usize, usize)>) -> Self {
        Self { input, padding }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for PadBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
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
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "PadBackward"
    }
}

/// Typed gradient for Slice: y = slice(x, ranges)
/// Same dimension, D -> D
pub struct SliceBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    ranges: Vec<(usize, usize)>,
}

impl<T: FloatDType, D: Dimension> SliceBackward<T, D> {
    pub fn new(input: Tensor<T, D>, ranges: Vec<(usize, usize)>) -> Self {
        Self { input, ranges }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for SliceBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        if self.input.requires_grad() {
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
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
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "SliceBackward"
    }
}

/// Typed gradient for Reshape: y = reshape(x, new_shape)
/// Dimension can change, but we treat output as DIn, input as DOut
pub struct ReshapeBackward<T: FloatDType, DIn: Dimension, DOut: Dimension> {
    input: Tensor<T, DIn>,
    _output_dim: PhantomData<DOut>,
}

impl<T: FloatDType, DIn: Dimension, DOut: Dimension> ReshapeBackward<T, DIn, DOut> {
    pub fn new(input: Tensor<T, DIn>) -> Self {
        Self {
            input,
            _output_dim: PhantomData,
        }
    }
}

impl<T: FloatDType, DIn: Dimension, DOut: Dimension> GradFn<T, DOut>
    for ReshapeBackward<T, DIn, DOut>
{
    fn backward(&self, grad_output: &Tensor<T, DOut>) {
        if self.input.requires_grad() {
            let grad_dyn: Tensor<T, DimDyn> = Tensor {
                inner: grad_output.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            let grad_reshaped = grad_dyn.reshape_dyn(self.input.shape());
            let grad_input: Tensor<T, DIn> = Tensor {
                inner: grad_reshaped.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

/// Typed gradient for Permute: y = permute(x, axes)
/// Input D -> Output DimDyn
pub struct PermuteBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    axes: Vec<usize>,
}

impl<T: FloatDType, D: Dimension> PermuteBackward<T, D> {
    pub fn new(input: Tensor<T, D>, axes: Vec<usize>) -> Self {
        Self { input, axes }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, DimDyn> for PermuteBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad() {
            // Compute inverse permutation
            let mut inverse = vec![0; self.axes.len()];
            for (i, &ax) in self.axes.iter().enumerate() {
                inverse[ax] = i;
            }
            let grad_permuted = grad_output.permute(&inverse);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_permuted.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "PermuteBackward"
    }
}

/// Typed gradient for Transpose: y = transpose(x)
/// Same dimension, D -> D (swaps last two axes)
pub struct TransposeBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

/// Typed gradient for Concat: z = concat([a, b, ...], axis)
/// ∂L/∂a = slice(∂L/∂z, a's range), etc.
pub struct ConcatBackward<T: FloatDType, D: Dimension> {
    inputs: Vec<Tensor<T, D>>,
    axis: usize,
}

impl<T: FloatDType, D: Dimension> ConcatBackward<T, D> {
    pub fn new(inputs: Vec<Tensor<T, D>>, axis: usize) -> Self {
        Self { inputs, axis }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, D> for ConcatBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, D>) {
        let mut offset = 0;
        for input in &self.inputs {
            if input.requires_grad() {
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
                input.backward_with(grad_slice);
            }
            offset += input.shape()[self.axis];
        }
    }

    fn name(&self) -> &'static str {
        "ConcatBackward"
    }
}

/// Typed gradient for Expand: y = expand(x, new_shape)
/// ∂L/∂x = sum(∂L/∂y) over broadcast dimensions
pub struct ExpandBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType, D: Dimension> ExpandBackward<T, D> {
    pub fn new(input: Tensor<T, D>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

// Expand returns DimDyn, so we implement GradFn for DimDyn
impl<T: FloatDType, D: Dimension> GradFn<T, DimDyn> for ExpandBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad() {
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
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}

/// Typed gradient for ReshapeDyn: y = reshape_dyn(x, new_shape)
/// ∂L/∂x = reshape(∂L/∂y, original_shape)
pub struct ReshapeDynBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
    original_shape: Vec<usize>,
}

impl<T: FloatDType, D: Dimension> ReshapeDynBackward<T, D> {
    pub fn new(input: Tensor<T, D>, original_shape: Vec<usize>) -> Self {
        Self {
            input,
            original_shape,
        }
    }
}

// reshape_dyn returns DimDyn
impl<T: FloatDType, D: Dimension> GradFn<T, DimDyn> for ReshapeDynBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad() {
            let grad_reshaped = grad_output.reshape_dyn(&self.original_shape);
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_reshaped.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "ReshapeDynBackward"
    }
}

impl<T: FloatDType, D: Dimension> TransposeBackward<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, DimDyn> for TransposeBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad() {
            let grad_transposed = grad_output.transpose();
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_transposed.inner,
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

// ============================================================================
// Typed Unfold/Fold Backward Structs
// ============================================================================

/// Typed gradient for Unfold1d: y = unfold1d(x, size, stride, dilation)
/// Input: Dim3, Output: Dim4
/// ∂L/∂x = fold1d(∂L/∂y, output_size, stride, dilation)
pub struct Unfold1dBackward<T: FloatDType> {
    input: Tensor<T, Dim3>,
    output_size: usize,
    stride: usize,
    dilation: usize,
}

impl<T: FloatDType> Unfold1dBackward<T> {
    pub fn new(input: Tensor<T, Dim3>, output_size: usize, stride: usize, dilation: usize) -> Self {
        Self {
            input,
            output_size,
            stride,
            dilation,
        }
    }
}

impl<T: FloatDType> GradFn<T, Dim4> for Unfold1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim4>) {
        if self.input.requires_grad() {
            let folded = grad_output.fold1d_dilated(self.output_size, self.stride, self.dilation);
            self.input.backward_with(folded);
        }
    }

    fn name(&self) -> &'static str {
        "Unfold1dBackward"
    }
}

/// Typed gradient for Unfold2d: y = unfold2d(x, sizes, strides, dilations)
/// Input: Dim4, Output: Dim6
/// ∂L/∂x = fold2d(∂L/∂y, output_size, strides, dilations)
pub struct Unfold2dBackward<T: FloatDType> {
    input: Tensor<T, Dim4>,
    output_size: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl<T: FloatDType> Unfold2dBackward<T> {
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

impl<T: FloatDType> GradFn<T, Dim6> for Unfold2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim6>) {
        if self.input.requires_grad() {
            let folded = grad_output.fold2d_dilated(self.output_size, self.strides, self.dilations);
            self.input.backward_with(folded);
        }
    }

    fn name(&self) -> &'static str {
        "Unfold2dBackward"
    }
}

/// Typed gradient for Unfold3d: y = unfold3d(x, sizes, strides, dilations)
/// Input: Dim5, Output: Dim8
/// ∂L/∂x = fold3d(∂L/∂y, output_size, strides, dilations)
pub struct Unfold3dBackward<T: FloatDType> {
    input: Tensor<T, Dim5>,
    output_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    dilations: (usize, usize, usize),
}

impl<T: FloatDType> Unfold3dBackward<T> {
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

impl<T: FloatDType> GradFn<T, Dim8> for Unfold3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim8>) {
        if self.input.requires_grad() {
            let folded = grad_output.fold3d_dilated(self.output_size, self.strides, self.dilations);
            self.input.backward_with(folded);
        }
    }

    fn name(&self) -> &'static str {
        "Unfold3dBackward"
    }
}

/// Typed gradient for Fold1d: y = fold1d(x, output_size, stride, dilation)
/// Input: Dim4, Output: Dim3
/// ∂L/∂x = unfold1d(∂L/∂y, kernel_size, stride, dilation)
pub struct Fold1dBackward<T: FloatDType> {
    input: Tensor<T, Dim4>,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
}

impl<T: FloatDType> Fold1dBackward<T> {
    pub fn new(input: Tensor<T, Dim4>, kernel_size: usize, stride: usize, dilation: usize) -> Self {
        Self {
            input,
            kernel_size,
            stride,
            dilation,
        }
    }
}

impl<T: FloatDType> GradFn<T, Dim3> for Fold1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim3>) {
        if self.input.requires_grad() {
            let unfolded =
                grad_output.unfold1d_dilated(self.kernel_size, self.stride, self.dilation);
            self.input.backward_with(unfolded);
        }
    }

    fn name(&self) -> &'static str {
        "Fold1dBackward"
    }
}

/// Typed gradient for Fold2d: y = fold2d(x, output_size, strides, dilations)
/// Input: Dim6, Output: Dim4
/// ∂L/∂x = unfold2d(∂L/∂y, kernel_size, strides, dilations)
pub struct Fold2dBackward<T: FloatDType> {
    input: Tensor<T, Dim6>,
    kernel_size: (usize, usize),
    strides: (usize, usize),
    dilations: (usize, usize),
}

impl<T: FloatDType> Fold2dBackward<T> {
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

impl<T: FloatDType> GradFn<T, Dim4> for Fold2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim4>) {
        if self.input.requires_grad() {
            let unfolded =
                grad_output.unfold2d_dilated(self.kernel_size, self.strides, self.dilations);
            self.input.backward_with(unfolded);
        }
    }

    fn name(&self) -> &'static str {
        "Fold2dBackward"
    }
}

/// Typed gradient for Fold3d: y = fold3d(x, output_size, strides, dilations)
/// Input: Dim8, Output: Dim5
/// ∂L/∂x = unfold3d(∂L/∂y, kernel_size, strides, dilations)
pub struct Fold3dBackward<T: FloatDType> {
    input: Tensor<T, Dim8>,
    kernel_size: (usize, usize, usize),
    strides: (usize, usize, usize),
    dilations: (usize, usize, usize),
}

impl<T: FloatDType> Fold3dBackward<T> {
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

impl<T: FloatDType> GradFn<T, Dim5> for Fold3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T, Dim5>) {
        if self.input.requires_grad() {
            let unfolded =
                grad_output.unfold3d_dilated(self.kernel_size, self.strides, self.dilations);
            self.input.backward_with(unfolded);
        }
    }

    fn name(&self) -> &'static str {
        "Fold3dBackward"
    }
}

// ============================================================================
// Dimension Conversion Backward Structs
// ============================================================================

/// Typed gradient for into_dyn: y = x.into_dyn()
/// Input D -> Output DimDyn
/// This is a no-op that just changes the static dimension type.
pub struct IntoDynBackward<T: FloatDType, D: Dimension> {
    input: Tensor<T, D>,
}

impl<T: FloatDType, D: Dimension> IntoDynBackward<T, D> {
    pub fn new(input: Tensor<T, D>) -> Self {
        Self { input }
    }
}

impl<T: FloatDType, D: Dimension> GradFn<T, DimDyn> for IntoDynBackward<T, D> {
    fn backward(&self, grad_output: &Tensor<T, DimDyn>) {
        if self.input.requires_grad() {
            // Convert gradient from DimDyn back to D
            let grad_input: Tensor<T, D> = Tensor {
                inner: grad_output.inner.clone(),
                _dtype: PhantomData,
                _dim: PhantomData,
            };
            self.input.backward_with(grad_input);
        }
    }

    fn name(&self) -> &'static str {
        "IntoDynBackward"
    }
}

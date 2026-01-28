//! Pooling layers
//!
//! Implements MaxPool1d, MaxPool2d, MaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d,
//! AdaptiveAvgPool2d, and AdaptiveMaxPool2d.

use super::{Module, ParameterBase};
use crate::functional;
use eclat::tensor::Tensor;
use eclat::tensor::dim::{D3, D4, D5};

// ============================================================================
// MaxPool2d
// ============================================================================

/// 2D Max Pooling layer.
///
/// Applies a 2D max pooling over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `[N, C, H, W]`
/// - Output: `[N, C, H_out, W_out]`
///
/// Where:
/// - `H_out = (H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
/// - `W_out = (W + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::MaxPool2d;
/// use eclat::tensor::{Tensor, dim::D4};
///
/// let pool = MaxPool2d::new((2, 2));
/// let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
/// let output = pool.forward_d4(&input);  // [1, 64, 16, 16]
/// ```
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

impl MaxPool2d {
    /// Create a new MaxPool2d layer.
    ///
    /// By default, stride equals kernel_size.
    ///
    /// Use builder methods to customize:
    /// ```ignore
    /// let pool = MaxPool2d::new((3, 3))
    ///     .with_stride((2, 2))
    ///     .with_padding((1, 1));
    /// ```
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size, // default: stride = kernel_size
            padding: (0, 0),
            dilation: (1, 1),
        }
    }

    /// Set the stride. Default is kernel_size.
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding. Default is (0, 0).
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation. Default is (1, 1).
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C, H, W]
    ///
    /// # Returns
    /// Output tensor of shape [N, C, H_out, W_out]
    pub fn forward_d4(&self, input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
        functional::max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation)
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Get the stride.
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// Get the padding.
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// Get the dilation.
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }
}

impl Module for MaxPool2d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![] // No learnable parameters
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {
        // No-op for pooling layers
    }

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for MaxPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool2d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .finish()
    }
}

// ============================================================================
// AvgPool2d
// ============================================================================

/// 2D Average Pooling layer.
///
/// Applies a 2D average pooling over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `[N, C, H, W]`
/// - Output: `[N, C, H_out, W_out]`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::AvgPool2d;
/// use eclat::tensor::{Tensor, dim::D4};
///
/// let pool = AvgPool2d::new((2, 2));
/// let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
/// let output = pool.forward_d4(&input);  // [1, 64, 16, 16]
/// ```
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    /// Create a new AvgPool2d layer.
    ///
    /// By default, stride equals kernel_size.
    pub fn new(kernel_size: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0),
        }
    }

    /// Set the stride. Default is kernel_size.
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding. Default is (0, 0).
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Forward pass.
    pub fn forward_d4(&self, input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
        functional::avg_pool2d(input, self.kernel_size, self.stride, self.padding)
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Get the stride.
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// Get the padding.
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

impl Module for AvgPool2d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {}

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for AvgPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AvgPool2d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .finish()
    }
}

// ============================================================================
// MaxPool1d
// ============================================================================

/// 1D Max Pooling layer.
///
/// # Shape
/// - Input: `[N, C, L]`
/// - Output: `[N, C, L_out]`
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl MaxPool1d {
    /// Create a new MaxPool1d layer.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
            dilation: 1,
        }
    }

    /// Set the stride.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation.
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Forward pass.
    pub fn forward_d3(&self, input: &Tensor<D3, f32>) -> Tensor<D3, f32> {
        functional::max_pool1d(input, self.kernel_size, self.stride, self.padding, self.dilation)
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl Module for MaxPool1d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {}

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for MaxPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool1d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .finish()
    }
}

// ============================================================================
// AvgPool1d
// ============================================================================

/// 1D Average Pooling layer.
///
/// # Shape
/// - Input: `[N, C, L]`
/// - Output: `[N, C, L_out]`
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool1d {
    /// Create a new AvgPool1d layer.
    pub fn new(kernel_size: usize) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: 0,
        }
    }

    /// Set the stride.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Forward pass.
    pub fn forward_d3(&self, input: &Tensor<D3, f32>) -> Tensor<D3, f32> {
        functional::avg_pool1d(input, self.kernel_size, self.stride, self.padding)
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl Module for AvgPool1d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {}

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for AvgPool1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AvgPool1d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .finish()
    }
}

// ============================================================================
// MaxPool3d
// ============================================================================

/// 3D Max Pooling layer.
///
/// # Shape
/// - Input: `[N, C, D, H, W]`
/// - Output: `[N, C, D_out, H_out, W_out]`
pub struct MaxPool3d {
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
}

impl MaxPool3d {
    /// Create a new MaxPool3d layer.
    pub fn new(kernel_size: (usize, usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
        }
    }

    /// Set the stride.
    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding.
    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation.
    pub fn with_dilation(mut self, dilation: (usize, usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Forward pass.
    pub fn forward_d5(&self, input: &Tensor<D5, f32>) -> Tensor<D5, f32> {
        functional::max_pool3d(input, self.kernel_size, self.stride, self.padding, self.dilation)
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> (usize, usize, usize) {
        self.kernel_size
    }
}

impl Module for MaxPool3d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {}

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for MaxPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MaxPool3d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .finish()
    }
}

// ============================================================================
// AvgPool3d
// ============================================================================

/// 3D Average Pooling layer.
///
/// # Shape
/// - Input: `[N, C, D, H, W]`
/// - Output: `[N, C, D_out, H_out, W_out]`
pub struct AvgPool3d {
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
}

impl AvgPool3d {
    /// Create a new AvgPool3d layer.
    pub fn new(kernel_size: (usize, usize, usize)) -> Self {
        Self {
            kernel_size,
            stride: kernel_size,
            padding: (0, 0, 0),
        }
    }

    /// Set the stride.
    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding.
    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Forward pass.
    pub fn forward_d5(&self, input: &Tensor<D5, f32>) -> Tensor<D5, f32> {
        functional::avg_pool3d(input, self.kernel_size, self.stride, self.padding)
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> (usize, usize, usize) {
        self.kernel_size
    }
}

impl Module for AvgPool3d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {}

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for AvgPool3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AvgPool3d")
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .finish()
    }
}

// ============================================================================
// AdaptiveAvgPool2d
// ============================================================================

/// Adaptive 2D Average Pooling layer.
///
/// Reduces spatial dimensions to (1, 1) regardless of input size.
/// Commonly used before the final fully-connected layer in CNNs.
///
/// # Shape
/// - Input: `[N, C, H, W]`
/// - Output: `[N, C, 1, 1]`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::AdaptiveAvgPool2d;
/// use eclat::tensor::{Tensor, dim::D4};
///
/// let pool = AdaptiveAvgPool2d::new();
/// let input: Tensor<D4, f32> = Tensor::input([1, 512, 7, 7]);
/// let output = pool.forward_d4(&input);  // [1, 512, 1, 1]
/// ```
pub struct AdaptiveAvgPool2d;

impl AdaptiveAvgPool2d {
    /// Create a new AdaptiveAvgPool2d layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass.
    pub fn forward_d4(&self, input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
        functional::adaptive_avg_pool2d(input)
    }
}

impl Default for AdaptiveAvgPool2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for AdaptiveAvgPool2d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {}

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for AdaptiveAvgPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveAvgPool2d").finish()
    }
}

// ============================================================================
// AdaptiveMaxPool2d
// ============================================================================

/// Adaptive 2D Max Pooling layer.
///
/// Reduces spatial dimensions to (1, 1) regardless of input size.
///
/// # Shape
/// - Input: `[N, C, H, W]`
/// - Output: `[N, C, 1, 1]`
pub struct AdaptiveMaxPool2d;

impl AdaptiveMaxPool2d {
    /// Create a new AdaptiveMaxPool2d layer.
    pub fn new() -> Self {
        Self
    }

    /// Forward pass.
    pub fn forward_d4(&self, input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
        functional::adaptive_max_pool2d(input)
    }
}

impl Default for AdaptiveMaxPool2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for AdaptiveMaxPool2d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        vec![]
    }

    fn train(&mut self, _mode: bool) {}

    fn is_training(&self) -> bool {
        false
    }
}

impl std::fmt::Debug for AdaptiveMaxPool2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveMaxPool2d").finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool2d_output_shape() {
        // Input: [1, 64, 32, 32], kernel: 2x2, stride: 2x2
        // Output: [1, 64, 16, 16]
        let pool = MaxPool2d::new((2, 2));
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = pool.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        // Input: [1, 64, 32, 32], kernel: 3x3, stride: 2x2, padding: 1
        // Output: [1, 64, 16, 16]
        let pool = MaxPool2d::new((3, 3))
            .with_stride((2, 2))
            .with_padding((1, 1));
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = pool.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_avg_pool2d_output_shape() {
        let pool = AvgPool2d::new((2, 2));
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = pool.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_max_pool1d_output_shape() {
        // Input: [2, 3, 10], kernel: 2, stride: 2
        // Output: [2, 3, 5]
        let pool = MaxPool1d::new(2);
        let input: Tensor<D3, f32> = Tensor::input([2, 3, 10]);
        let output = pool.forward_d3(&input);
        assert_eq!(output.shape(), vec![2, 3, 5]);
    }

    #[test]
    fn test_avg_pool1d_output_shape() {
        let pool = AvgPool1d::new(2);
        let input: Tensor<D3, f32> = Tensor::input([2, 3, 10]);
        let output = pool.forward_d3(&input);
        assert_eq!(output.shape(), vec![2, 3, 5]);
    }

    #[test]
    fn test_max_pool3d_output_shape() {
        // Input: [1, 64, 16, 32, 32], kernel: 2x2x2, stride: 2x2x2
        // Output: [1, 64, 8, 16, 16]
        let pool = MaxPool3d::new((2, 2, 2));
        let input: Tensor<D5, f32> = Tensor::input([1, 64, 16, 32, 32]);
        let output = pool.forward_d5(&input);
        assert_eq!(output.shape(), vec![1, 64, 8, 16, 16]);
    }

    #[test]
    fn test_avg_pool3d_output_shape() {
        let pool = AvgPool3d::new((2, 2, 2));
        let input: Tensor<D5, f32> = Tensor::input([1, 64, 16, 32, 32]);
        let output = pool.forward_d5(&input);
        assert_eq!(output.shape(), vec![1, 64, 8, 16, 16]);
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        let pool = AdaptiveAvgPool2d::new();
        let input: Tensor<D4, f32> = Tensor::input([1, 512, 7, 7]);
        let output = pool.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 512, 1, 1]);
    }

    #[test]
    fn test_adaptive_max_pool2d() {
        let pool = AdaptiveMaxPool2d::new();
        let input: Tensor<D4, f32> = Tensor::input([1, 512, 7, 7]);
        let output = pool.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 512, 1, 1]);
    }

    #[test]
    fn test_pooling_no_parameters() {
        let pool = MaxPool2d::new((2, 2));
        assert!(pool.parameters().is_empty());
        assert!(pool.named_parameters().is_empty());
    }
}

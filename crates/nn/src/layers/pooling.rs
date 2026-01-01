//! Pooling layers
//!
//! プーリング層の実装を提供します。
//!
//! # 層の種類
//!
//! - [`MaxPool1d`] - 1D最大プーリング
//! - [`MaxPool2d`] - 2D最大プーリング
//! - [`MaxPool3d`] - 3D最大プーリング
//! - [`AvgPool1d`] - 1D平均プーリング
//! - [`AvgPool2d`] - 2D平均プーリング
//! - [`AvgPool3d`] - 3D平均プーリング
//! - [`AdaptiveAvgPool2d`] - 2D適応的平均プーリング
//! - [`AdaptiveMaxPool2d`] - 2D適応的最大プーリング

use std::collections::HashMap;

use harp::tensor::{Dim3, Dim4, Dim5, DimDyn, Tensor};

use crate::functional::{
    adaptive_avg_pool2d, adaptive_max_pool2d, avg_pool1d, avg_pool2d, avg_pool3d, max_pool1d,
    max_pool2d, max_pool3d,
};
use crate::{Module, ParameterMut};

// ============================================================================
// MaxPool1d
// ============================================================================

/// 1D Max Pooling layer
///
/// # Example
///
/// ```ignore
/// let pool = MaxPool1d::builder(2).stride(2).build();
/// let input = Tensor::<f32, Dim3>::ones([1, 16, 100]);
/// let output = pool.forward(&input); // [1, 16, 50]
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool1d {
    /// Create a new MaxPool1d builder
    pub fn builder(kernel_size: usize) -> MaxPool1dBuilder {
        MaxPool1dBuilder {
            kernel_size,
            stride: None,
            padding: 0,
        }
    }

    /// Kernel size
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Stride
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Padding
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// Forward pass
    pub fn forward(&self, input: &Tensor<f32, Dim3>) -> Tensor<f32, Dim3> {
        max_pool1d(input, self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Debug, Clone)]
pub struct MaxPool1dBuilder {
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
}

impl MaxPool1dBuilder {
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = Some(stride);
        self
    }

    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    pub fn build(self) -> MaxPool1d {
        MaxPool1d {
            kernel_size: self.kernel_size,
            stride: self.stride.unwrap_or(self.kernel_size),
            padding: self.padding,
        }
    }
}

impl Module<f32> for MaxPool1d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// AvgPool1d
// ============================================================================

/// 1D Average Pooling layer
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl AvgPool1d {
    pub fn builder(kernel_size: usize) -> AvgPool1dBuilder {
        AvgPool1dBuilder {
            kernel_size,
            stride: None,
            padding: 0,
        }
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn forward(&self, input: &Tensor<f32, Dim3>) -> Tensor<f32, Dim3> {
        avg_pool1d(input, self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Debug, Clone)]
pub struct AvgPool1dBuilder {
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
}

impl AvgPool1dBuilder {
    pub fn stride(mut self, stride: usize) -> Self {
        self.stride = Some(stride);
        self
    }

    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    pub fn build(self) -> AvgPool1d {
        AvgPool1d {
            kernel_size: self.kernel_size,
            stride: self.stride.unwrap_or(self.kernel_size),
            padding: self.padding,
        }
    }
}

impl Module<f32> for AvgPool1d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// MaxPool2d
// ============================================================================

/// 2D Max Pooling layer
///
/// # Example
///
/// ```ignore
/// let pool = MaxPool2d::builder((2, 2)).stride((2, 2)).build();
/// let input = Tensor::<f32, Dim4>::ones([1, 64, 32, 32]);
/// let output = pool.forward(&input); // [1, 64, 16, 16]
/// ```
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    pub fn builder(kernel_size: (usize, usize)) -> MaxPool2dBuilder {
        MaxPool2dBuilder {
            kernel_size,
            stride: None,
            padding: (0, 0),
        }
    }

    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    pub fn forward(&self, input: &Tensor<f32, Dim4>) -> Tensor<f32, Dim4> {
        max_pool2d(input, self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Debug, Clone)]
pub struct MaxPool2dBuilder {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
}

impl MaxPool2dBuilder {
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = Some(stride);
        self
    }

    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn build(self) -> MaxPool2d {
        MaxPool2d {
            kernel_size: self.kernel_size,
            stride: self.stride.unwrap_or(self.kernel_size),
            padding: self.padding,
        }
    }
}

impl Module<f32> for MaxPool2d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// AvgPool2d
// ============================================================================

/// 2D Average Pooling layer
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl AvgPool2d {
    pub fn builder(kernel_size: (usize, usize)) -> AvgPool2dBuilder {
        AvgPool2dBuilder {
            kernel_size,
            stride: None,
            padding: (0, 0),
        }
    }

    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    pub fn forward(&self, input: &Tensor<f32, Dim4>) -> Tensor<f32, Dim4> {
        avg_pool2d(input, self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Debug, Clone)]
pub struct AvgPool2dBuilder {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
}

impl AvgPool2dBuilder {
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = Some(stride);
        self
    }

    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn build(self) -> AvgPool2d {
        AvgPool2d {
            kernel_size: self.kernel_size,
            stride: self.stride.unwrap_or(self.kernel_size),
            padding: self.padding,
        }
    }
}

impl Module<f32> for AvgPool2d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// MaxPool3d
// ============================================================================

/// 3D Max Pooling layer
#[derive(Debug, Clone)]
pub struct MaxPool3d {
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
}

impl MaxPool3d {
    pub fn builder(kernel_size: (usize, usize, usize)) -> MaxPool3dBuilder {
        MaxPool3dBuilder {
            kernel_size,
            stride: None,
            padding: (0, 0, 0),
        }
    }

    pub fn kernel_size(&self) -> (usize, usize, usize) {
        self.kernel_size
    }

    pub fn stride(&self) -> (usize, usize, usize) {
        self.stride
    }

    pub fn padding(&self) -> (usize, usize, usize) {
        self.padding
    }

    pub fn forward(&self, input: &Tensor<f32, Dim5>) -> Tensor<f32, Dim5> {
        max_pool3d(input, self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Debug, Clone)]
pub struct MaxPool3dBuilder {
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
}

impl MaxPool3dBuilder {
    pub fn stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = Some(stride);
        self
    }

    pub fn padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn build(self) -> MaxPool3d {
        MaxPool3d {
            kernel_size: self.kernel_size,
            stride: self.stride.unwrap_or(self.kernel_size),
            padding: self.padding,
        }
    }
}

impl Module<f32> for MaxPool3d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// AvgPool3d
// ============================================================================

/// 3D Average Pooling layer
#[derive(Debug, Clone)]
pub struct AvgPool3d {
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
}

impl AvgPool3d {
    pub fn builder(kernel_size: (usize, usize, usize)) -> AvgPool3dBuilder {
        AvgPool3dBuilder {
            kernel_size,
            stride: None,
            padding: (0, 0, 0),
        }
    }

    pub fn kernel_size(&self) -> (usize, usize, usize) {
        self.kernel_size
    }

    pub fn stride(&self) -> (usize, usize, usize) {
        self.stride
    }

    pub fn padding(&self) -> (usize, usize, usize) {
        self.padding
    }

    pub fn forward(&self, input: &Tensor<f32, Dim5>) -> Tensor<f32, Dim5> {
        avg_pool3d(input, self.kernel_size, self.stride, self.padding)
    }
}

#[derive(Debug, Clone)]
pub struct AvgPool3dBuilder {
    kernel_size: (usize, usize, usize),
    stride: Option<(usize, usize, usize)>,
    padding: (usize, usize, usize),
}

impl AvgPool3dBuilder {
    pub fn stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = Some(stride);
        self
    }

    pub fn padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn build(self) -> AvgPool3d {
        AvgPool3d {
            kernel_size: self.kernel_size,
            stride: self.stride.unwrap_or(self.kernel_size),
            padding: self.padding,
        }
    }
}

impl Module<f32> for AvgPool3d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// AdaptiveAvgPool2d
// ============================================================================

/// 2D Adaptive Average Pooling layer
///
/// Automatically adjusts kernel size and stride to produce the target output size.
///
/// # Example
///
/// ```ignore
/// let pool = AdaptiveAvgPool2d::new((1, 1));
/// let input = Tensor::<f32, Dim4>::ones([1, 512, 7, 7]);
/// let output = pool.forward(&input); // [1, 512, 1, 1]
/// ```
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    pub fn output_size(&self) -> (usize, usize) {
        self.output_size
    }

    pub fn forward(&self, input: &Tensor<f32, Dim4>) -> Tensor<f32, Dim4> {
        adaptive_avg_pool2d(input, self.output_size)
    }
}

impl Module<f32> for AdaptiveAvgPool2d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// AdaptiveMaxPool2d
// ============================================================================

/// 2D Adaptive Max Pooling layer
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool2d {
    output_size: (usize, usize),
}

impl AdaptiveMaxPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    pub fn output_size(&self) -> (usize, usize) {
        self.output_size
    }

    pub fn forward(&self, input: &Tensor<f32, Dim4>) -> Tensor<f32, Dim4> {
        adaptive_max_pool2d(input, self.output_size)
    }
}

impl Module<f32> for AdaptiveMaxPool2d {
    fn parameters(&mut self) -> HashMap<String, &mut dyn ParameterMut<f32>> {
        HashMap::new()
    }

    fn load_parameters(&mut self, _params: HashMap<String, Tensor<f32, DimDyn>>) {}
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool1d_layer() {
        let pool = MaxPool1d::builder(2).stride(2).build();
        let input = Tensor::<f32, Dim3>::ones([1, 16, 100]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 16, 50]);
    }

    #[test]
    fn test_avg_pool1d_layer() {
        let pool = AvgPool1d::builder(2).build();
        let input = Tensor::<f32, Dim3>::ones([1, 16, 100]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 16, 50]);
    }

    #[test]
    fn test_max_pool2d_layer() {
        let pool = MaxPool2d::builder((2, 2)).stride((2, 2)).build();
        let input = Tensor::<f32, Dim4>::ones([1, 64, 32, 32]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 64, 16, 16]);
    }

    #[test]
    fn test_avg_pool2d_layer() {
        let pool = AvgPool2d::builder((2, 2)).build();
        let input = Tensor::<f32, Dim4>::ones([1, 64, 32, 32]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 64, 16, 16]);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        let pool = MaxPool2d::builder((3, 3))
            .stride((1, 1))
            .padding((1, 1))
            .build();
        let input = Tensor::<f32, Dim4>::ones([1, 64, 32, 32]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 64, 32, 32]);
    }

    #[test]
    fn test_adaptive_avg_pool2d_layer() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let input = Tensor::<f32, Dim4>::ones([1, 512, 7, 7]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 512, 1, 1]);
    }

    #[test]
    fn test_adaptive_max_pool2d_layer() {
        let pool = AdaptiveMaxPool2d::new((2, 2));
        let input = Tensor::<f32, Dim4>::ones([1, 512, 8, 8]);
        let output = pool.forward(&input);
        assert_eq!(output.shape(), &[1, 512, 2, 2]);
    }

    #[test]
    fn test_max_pool2d_no_parameters() {
        let mut pool = MaxPool2d::builder((2, 2)).build();
        let params = pool.parameters();
        assert!(params.is_empty());
    }
}

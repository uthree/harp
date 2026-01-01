//! Pooling operations
//!
//! プーリング操作を関数として提供します。

use harp::tensor::ops::PadValue;
use harp::tensor::{Dim3, Dim4, Dim5, Tensor};

// ============================================================================
// 1D Pooling
// ============================================================================

/// Max pooling 1D
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Stride of the pooling window
/// * `padding` - Padding to apply before pooling
///
/// # Returns
/// Output tensor of shape [N, C, L_out] where L_out = (L + 2*padding - kernel_size) / stride + 1
pub fn max_pool1d(
    input: &Tensor<f32, Dim3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor<f32, Dim3> {
    let padded = if padding > 0 {
        input.pad(&[(0, 0), (0, 0), (padding, padding)], PadValue::NegInf)
    } else {
        input.clone()
    };

    // unfold1d: [N, C, L] -> [N, C, L_out, K]
    let unfolded = padded.unfold1d(kernel_size, stride);

    // max along the last axis: [N, C, L_out, K] -> [N, C, L_out]
    unfolded.max(3)
}

/// Average pooling 1D
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `kernel_size` - Size of the pooling window
/// * `stride` - Stride of the pooling window
/// * `padding` - Padding to apply before pooling
///
/// # Returns
/// Output tensor of shape [N, C, L_out] where L_out = (L + 2*padding - kernel_size) / stride + 1
pub fn avg_pool1d(
    input: &Tensor<f32, Dim3>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor<f32, Dim3> {
    let padded = if padding > 0 {
        input.pad_zero(&[(0, 0), (0, 0), (padding, padding)])
    } else {
        input.clone()
    };

    // unfold1d: [N, C, L] -> [N, C, L_out, K]
    let unfolded = padded.unfold1d(kernel_size, stride);

    // sum along the last axis and divide by pool size
    let summed = unfolded.sum(3);
    let inv_pool_size = 1.0 / kernel_size as f32;
    summed * inv_pool_size
}

// ============================================================================
// 2D Pooling
// ============================================================================

/// Max pooling 2D
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `kernel_size` - Size of the pooling window (kH, kW)
/// * `stride` - Stride of the pooling window (sH, sW)
/// * `padding` - Padding to apply before pooling (pH, pW)
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
pub fn max_pool2d(
    input: &Tensor<f32, Dim4>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor<f32, Dim4> {
    let padded = if padding.0 > 0 || padding.1 > 0 {
        input.pad(
            &[
                (0, 0),
                (0, 0),
                (padding.0, padding.0),
                (padding.1, padding.1),
            ],
            PadValue::NegInf,
        )
    } else {
        input.clone()
    };

    // unfold2d: [N, C, H, W] -> [N, C, H_out, W_out, kH, kW]
    let unfolded = padded.unfold2d(kernel_size, stride);
    let shape = unfolded.shape().to_vec();

    // Reshape to merge the last two dimensions: [N, C, H_out, W_out, kH*kW]
    let pool_size = kernel_size.0 * kernel_size.1;
    let reshaped: Tensor<f32, Dim5> =
        unfolded.reshape([shape[0], shape[1], shape[2], shape[3], pool_size]);

    // Max along the last axis: [N, C, H_out, W_out]
    reshaped.max(4)
}

/// Average pooling 2D
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `kernel_size` - Size of the pooling window (kH, kW)
/// * `stride` - Stride of the pooling window (sH, sW)
/// * `padding` - Padding to apply before pooling (pH, pW)
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
pub fn avg_pool2d(
    input: &Tensor<f32, Dim4>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor<f32, Dim4> {
    let padded = if padding.0 > 0 || padding.1 > 0 {
        input.pad_zero(&[
            (0, 0),
            (0, 0),
            (padding.0, padding.0),
            (padding.1, padding.1),
        ])
    } else {
        input.clone()
    };

    // unfold2d: [N, C, H, W] -> [N, C, H_out, W_out, kH, kW]
    let unfolded = padded.unfold2d(kernel_size, stride);
    let shape = unfolded.shape().to_vec();

    // Reshape to merge the last two dimensions: [N, C, H_out, W_out, kH*kW]
    let pool_size = kernel_size.0 * kernel_size.1;
    let reshaped: Tensor<f32, Dim5> =
        unfolded.reshape([shape[0], shape[1], shape[2], shape[3], pool_size]);

    // Sum along the last axis and divide by pool size
    let summed = reshaped.sum(4);
    let inv_pool_size = 1.0 / pool_size as f32;
    summed * inv_pool_size
}

// ============================================================================
// 3D Pooling
// ============================================================================

/// Max pooling 3D
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `kernel_size` - Size of the pooling window (kD, kH, kW)
/// * `stride` - Stride of the pooling window (sD, sH, sW)
/// * `padding` - Padding to apply before pooling (pD, pH, pW)
///
/// # Returns
/// Output tensor of shape [N, C, D_out, H_out, W_out]
pub fn max_pool3d(
    input: &Tensor<f32, Dim5>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Tensor<f32, Dim5> {
    let padded = if padding.0 > 0 || padding.1 > 0 || padding.2 > 0 {
        input.pad(
            &[
                (0, 0),
                (0, 0),
                (padding.0, padding.0),
                (padding.1, padding.1),
                (padding.2, padding.2),
            ],
            PadValue::NegInf,
        )
    } else {
        input.clone()
    };

    // unfold3d: [N, C, D, H, W] -> [N, C, D_out, H_out, W_out, kD, kH, kW]
    let unfolded = padded.unfold3d(kernel_size, stride);
    let shape = unfolded.shape().to_vec();

    // Reshape to merge the last three dimensions: [N, C, D_out, H_out, W_out, kD*kH*kW]
    let pool_size = kernel_size.0 * kernel_size.1 * kernel_size.2;
    let reshaped = unfolded
        .into_dyn()
        .reshape_dyn(&[shape[0], shape[1], shape[2], shape[3], shape[4], pool_size]);

    // Max along the last axis: [N, C, D_out, H_out, W_out]
    reshaped.max(5).into_dim5()
}

/// Average pooling 3D
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `kernel_size` - Size of the pooling window (kD, kH, kW)
/// * `stride` - Stride of the pooling window (sD, sH, sW)
/// * `padding` - Padding to apply before pooling (pD, pH, pW)
///
/// # Returns
/// Output tensor of shape [N, C, D_out, H_out, W_out]
pub fn avg_pool3d(
    input: &Tensor<f32, Dim5>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Tensor<f32, Dim5> {
    let padded = if padding.0 > 0 || padding.1 > 0 || padding.2 > 0 {
        input.pad_zero(&[
            (0, 0),
            (0, 0),
            (padding.0, padding.0),
            (padding.1, padding.1),
            (padding.2, padding.2),
        ])
    } else {
        input.clone()
    };

    // unfold3d: [N, C, D, H, W] -> [N, C, D_out, H_out, W_out, kD, kH, kW]
    let unfolded = padded.unfold3d(kernel_size, stride);
    let shape = unfolded.shape().to_vec();

    // Reshape to merge the last three dimensions: [N, C, D_out, H_out, W_out, kD*kH*kW]
    let pool_size = kernel_size.0 * kernel_size.1 * kernel_size.2;
    let reshaped = unfolded
        .into_dyn()
        .reshape_dyn(&[shape[0], shape[1], shape[2], shape[3], shape[4], pool_size]);

    // Sum along the last axis and divide by pool size
    let summed = reshaped.sum(5).into_dim5();
    let inv_pool_size = 1.0 / pool_size as f32;
    summed * inv_pool_size
}

// ============================================================================
// Adaptive Pooling
// ============================================================================

/// Adaptive average pooling 2D
///
/// Automatically calculates kernel size and stride to produce the target output size.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `output_size` - Target output size (H_out, W_out)
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
pub fn adaptive_avg_pool2d(
    input: &Tensor<f32, Dim4>,
    output_size: (usize, usize),
) -> Tensor<f32, Dim4> {
    let shape = input.shape();
    let (h, w) = (shape[2], shape[3]);
    let (out_h, out_w) = output_size;

    // Calculate kernel size and stride for each dimension
    let stride_h = h / out_h;
    let stride_w = w / out_w;
    let kernel_h = h - (out_h - 1) * stride_h;
    let kernel_w = w - (out_w - 1) * stride_w;

    avg_pool2d(input, (kernel_h, kernel_w), (stride_h, stride_w), (0, 0))
}

/// Adaptive max pooling 2D
///
/// Automatically calculates kernel size and stride to produce the target output size.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `output_size` - Target output size (H_out, W_out)
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
pub fn adaptive_max_pool2d(
    input: &Tensor<f32, Dim4>,
    output_size: (usize, usize),
) -> Tensor<f32, Dim4> {
    let shape = input.shape();
    let (h, w) = (shape[2], shape[3]);
    let (out_h, out_w) = output_size;

    // Calculate kernel size and stride for each dimension
    let stride_h = h / out_h;
    let stride_w = w / out_w;
    let kernel_h = h - (out_h - 1) * stride_h;
    let kernel_w = w - (out_w - 1) * stride_w;

    max_pool2d(input, (kernel_h, kernel_w), (stride_h, stride_w), (0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool1d_basic() {
        let input = Tensor::<f32, Dim3>::ones([1, 1, 4]);
        let output = max_pool1d(&input, 2, 2, 0);
        assert_eq!(output.shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_avg_pool1d_basic() {
        let input = Tensor::<f32, Dim3>::ones([1, 1, 4]);
        let output = avg_pool1d(&input, 2, 2, 0);
        assert_eq!(output.shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_max_pool2d_basic() {
        let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 4]);
        let output = max_pool2d(&input, (2, 2), (2, 2), (0, 0));
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_avg_pool2d_basic() {
        let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 4]);
        let output = avg_pool2d(&input, (2, 2), (2, 2), (0, 0));
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 4]);
        let output = max_pool2d(&input, (3, 3), (1, 1), (1, 1));
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        let input = Tensor::<f32, Dim4>::ones([1, 1, 8, 8]);
        let output = adaptive_avg_pool2d(&input, (2, 2));
        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_adaptive_max_pool2d() {
        let input = Tensor::<f32, Dim4>::ones([1, 1, 8, 8]);
        let output = adaptive_max_pool2d(&input, (1, 1));
        assert_eq!(output.shape(), &[1, 1, 1, 1]);
    }
}

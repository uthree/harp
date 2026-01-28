//! Pooling operations
//!
//! Implements max pooling and average pooling operations.

use eclat::tensor::Tensor;
use eclat::tensor::dim::{D3, D4, D5};

// ============================================================================
// MaxPool1d
// ============================================================================

/// Applies a 1D max pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `kernel_size` - Size of the window
/// * `stride` - Stride of the window. Default: kernel_size
/// * `padding` - Padding added to both sides
/// * `dilation` - Spacing between kernel elements
///
/// # Returns
/// Output tensor of shape [N, C, L_out]
///
/// Where: `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
pub fn max_pool1d(
    input: &Tensor<D3, f32>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Tensor<D3, f32> {
    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let l = input_shape[2];

    // Calculate output length
    let l_out = (l + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // 1. Padding
    let padded = if padding > 0 {
        input.pad(&[(0, 0), (0, 0), (padding, padding)])
    } else {
        input.clone()
    };

    // 2. unfold_1d: D3 -> D4 [N, C, L_out, K]
    let unfolded: Tensor<D4, f32> = padded.unfold_1d(kernel_size, stride, dilation);

    // 3. max over kernel dimension (axis 3): [N, C, L_out, K] -> [N, C, L_out]
    let output: Tensor<D3, f32> = unfolded.max(3);

    debug_assert_eq!(output.shape(), vec![batch, channels, l_out]);
    output
}

/// Applies a 1D average pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, L]
/// * `kernel_size` - Size of the window
/// * `stride` - Stride of the window. Default: kernel_size
/// * `padding` - Padding added to both sides
///
/// # Returns
/// Output tensor of shape [N, C, L_out]
pub fn avg_pool1d(
    input: &Tensor<D3, f32>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Tensor<D3, f32> {
    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let l = input_shape[2];

    // Calculate output length (dilation = 1 for avg pooling)
    let l_out = (l + 2 * padding - kernel_size) / stride + 1;

    // 1. Padding
    let padded = if padding > 0 {
        input.pad(&[(0, 0), (0, 0), (padding, padding)])
    } else {
        input.clone()
    };

    // 2. unfold_1d: D3 -> D4 [N, C, L_out, K]
    let unfolded: Tensor<D4, f32> = padded.unfold_1d(kernel_size, stride, 1);

    // 3. mean over kernel dimension (axis 3): [N, C, L_out, K] -> [N, C, L_out]
    let output: Tensor<D3, f32> = unfolded.mean(3);

    debug_assert_eq!(output.shape(), vec![batch, channels, l_out]);
    output
}

// ============================================================================
// MaxPool2d
// ============================================================================

/// Applies a 2D max pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `kernel_size` - Size of the window (kH, kW)
/// * `stride` - Stride of the window (sH, sW). Default: kernel_size
/// * `padding` - Padding added to all sides (pH, pW)
/// * `dilation` - Spacing between kernel elements (dH, dW)
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
///
/// Where:
/// - `H_out = (H + 2*pH - dH*(kH-1) - 1) / sH + 1`
/// - `W_out = (W + 2*pW - dW*(kW-1) - 1) / sW + 1`
pub fn max_pool2d(
    input: &Tensor<D4, f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor<D4, f32> {
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    // Calculate output size
    let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    // 1. Padding
    let padded = if ph > 0 || pw > 0 {
        input.pad(&[(0, 0), (0, 0), (ph, ph), (pw, pw)])
    } else {
        input.clone()
    };

    // 2. unfold_2d: D4 -> D6 [N, C, H_out, W_out, kH, kW]
    let unfolded = padded.unfold_2d((kh, kw), (sh, sw), (dh, dw));

    // 3. contiguous + reshape to merge kernel dims: [N, C, H_out, W_out, kH*kW]
    let reshaped: Tensor<D5, f32> =
        unfolded
            .contiguous()
            .reshape([batch, channels, h_out, w_out, kh * kw]);

    // 4. max over kernel dimension (axis 4): [N, C, H_out, W_out]
    let output: Tensor<D4, f32> = reshaped.max(4);

    debug_assert_eq!(output.shape(), vec![batch, channels, h_out, w_out]);
    output
}

/// Applies a 2D average pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `kernel_size` - Size of the window (kH, kW)
/// * `stride` - Stride of the window (sH, sW). Default: kernel_size
/// * `padding` - Padding added to all sides (pH, pW)
///
/// # Returns
/// Output tensor of shape [N, C, H_out, W_out]
pub fn avg_pool2d(
    input: &Tensor<D4, f32>,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor<D4, f32> {
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    // Calculate output size (dilation = 1)
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;

    // 1. Padding
    let padded = if ph > 0 || pw > 0 {
        input.pad(&[(0, 0), (0, 0), (ph, ph), (pw, pw)])
    } else {
        input.clone()
    };

    // 2. unfold_2d: D4 -> D6 [N, C, H_out, W_out, kH, kW]
    let unfolded = padded.unfold_2d((kh, kw), (sh, sw), (1, 1));

    // 3. contiguous + reshape to merge kernel dims: [N, C, H_out, W_out, kH*kW]
    let reshaped: Tensor<D5, f32> =
        unfolded
            .contiguous()
            .reshape([batch, channels, h_out, w_out, kh * kw]);

    // 4. mean over kernel dimension (axis 4): [N, C, H_out, W_out]
    let output: Tensor<D4, f32> = reshaped.mean(4);

    debug_assert_eq!(output.shape(), vec![batch, channels, h_out, w_out]);
    output
}

// ============================================================================
// MaxPool3d
// ============================================================================

/// Applies a 3D max pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `kernel_size` - Size of the window (kD, kH, kW)
/// * `stride` - Stride of the window (sD, sH, sW). Default: kernel_size
/// * `padding` - Padding added to all sides (pD, pH, pW)
/// * `dilation` - Spacing between kernel elements (dD, dH, dW)
///
/// # Returns
/// Output tensor of shape [N, C, D_out, H_out, W_out]
pub fn max_pool3d(
    input: &Tensor<D5, f32>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
) -> Tensor<D5, f32> {
    let (kd, kh, kw) = kernel_size;
    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;
    let (dd, dh, dw) = dilation;

    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let d = input_shape[2];
    let h = input_shape[3];
    let w = input_shape[4];

    // Calculate output size
    let d_out = (d + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
    let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    // 1. Padding
    let padded = if pd > 0 || ph > 0 || pw > 0 {
        input.pad(&[(0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)])
    } else {
        input.clone()
    };

    // 2. unfold_3d: D5 -> D8 [N, C, D_out, H_out, W_out, kD, kH, kW]
    let unfolded = padded.unfold_3d((kd, kh, kw), (sd, sh, sw), (dd, dh, dw));

    // 3. contiguous + reshape to merge kernel dims: [N, C, D_out, H_out, W_out, kD*kH*kW]
    use eclat::tensor::dim::D6;
    let reshaped: Tensor<D6, f32> =
        unfolded
            .contiguous()
            .reshape([batch, channels, d_out, h_out, w_out, kd * kh * kw]);

    // 4. max over kernel dimension (axis 5): [N, C, D_out, H_out, W_out]
    let output: Tensor<D5, f32> = reshaped.max(5);

    debug_assert_eq!(output.shape(), vec![batch, channels, d_out, h_out, w_out]);
    output
}

/// Applies a 3D average pooling over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `kernel_size` - Size of the window (kD, kH, kW)
/// * `stride` - Stride of the window (sD, sH, sW). Default: kernel_size
/// * `padding` - Padding added to all sides (pD, pH, pW)
///
/// # Returns
/// Output tensor of shape [N, C, D_out, H_out, W_out]
pub fn avg_pool3d(
    input: &Tensor<D5, f32>,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> Tensor<D5, f32> {
    let (kd, kh, kw) = kernel_size;
    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;

    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let d = input_shape[2];
    let h = input_shape[3];
    let w = input_shape[4];

    // Calculate output size (dilation = 1)
    let d_out = (d + 2 * pd - kd) / sd + 1;
    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;

    // 1. Padding
    let padded = if pd > 0 || ph > 0 || pw > 0 {
        input.pad(&[(0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)])
    } else {
        input.clone()
    };

    // 2. unfold_3d: D5 -> D8 [N, C, D_out, H_out, W_out, kD, kH, kW]
    let unfolded = padded.unfold_3d((kd, kh, kw), (sd, sh, sw), (1, 1, 1));

    // 3. contiguous + reshape to merge kernel dims: [N, C, D_out, H_out, W_out, kD*kH*kW]
    use eclat::tensor::dim::D6;
    let reshaped: Tensor<D6, f32> =
        unfolded
            .contiguous()
            .reshape([batch, channels, d_out, h_out, w_out, kd * kh * kw]);

    // 4. mean over kernel dimension (axis 5): [N, C, D_out, H_out, W_out]
    let output: Tensor<D5, f32> = reshaped.mean(5);

    debug_assert_eq!(output.shape(), vec![batch, channels, d_out, h_out, w_out]);
    output
}

// ============================================================================
// Adaptive Pooling
// ============================================================================

/// Applies adaptive average pooling over an input signal.
///
/// The output spatial size is always (1, 1) regardless of input size.
/// This is commonly used before the final fully-connected layer.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
///
/// # Returns
/// Output tensor of shape [N, C, 1, 1]
pub fn adaptive_avg_pool2d(input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    // Global average pooling: mean over spatial dimensions
    // Reshape [N, C, H, W] -> [N, C, H*W], then mean over last axis
    use eclat::tensor::dim::D2;
    let flattened: Tensor<D3, f32> = input.reshape([batch, channels, h * w]);
    let mean_spatial: Tensor<D2, f32> = flattened.mean(2);

    // Reshape to [N, C, 1, 1]
    mean_spatial.reshape([batch, channels, 1, 1])
}

/// Applies adaptive max pooling over an input signal.
///
/// The output spatial size is always (1, 1) regardless of input size.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
///
/// # Returns
/// Output tensor of shape [N, C, 1, 1]
pub fn adaptive_max_pool2d(input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
    let input_shape = input.shape();
    let batch = input_shape[0];
    let channels = input_shape[1];
    let h = input_shape[2];
    let w = input_shape[3];

    // Global max pooling: max over spatial dimensions
    // Reshape [N, C, H, W] -> [N, C, H*W], then max over last axis
    use eclat::tensor::dim::D2;
    let flattened: Tensor<D3, f32> = input.reshape([batch, channels, h * w]);
    let max_spatial: Tensor<D2, f32> = flattened.max(2);

    // Reshape to [N, C, 1, 1]
    max_spatial.reshape([batch, channels, 1, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool1d_shape() {
        // Input: [2, 3, 10], kernel: 3, stride: 2, padding: 0
        // Output: [2, 3, 4]  ((10 - 3) / 2 + 1 = 4)
        let input: Tensor<D3, f32> = Tensor::input([2, 3, 10]);
        let output = max_pool1d(&input, 3, 2, 0, 1);
        assert_eq!(output.shape(), vec![2, 3, 4]);
    }

    #[test]
    fn test_avg_pool1d_shape() {
        // Input: [2, 3, 10], kernel: 2, stride: 2, padding: 0
        // Output: [2, 3, 5]  ((10 - 2) / 2 + 1 = 5)
        let input: Tensor<D3, f32> = Tensor::input([2, 3, 10]);
        let output = avg_pool1d(&input, 2, 2, 0);
        assert_eq!(output.shape(), vec![2, 3, 5]);
    }

    #[test]
    fn test_max_pool2d_shape() {
        // Input: [1, 64, 32, 32], kernel: 2x2, stride: 2x2, padding: 0
        // Output: [1, 64, 16, 16]
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = max_pool2d(&input, (2, 2), (2, 2), (0, 0), (1, 1));
        assert_eq!(output.shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_max_pool2d_with_padding() {
        // Input: [1, 64, 32, 32], kernel: 3x3, stride: 2x2, padding: 1
        // Output: [1, 64, 16, 16]
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = max_pool2d(&input, (3, 3), (2, 2), (1, 1), (1, 1));
        assert_eq!(output.shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_avg_pool2d_shape() {
        // Input: [1, 64, 32, 32], kernel: 2x2, stride: 2x2, padding: 0
        // Output: [1, 64, 16, 16]
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = avg_pool2d(&input, (2, 2), (2, 2), (0, 0));
        assert_eq!(output.shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_max_pool3d_shape() {
        // Input: [1, 64, 16, 32, 32], kernel: 2x2x2, stride: 2x2x2, padding: 0
        // Output: [1, 64, 8, 16, 16]
        let input: Tensor<D5, f32> = Tensor::input([1, 64, 16, 32, 32]);
        let output = max_pool3d(&input, (2, 2, 2), (2, 2, 2), (0, 0, 0), (1, 1, 1));
        assert_eq!(output.shape(), vec![1, 64, 8, 16, 16]);
    }

    #[test]
    fn test_avg_pool3d_shape() {
        // Input: [1, 64, 16, 32, 32], kernel: 2x2x2, stride: 2x2x2, padding: 0
        // Output: [1, 64, 8, 16, 16]
        let input: Tensor<D5, f32> = Tensor::input([1, 64, 16, 32, 32]);
        let output = avg_pool3d(&input, (2, 2, 2), (2, 2, 2), (0, 0, 0));
        assert_eq!(output.shape(), vec![1, 64, 8, 16, 16]);
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        // Input: [1, 64, 32, 32]
        // Output: [1, 64, 1, 1]
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = adaptive_avg_pool2d(&input);
        assert_eq!(output.shape(), vec![1, 64, 1, 1]);
    }

    #[test]
    fn test_adaptive_max_pool2d() {
        // Input: [1, 64, 32, 32]
        // Output: [1, 64, 1, 1]
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 32, 32]);
        let output = adaptive_max_pool2d(&input);
        assert_eq!(output.shape(), vec![1, 64, 1, 1]);
    }
}

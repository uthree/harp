//! Convolution operations
//!
//! Implements convolution operations using unfold (im2col) approach.

use eclat::tensor::Tensor;
use eclat::tensor::dim::{D1, D2, D3, D4, D5, D6, D8};

// ============================================================================
// Conv1d
// ============================================================================

/// Applies a 1D convolution over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C_in, L]
/// * `weight` - Weight tensor of shape [C_out, C_in, K]
/// * `bias` - Optional bias tensor of shape [C_out]
/// * `stride` - Stride of the convolution
/// * `padding` - Padding added to both sides
/// * `dilation` - Spacing between kernel elements
///
/// # Returns
/// Output tensor of shape [N, C_out, L_out]
///
/// Where: `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
pub fn conv1d(
    input: &Tensor<D3, f32>,
    weight: &Tensor<D3, f32>,
    bias: Option<&Tensor<D1, f32>>,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Tensor<D3, f32> {
    let weight_shape = weight.shape();
    let out_channels = weight_shape[0];
    let in_channels = weight_shape[1];
    let k = weight_shape[2];

    // Get input shape
    let input_shape = input.shape();
    let batch = input_shape[0];
    let l = input_shape[2];

    // Calculate output length
    let l_out = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;

    // 1. Padding: D3 -> D3
    let padded = input.pad(&[(0, 0), (0, 0), (padding, padding)]);

    // 2. unfold_1d: D3 -> D4 [N, C_in, L_out, K]
    let unfolded: Tensor<D4, f32> = padded.unfold_1d(k, stride, dilation);

    // 3. permute: [N, C_in, L_out, K] -> [N, L_out, C_in, K]
    let permuted: Tensor<D4, f32> = unfolded.permute(&[0, 2, 1, 3]);

    // 4. contiguous + reshape: D4 -> D2 [N*L_out, C_in*K]
    let cols: Tensor<D2, f32> = permuted
        .contiguous()
        .reshape([batch * l_out, in_channels * k]);

    // 5. Get weight and reshape: [C_out, C_in, K] -> [C_out, C_in*K]
    let weight_flat: Tensor<D2, f32> = weight.reshape([out_channels, in_channels * k]);

    // 6. Matrix multiplication using broadcast multiply + sum
    // cols: [N*L_out, C_in*K] -> [N*L_out, 1, C_in*K]
    // weight: [C_out, C_in*K] -> [1, C_out, C_in*K]
    let cols_expanded: Tensor<D3, f32> = cols.unsqueeze(1);
    let weight_expanded: Tensor<D3, f32> = weight_flat.unsqueeze(0);

    // broadcast multiply: [N*L_out, C_out, C_in*K]
    let product: Tensor<D3, f32> = &cols_expanded * &weight_expanded;

    // sum over last axis: [N*L_out, C_out]
    let result: Tensor<D2, f32> = product.sum(2);

    // 7. reshape: [N*L_out, C_out] -> [N, L_out, C_out]
    let reshaped: Tensor<D3, f32> = result.reshape([batch, l_out, out_channels]);

    // 8. permute: [N, L_out, C_out] -> [N, C_out, L_out]
    let output: Tensor<D3, f32> = reshaped.permute(&[0, 2, 1]);

    // 9. Add bias if present
    match bias {
        Some(bias_tensor) => {
            // bias: [C_out] -> [1, C_out, 1]
            let bias_expanded: Tensor<D3, f32> = bias_tensor.unsqueeze(0).unsqueeze(2);
            &output + &bias_expanded
        }
        None => output,
    }
}

// ============================================================================
// Conv2d
// ============================================================================

/// Applies a 2D convolution over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C_in, H, W]
/// * `weight` - Weight tensor of shape [C_out, C_in, kH, kW]
/// * `bias` - Optional bias tensor of shape [C_out]
/// * `stride` - Stride of the convolution (sH, sW)
/// * `padding` - Padding added to both sides (pH, pW)
/// * `dilation` - Spacing between kernel elements (dH, dW)
///
/// # Returns
/// Output tensor of shape [N, C_out, H_out, W_out]
///
/// Where:
/// - `H_out = (H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
/// - `W_out = (W + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
pub fn conv2d(
    input: &Tensor<D4, f32>,
    weight: &Tensor<D4, f32>,
    bias: Option<&Tensor<D1, f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor<D4, f32> {
    let weight_shape = weight.shape();
    let out_channels = weight_shape[0];
    let in_channels = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (dh, dw) = dilation;

    // Get input shape
    let input_shape = input.shape();
    let batch = input_shape[0];
    let h = input_shape[2];
    let w = input_shape[3];

    // Calculate output spatial dimensions
    let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    // 1. Padding: D4 -> D4
    let padded = input.pad(&[(0, 0), (0, 0), (ph, ph), (pw, pw)]);

    // 2. unfold_2d: D4 -> D6 [N, C_in, H_out, W_out, kH, kW]
    let unfolded: Tensor<D6, f32> = padded.unfold_2d((kh, kw), (sh, sw), (dh, dw));

    // 3. permute: [N, C_in, H_out, W_out, kH, kW] -> [N, H_out, W_out, C_in, kH, kW]
    let permuted: Tensor<D6, f32> = unfolded.permute(&[0, 2, 3, 1, 4, 5]);

    // 4. contiguous + reshape: D6 -> D3 [N, H_out*W_out, C_in*kH*kW]
    let cols: Tensor<D3, f32> =
        permuted
            .contiguous()
            .reshape([batch, h_out * w_out, in_channels * kh * kw]);

    // 5. Get weight and reshape: [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
    let weight_flat: Tensor<D2, f32> = weight.reshape([out_channels, in_channels * kh * kw]);

    // 6. Matrix multiplication using broadcast multiply + sum
    // cols: [N, H_out*W_out, C_in*kH*kW] -> [N, H_out*W_out, 1, C_in*kH*kW]
    // weight: [C_out, C_in*kH*kW] -> [1, 1, C_out, C_in*kH*kW]
    let cols_expanded: Tensor<D4, f32> = cols.unsqueeze(2);
    let weight_expanded: Tensor<D4, f32> = weight_flat.unsqueeze(0).unsqueeze(0);

    // broadcast multiply: [N, H_out*W_out, C_out, C_in*kH*kW]
    let product: Tensor<D4, f32> = &cols_expanded * &weight_expanded;

    // sum over last axis: [N, H_out*W_out, C_out]
    let result: Tensor<D3, f32> = product.sum(3);

    // 7. reshape + permute: [N, H_out*W_out, C_out] -> [N, H_out, W_out, C_out] -> [N, C_out, H_out, W_out]
    let reshaped: Tensor<D4, f32> = result.reshape([batch, h_out, w_out, out_channels]);
    let output: Tensor<D4, f32> = reshaped.permute(&[0, 3, 1, 2]);

    // 8. Add bias if present
    match bias {
        Some(bias_tensor) => {
            // bias: [C_out] -> [1, C_out, 1, 1]
            let bias_expanded: Tensor<D4, f32> = bias_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3);
            &output + &bias_expanded
        }
        None => output,
    }
}

// ============================================================================
// Conv3d
// ============================================================================

/// Applies a 3D convolution over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C_in, D, H, W]
/// * `weight` - Weight tensor of shape [C_out, C_in, kD, kH, kW]
/// * `bias` - Optional bias tensor of shape [C_out]
/// * `stride` - Stride of the convolution (sD, sH, sW)
/// * `padding` - Padding added to both sides (pD, pH, pW)
/// * `dilation` - Spacing between kernel elements (dD, dH, dW)
///
/// # Returns
/// Output tensor of shape [N, C_out, D_out, H_out, W_out]
pub fn conv3d(
    input: &Tensor<D5, f32>,
    weight: &Tensor<D5, f32>,
    bias: Option<&Tensor<D1, f32>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
) -> Tensor<D5, f32> {
    let weight_shape = weight.shape();
    let out_channels = weight_shape[0];
    let in_channels = weight_shape[1];
    let kd = weight_shape[2];
    let kh = weight_shape[3];
    let kw = weight_shape[4];

    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;
    let (dd, dh, dw) = dilation;

    // Get input shape
    let input_shape = input.shape();
    let batch = input_shape[0];
    let d = input_shape[2];
    let h = input_shape[3];
    let w = input_shape[4];

    // Calculate output size
    let d_out = (d + 2 * pd - dd * (kd - 1) - 1) / sd + 1;
    let h_out = (h + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
    let w_out = (w + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

    // 1. Padding: D5 -> D5
    let padded = input.pad(&[(0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)]);

    // 2. unfold_3d: D5 -> D8 [N, C_in, D_out, H_out, W_out, kD, kH, kW]
    let unfolded: Tensor<D8, f32> = padded.unfold_3d((kd, kh, kw), (sd, sh, sw), (dd, dh, dw));

    // 3. permute: [N, C_in, D_out, H_out, W_out, kD, kH, kW] -> [N, D_out, H_out, W_out, C_in, kD, kH, kW]
    let permuted: Tensor<D8, f32> = unfolded.permute(&[0, 2, 3, 4, 1, 5, 6, 7]);

    // 4. contiguous + reshape: D8 -> D3 [N, D_out*H_out*W_out, C_in*kD*kH*kW]
    let cols: Tensor<D3, f32> =
        permuted
            .contiguous()
            .reshape([batch, d_out * h_out * w_out, in_channels * kd * kh * kw]);

    // 5. Get weight and reshape: [C_out, C_in, kD, kH, kW] -> [C_out, C_in*kD*kH*kW]
    let weight_flat: Tensor<D2, f32> = weight.reshape([out_channels, in_channels * kd * kh * kw]);

    // 6. Matrix multiplication using broadcast multiply + sum
    let cols_expanded: Tensor<D4, f32> = cols.unsqueeze(2);
    let weight_expanded: Tensor<D4, f32> = weight_flat.unsqueeze(0).unsqueeze(0);

    // broadcast multiply: [N, D_out*H_out*W_out, C_out, C_in*kD*kH*kW]
    let product: Tensor<D4, f32> = &cols_expanded * &weight_expanded;

    // sum over last axis: [N, D_out*H_out*W_out, C_out]
    let result: Tensor<D3, f32> = product.sum(3);

    // 7. reshape + permute
    let reshaped: Tensor<D5, f32> = result.reshape([batch, d_out, h_out, w_out, out_channels]);
    let output: Tensor<D5, f32> = reshaped.permute(&[0, 4, 1, 2, 3]);

    // 8. Add bias if present
    match bias {
        Some(bias_tensor) => {
            let bias_expanded: Tensor<D5, f32> = bias_tensor
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4);
            &output + &bias_expanded
        }
        None => output,
    }
}

// ============================================================================
// ConvTranspose1d
// ============================================================================

/// Applies a 1D transposed convolution over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C_in, L]
/// * `weight` - Weight tensor of shape [C_in, C_out, K]
/// * `bias` - Optional bias tensor of shape [C_out]
/// * `stride` - Stride of the convolution
/// * `padding` - Padding to remove from output
/// * `output_padding` - Additional size added to one side of the output
/// * `dilation` - Spacing between kernel elements
///
/// # Returns
/// Output tensor of shape [N, C_out, L_out]
///
/// Where: `L_out = (L - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
pub fn conv_transpose1d(
    input: &Tensor<D3, f32>,
    weight: &Tensor<D3, f32>,
    bias: Option<&Tensor<D1, f32>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
) -> Tensor<D3, f32> {
    let weight_shape = weight.shape();
    let in_channels = weight_shape[0];
    let out_channels = weight_shape[1];
    let k = weight_shape[2];

    // Get input shape
    let input_shape = input.shape();
    let batch = input_shape[0];
    let l_in = input_shape[2];

    // Calculate output length
    let l_out = (l_in - 1) * stride + dilation * (k - 1) + 1 + output_padding - 2 * padding;

    // 1. Reshape input: [N, C_in, L] -> [N, L, C_in]
    let input_reshaped: Tensor<D3, f32> = input.permute(&[0, 2, 1]);

    // 2. Get weight and reshape: [C_in, C_out, K] -> [C_in, C_out*K]
    let weight_flat: Tensor<D2, f32> = weight.reshape([in_channels, out_channels * k]);

    // 3. Matrix multiplication using broadcast multiply + sum
    // input: [N, L, C_in] -> [N, L, C_in, 1]
    // weight: [C_in, C_out*K] -> [1, 1, C_in, C_out*K]
    let input_expanded: Tensor<D4, f32> = input_reshaped.unsqueeze(3);
    let weight_expanded: Tensor<D4, f32> = weight_flat.unsqueeze(0).unsqueeze(0);

    // broadcast multiply: [N, L, C_in, C_out*K]
    let product: Tensor<D4, f32> = &input_expanded * &weight_expanded;

    // sum over C_in axis: [N, L, C_out*K]
    let cols: Tensor<D3, f32> = product.sum(2);

    // 4. Reshape: [N, L, C_out*K] -> [N, L, C_out, K]
    let cols_reshaped: Tensor<D4, f32> = cols.reshape([batch, l_in, out_channels, k]);

    // 5. Permute: [N, L, C_out, K] -> [N, C_out, L, K]
    let cols_permuted: Tensor<D4, f32> = cols_reshaped.permute(&[0, 2, 1, 3]);

    // 6. Use fold to scatter-add back to output shape
    let output_shape = &[batch, out_channels, l_out];
    let folded = cols_permuted.contiguous().into_dyn().fold(
        output_shape,
        &[2],        // axis that was unfolded
        &[k],        // kernel size
        &[stride],   // stride
        &[dilation], // dilation
    );

    // 7. Reshape to D3
    let output: Tensor<D3, f32> = folded.reshape([batch, out_channels, l_out]);

    // 8. Add bias if present
    match bias {
        Some(bias_tensor) => {
            // bias: [C_out] -> [1, C_out, 1]
            let bias_expanded: Tensor<D3, f32> = bias_tensor.unsqueeze(0).unsqueeze(2);
            &output + &bias_expanded
        }
        None => output,
    }
}

// ============================================================================
// ConvTranspose2d
// ============================================================================

/// Applies a 2D transposed convolution over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C_in, H, W]
/// * `weight` - Weight tensor of shape [C_in, C_out, kH, kW]
/// * `bias` - Optional bias tensor of shape [C_out]
/// * `stride` - Stride of the convolution (sH, sW)
/// * `padding` - Padding to remove from output (pH, pW)
/// * `output_padding` - Additional size added to one side (opH, opW)
/// * `dilation` - Spacing between kernel elements (dH, dW)
///
/// # Returns
/// Output tensor of shape [N, C_out, H_out, W_out]
pub fn conv_transpose2d(
    input: &Tensor<D4, f32>,
    weight: &Tensor<D4, f32>,
    bias: Option<&Tensor<D1, f32>>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
) -> Tensor<D4, f32> {
    let weight_shape = weight.shape();
    let in_channels = weight_shape[0];
    let out_channels = weight_shape[1];
    let kh = weight_shape[2];
    let kw = weight_shape[3];

    let (sh, sw) = stride;
    let (ph, pw) = padding;
    let (oph, opw) = output_padding;
    let (dh, dw) = dilation;

    // Get input shape
    let input_shape = input.shape();
    let batch = input_shape[0];
    let h_in = input_shape[2];
    let w_in = input_shape[3];

    // Calculate output spatial dimensions
    let h_out = (h_in - 1) * sh + dh * (kh - 1) + 1 + oph - 2 * ph;
    let w_out = (w_in - 1) * sw + dw * (kw - 1) + 1 + opw - 2 * pw;

    // 1. Reshape input: [N, C_in, H, W] -> [N, H*W, C_in]
    let input_reshaped: Tensor<D3, f32> =
        input
            .permute(&[0, 2, 3, 1])
            .contiguous()
            .reshape([batch, h_in * w_in, in_channels]);

    // 2. Get weight and reshape: [C_in, C_out, kH, kW] -> [C_in, C_out*kH*kW]
    let weight_flat: Tensor<D2, f32> = weight.reshape([in_channels, out_channels * kh * kw]);

    // 3. Matrix multiplication: [N, H*W, C_in] x [C_in, C_out*kH*kW] -> [N, H*W, C_out*kH*kW]
    let input_expanded: Tensor<D4, f32> = input_reshaped.unsqueeze(3);
    let weight_expanded: Tensor<D4, f32> = weight_flat.unsqueeze(0).unsqueeze(0);

    // broadcast multiply: [N, H*W, C_in, C_out*kH*kW]
    let product: Tensor<D4, f32> = &input_expanded * &weight_expanded;

    // sum over C_in axis: [N, H*W, C_out*kH*kW]
    let cols: Tensor<D3, f32> = product.sum(2);

    // 4. Reshape: [N, H*W, C_out*kH*kW] -> [N, H, W, C_out, kH, kW]
    let cols_reshaped: Tensor<D6, f32> = cols.reshape([batch, h_in, w_in, out_channels, kh, kw]);

    // 5. Permute: [N, H, W, C_out, kH, kW] -> [N, C_out, H, W, kH, kW]
    let cols_permuted: Tensor<D6, f32> = cols_reshaped.permute(&[0, 3, 1, 2, 4, 5]);

    // 6. Use fold to scatter-add back to output shape
    let output_shape = &[batch, out_channels, h_out, w_out];
    let folded = cols_permuted.contiguous().into_dyn().fold(
        output_shape,
        &[2, 3],   // axes that were unfolded
        &[kh, kw], // kernel sizes
        &[sh, sw], // strides
        &[dh, dw], // dilations
    );

    // 7. Reshape to D4
    let output: Tensor<D4, f32> = folded.reshape([batch, out_channels, h_out, w_out]);

    // 8. Add bias if present
    match bias {
        Some(bias_tensor) => {
            let bias_expanded: Tensor<D4, f32> = bias_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3);
            &output + &bias_expanded
        }
        None => output,
    }
}

// ============================================================================
// ConvTranspose3d
// ============================================================================

/// Applies a 3D transposed convolution over an input signal.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C_in, D, H, W]
/// * `weight` - Weight tensor of shape [C_in, C_out, kD, kH, kW]
/// * `bias` - Optional bias tensor of shape [C_out]
/// * `stride` - Stride of the convolution (sD, sH, sW)
/// * `padding` - Padding to remove from output (pD, pH, pW)
/// * `output_padding` - Additional size added to one side (opD, opH, opW)
/// * `dilation` - Spacing between kernel elements (dD, dH, dW)
///
/// # Returns
/// Output tensor of shape [N, C_out, D_out, H_out, W_out]
pub fn conv_transpose3d(
    input: &Tensor<D5, f32>,
    weight: &Tensor<D5, f32>,
    bias: Option<&Tensor<D1, f32>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
) -> Tensor<D5, f32> {
    let weight_shape = weight.shape();
    let in_channels = weight_shape[0];
    let out_channels = weight_shape[1];
    let kd = weight_shape[2];
    let kh = weight_shape[3];
    let kw = weight_shape[4];

    let (sd, sh, sw) = stride;
    let (pd, ph, pw) = padding;
    let (opd, oph, opw) = output_padding;
    let (dd, dh, dw) = dilation;

    // Get input shape
    let input_shape = input.shape();
    let batch = input_shape[0];
    let d_in = input_shape[2];
    let h_in = input_shape[3];
    let w_in = input_shape[4];

    // Calculate output size
    let d_out = (d_in - 1) * sd + dd * (kd - 1) + 1 + opd - 2 * pd;
    let h_out = (h_in - 1) * sh + dh * (kh - 1) + 1 + oph - 2 * ph;
    let w_out = (w_in - 1) * sw + dw * (kw - 1) + 1 + opw - 2 * pw;

    // 1. Reshape input: [N, C_in, D, H, W] -> [N, D*H*W, C_in]
    let input_reshaped: Tensor<D3, f32> = input.permute(&[0, 2, 3, 4, 1]).contiguous().reshape([
        batch,
        d_in * h_in * w_in,
        in_channels,
    ]);

    // 2. Get weight and reshape: [C_in, C_out, kD, kH, kW] -> [C_in, C_out*kD*kH*kW]
    let weight_flat: Tensor<D2, f32> = weight.reshape([in_channels, out_channels * kd * kh * kw]);

    // 3. Matrix multiplication using broadcast multiply + sum
    let input_expanded: Tensor<D4, f32> = input_reshaped.unsqueeze(3);
    let weight_expanded: Tensor<D4, f32> = weight_flat.unsqueeze(0).unsqueeze(0);

    // broadcast multiply: [N, D*H*W, C_in, C_out*kD*kH*kW]
    let product: Tensor<D4, f32> = &input_expanded * &weight_expanded;

    // sum over C_in axis: [N, D*H*W, C_out*kD*kH*kW]
    let cols: Tensor<D3, f32> = product.sum(2);

    // 4. Reshape: [N, D*H*W, C_out*kD*kH*kW] -> [N, D, H, W, C_out, kD, kH, kW]
    let cols_reshaped: Tensor<D8, f32> =
        cols.reshape([batch, d_in, h_in, w_in, out_channels, kd, kh, kw]);

    // 5. Permute: [N, D, H, W, C_out, kD, kH, kW] -> [N, C_out, D, H, W, kD, kH, kW]
    let cols_permuted: Tensor<D8, f32> = cols_reshaped.permute(&[0, 4, 1, 2, 3, 5, 6, 7]);

    // 6. Use fold to scatter-add back to output shape
    let output_shape = &[batch, out_channels, d_out, h_out, w_out];
    let folded = cols_permuted.contiguous().into_dyn().fold(
        output_shape,
        &[2, 3, 4],    // axes that were unfolded
        &[kd, kh, kw], // kernel sizes
        &[sd, sh, sw], // strides
        &[dd, dh, dw], // dilations
    );

    // 7. Reshape to D5
    let output: Tensor<D5, f32> = folded.reshape([batch, out_channels, d_out, h_out, w_out]);

    // 8. Add bias if present
    match bias {
        Some(bias_tensor) => {
            let bias_expanded: Tensor<D5, f32> = bias_tensor
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4);
            &output + &bias_expanded
        }
        None => output,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv1d() {
        let input: Tensor<D3, f32> = Tensor::input([1, 64, 100]);
        let weight: Tensor<D3, f32> = Tensor::input([128, 64, 5]);
        let output = conv1d(&input, &weight, None, 1, 0, 1);
        assert_eq!(output.shape(), vec![1, 128, 96]);
    }

    #[test]
    fn test_conv2d() {
        let input: Tensor<D4, f32> = Tensor::input([1, 3, 32, 32]);
        let weight: Tensor<D4, f32> = Tensor::input([64, 3, 3, 3]);
        let output = conv2d(&input, &weight, None, (1, 1), (0, 0), (1, 1));
        assert_eq!(output.shape(), vec![1, 64, 30, 30]);
    }

    #[test]
    fn test_conv3d() {
        let input: Tensor<D5, f32> = Tensor::input([1, 3, 16, 32, 32]);
        let weight: Tensor<D5, f32> = Tensor::input([64, 3, 3, 3, 3]);
        let output = conv3d(&input, &weight, None, (1, 1, 1), (0, 0, 0), (1, 1, 1));
        assert_eq!(output.shape(), vec![1, 64, 14, 30, 30]);
    }

    #[test]
    fn test_conv_transpose1d() {
        let input: Tensor<D3, f32> = Tensor::input([1, 128, 96]);
        let weight: Tensor<D3, f32> = Tensor::input([128, 64, 5]);
        let output = conv_transpose1d(&input, &weight, None, 1, 0, 0, 1);
        assert_eq!(output.shape(), vec![1, 64, 100]);
    }

    #[test]
    fn test_conv_transpose2d() {
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 30, 30]);
        let weight: Tensor<D4, f32> = Tensor::input([64, 3, 3, 3]);
        let output = conv_transpose2d(&input, &weight, None, (1, 1), (0, 0), (0, 0), (1, 1));
        assert_eq!(output.shape(), vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_conv_transpose3d() {
        let input: Tensor<D5, f32> = Tensor::input([1, 64, 14, 30, 30]);
        let weight: Tensor<D5, f32> = Tensor::input([64, 3, 3, 3, 3]);
        let output = conv_transpose3d(
            &input,
            &weight,
            None,
            (1, 1, 1),
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
        );
        assert_eq!(output.shape(), vec![1, 3, 16, 32, 32]);
    }
}

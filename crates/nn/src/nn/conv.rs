//! Convolution layers
//!
//! Implements Conv1d, Conv2d, and Conv3d using unfold (im2col) approach.

use super::{Module, Parameter, ParameterBase};
use eclat::tensor::Tensor;
use eclat::tensor::dim::{D1, D2, D3, D4, D5, D6, D8};

// ============================================================================
// Conv2d
// ============================================================================

/// 2D Convolution layer.
///
/// Applies a 2D convolution over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `[N, C_in, H, W]`
/// - Output: `[N, C_out, H_out, W_out]`
///
/// Where:
/// - `H_out = (H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
/// - `W_out = (W + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::Conv2d;
/// use eclat::tensor::{Tensor, dim::D4};
///
/// let conv = Conv2d::new(3, 64, (3, 3));
/// let input: Tensor<D4, f32> = Tensor::input([1, 3, 32, 32]);
/// let output = conv.forward_d4(&input);  // [1, 64, 30, 30]
/// ```
pub struct Conv2d {
    /// Weight parameter [out_channels, in_channels, kH, kW]
    weight: Parameter<D4>,
    /// Bias parameter [out_channels] (optional)
    bias: Option<Parameter<D1>>,
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size (kH, kW)
    kernel_size: (usize, usize),
    /// Stride (sH, sW)
    stride: (usize, usize),
    /// Padding (pH, pW)
    padding: (usize, usize),
    /// Dilation (dH, dW)
    dilation: (usize, usize),
    /// Training mode flag
    training: bool,
}

impl Conv2d {
    /// Create a new Conv2d layer with default stride=1, padding=0, dilation=1, bias=true.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            kernel_size,
            (1, 1),
            (0, 0),
            (1, 1),
            true,
        )
    }

    /// Create a new Conv2d layer with all options.
    ///
    /// # Arguments
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Size of the convolving kernel (kH, kW)
    /// * `stride` - Stride of the convolution (sH, sW)
    /// * `padding` - Zero-padding added to both sides (pH, pW)
    /// * `dilation` - Spacing between kernel elements (dH, dW)
    /// * `bias` - If true, adds a learnable bias
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
    ) -> Self {
        let (kh, kw) = kernel_size;

        // Kaiming uniform initialization
        let fan_in = in_channels * kh * kw;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Initialize weight
        let weight_size = out_channels * in_channels * kh * kw;
        let weight_data: Vec<f32> = (0..weight_size)
            .map(|i| (i as f32 * 0.1).sin() * bound)
            .collect();
        let weight: Parameter<D4> =
            Parameter::from_data("weight", &weight_data, &[out_channels, in_channels, kh, kw]);

        // Initialize bias
        let bias = if bias {
            let bias_data = vec![0.0f32; out_channels];
            Some(Parameter::from_data("bias", &bias_data, &[out_channels]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            training: true,
        }
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, H, W]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, H_out, W_out]
    pub fn forward_d4(&self, input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (dh, dw) = self.dilation;

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
        // Note: permute creates a non-contiguous view, so we need contiguous() before reshape()
        let cols: Tensor<D3, f32> =
            permuted
                .contiguous()
                .reshape([batch, h_out * w_out, self.in_channels * kh * kw]);

        // 5. Get weight and reshape: [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
        let weight = self.weight.tensor();
        let weight_flat: Tensor<D2, f32> =
            weight.reshape([self.out_channels, self.in_channels * kh * kw]);

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
        let reshaped: Tensor<D4, f32> = result.reshape([batch, h_out, w_out, self.out_channels]);
        let output: Tensor<D4, f32> = reshaped.permute(&[0, 3, 1, 2]);

        // 8. Add bias if present
        match &self.bias {
            Some(bias) => {
                let bias_tensor = bias.tensor();
                // bias: [C_out] -> [1, C_out, 1, 1]
                let bias_expanded: Tensor<D4, f32> =
                    bias_tensor.unsqueeze(0).unsqueeze(2).unsqueeze(3);
                &output + &bias_expanded
            }
            None => output,
        }
    }

    /// Get the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
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

impl Module for Conv2d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        let mut params: Vec<Box<dyn ParameterBase>> = vec![Box::new(self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(Box::new(b.clone()));
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        let mut params: Vec<(String, Box<dyn ParameterBase>)> =
            vec![("weight".to_string(), Box::new(self.weight.clone()))];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), Box::new(b.clone())));
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

// ============================================================================
// Conv1d
// ============================================================================

/// 1D Convolution layer.
///
/// Applies a 1D convolution over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `[N, C_in, L]`
/// - Output: `[N, C_out, L_out]`
///
/// Where:
/// - `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
pub struct Conv1d {
    /// Weight parameter [out_channels, in_channels, K]
    weight: Parameter<D3>,
    /// Bias parameter [out_channels] (optional)
    bias: Option<Parameter<D1>>,
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size
    kernel_size: usize,
    /// Stride
    stride: usize,
    /// Padding
    padding: usize,
    /// Dilation
    dilation: usize,
    /// Training mode flag
    training: bool,
}

impl Conv1d {
    /// Create a new Conv1d layer with default stride=1, padding=0, dilation=1, bias=true.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        Self::with_options(in_channels, out_channels, kernel_size, 1, 0, 1, true)
    }

    /// Create a new Conv1d layer with all options.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Self {
        // Kaiming uniform initialization
        let fan_in = in_channels * kernel_size;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Initialize weight
        let weight_size = out_channels * in_channels * kernel_size;
        let weight_data: Vec<f32> = (0..weight_size)
            .map(|i| (i as f32 * 0.1).sin() * bound)
            .collect();
        let weight: Parameter<D3> = Parameter::from_data(
            "weight",
            &weight_data,
            &[out_channels, in_channels, kernel_size],
        );

        // Initialize bias
        let bias = if bias {
            let bias_data = vec![0.0f32; out_channels];
            Some(Parameter::from_data("bias", &bias_data, &[out_channels]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            training: true,
        }
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, L]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, L_out]
    pub fn forward_d3(&self, input: &Tensor<D3, f32>) -> Tensor<D3, f32> {
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let d = self.dilation;

        // Get input shape
        let input_shape = input.shape();
        let batch = input_shape[0];
        let l = input_shape[2];

        // Calculate output length
        let l_out = (l + 2 * p - d * (k - 1) - 1) / s + 1;

        // 1. Padding: D3 -> D3
        let padded = input.pad(&[(0, 0), (0, 0), (p, p)]);

        // 2. unfold_1d: D3 -> D4 [N, C_in, L_out, K]
        let unfolded: Tensor<D4, f32> = padded.unfold_1d(k, s, d);

        // 3. permute: [N, C_in, L_out, K] -> [N, L_out, C_in, K]
        let permuted: Tensor<D4, f32> = unfolded.permute(&[0, 2, 1, 3]);

        // 4. contiguous + reshape: D4 -> D2 [N*L_out, C_in*K]
        // Note: permute creates a non-contiguous view, so we need contiguous() before reshape()
        let cols: Tensor<D2, f32> = permuted
            .contiguous()
            .reshape([batch * l_out, self.in_channels * k]);

        // 5. Get weight and reshape: [C_out, C_in, K] -> [C_out, C_in*K]
        let weight = self.weight.tensor();
        let weight_flat: Tensor<D2, f32> =
            weight.reshape([self.out_channels, self.in_channels * k]);

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
        let reshaped: Tensor<D3, f32> = result.reshape([batch, l_out, self.out_channels]);

        // 8. permute: [N, L_out, C_out] -> [N, C_out, L_out]
        let output: Tensor<D3, f32> = reshaped.permute(&[0, 2, 1]);

        // 9. Add bias if present
        match &self.bias {
            Some(bias) => {
                let bias_tensor = bias.tensor();
                // bias: [C_out] -> [1, C_out, 1]
                let bias_expanded: Tensor<D3, f32> = bias_tensor.unsqueeze(0).unsqueeze(2);
                &output + &bias_expanded
            }
            None => output,
        }
    }

    /// Get the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }
}

impl Module for Conv1d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        let mut params: Vec<Box<dyn ParameterBase>> = vec![Box::new(self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(Box::new(b.clone()));
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        let mut params: Vec<(String, Box<dyn ParameterBase>)> =
            vec![("weight".to_string(), Box::new(self.weight.clone()))];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), Box::new(b.clone())));
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Conv1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv1d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

// ============================================================================
// Conv3d
// ============================================================================

/// 3D Convolution layer.
///
/// Applies a 3D convolution over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `[N, C_in, D, H, W]`
/// - Output: `[N, C_out, D_out, H_out, W_out]`
///
/// Where:
/// - `D_out = (D + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
/// - `H_out = (H + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
/// - `W_out = (W + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::Conv3d;
/// use eclat::tensor::{Tensor, dim::D5};
///
/// let conv = Conv3d::new(3, 64, (3, 3, 3));
/// let input: Tensor<D5, f32> = Tensor::input([1, 3, 16, 32, 32]);
/// let output = conv.forward_d5(&input);  // [1, 64, 14, 30, 30]
/// ```
pub struct Conv3d {
    /// Weight parameter [out_channels, in_channels, kD, kH, kW]
    weight: Parameter<D5>,
    /// Bias parameter [out_channels] (optional)
    bias: Option<Parameter<D1>>,
    /// Number of input channels
    in_channels: usize,
    /// Number of output channels
    out_channels: usize,
    /// Kernel size (kD, kH, kW)
    kernel_size: (usize, usize, usize),
    /// Stride (sD, sH, sW)
    stride: (usize, usize, usize),
    /// Padding (pD, pH, pW)
    padding: (usize, usize, usize),
    /// Dilation (dD, dH, dW)
    dilation: (usize, usize, usize),
    /// Training mode flag
    training: bool,
}

impl Conv3d {
    /// Create a new Conv3d layer with default stride=1, padding=0, dilation=1, bias=true.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        Self::with_options(
            in_channels,
            out_channels,
            kernel_size,
            (1, 1, 1),
            (0, 0, 0),
            (1, 1, 1),
            true,
        )
    }

    /// Create a new Conv3d layer with all options.
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        bias: bool,
    ) -> Self {
        let (kd, kh, kw) = kernel_size;

        // Kaiming uniform initialization
        let fan_in = in_channels * kd * kh * kw;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Initialize weight
        let weight_size = out_channels * in_channels * kd * kh * kw;
        let weight_data: Vec<f32> = (0..weight_size)
            .map(|i| (i as f32 * 0.1).sin() * bound)
            .collect();
        let weight: Parameter<D5> = Parameter::from_data(
            "weight",
            &weight_data,
            &[out_channels, in_channels, kd, kh, kw],
        );

        // Initialize bias
        let bias = if bias {
            let bias_data = vec![0.0f32; out_channels];
            Some(Parameter::from_data("bias", &bias_data, &[out_channels]))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            training: true,
        }
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, D, H, W]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, D_out, H_out, W_out]
    pub fn forward_d5(&self, input: &Tensor<D5, f32>) -> Tensor<D5, f32> {
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let (dd, dh, dw) = self.dilation;

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
        let cols: Tensor<D3, f32> = permuted.contiguous().reshape([
            batch,
            d_out * h_out * w_out,
            self.in_channels * kd * kh * kw,
        ]);

        // 5. Get weight and reshape: [C_out, C_in, kD, kH, kW] -> [C_out, C_in*kD*kH*kW]
        let weight = self.weight.tensor();
        let weight_flat: Tensor<D2, f32> =
            weight.reshape([self.out_channels, self.in_channels * kd * kh * kw]);

        // 6. Matrix multiplication using broadcast multiply + sum
        // cols: [N, D_out*H_out*W_out, C_in*kD*kH*kW] -> [N, D_out*H_out*W_out, 1, C_in*kD*kH*kW]
        // weight: [C_out, C_in*kD*kH*kW] -> [1, 1, C_out, C_in*kD*kH*kW]
        let cols_expanded: Tensor<D4, f32> = cols.unsqueeze(2);
        let weight_expanded: Tensor<D4, f32> = weight_flat.unsqueeze(0).unsqueeze(0);

        // broadcast multiply: [N, D_out*H_out*W_out, C_out, C_in*kD*kH*kW]
        let product: Tensor<D4, f32> = &cols_expanded * &weight_expanded;

        // sum over last axis: [N, D_out*H_out*W_out, C_out]
        let result: Tensor<D3, f32> = product.sum(3);

        // 7. reshape + permute: [N, D_out*H_out*W_out, C_out] -> [N, D_out, H_out, W_out, C_out] -> [N, C_out, D_out, H_out, W_out]
        let reshaped: Tensor<D5, f32> =
            result.reshape([batch, d_out, h_out, w_out, self.out_channels]);
        let output: Tensor<D5, f32> = reshaped.permute(&[0, 4, 1, 2, 3]);

        // 8. Add bias if present
        match &self.bias {
            Some(bias) => {
                let bias_tensor = bias.tensor();
                // bias: [C_out] -> [1, C_out, 1, 1, 1]
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

    /// Get the number of input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get the kernel size.
    pub fn kernel_size(&self) -> (usize, usize, usize) {
        self.kernel_size
    }

    /// Get the stride.
    pub fn stride(&self) -> (usize, usize, usize) {
        self.stride
    }

    /// Get the padding.
    pub fn padding(&self) -> (usize, usize, usize) {
        self.padding
    }

    /// Get the dilation.
    pub fn dilation(&self) -> (usize, usize, usize) {
        self.dilation
    }
}

impl Module for Conv3d {
    fn parameters(&self) -> Vec<Box<dyn ParameterBase>> {
        let mut params: Vec<Box<dyn ParameterBase>> = vec![Box::new(self.weight.clone())];
        if let Some(ref b) = self.bias {
            params.push(Box::new(b.clone()));
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, Box<dyn ParameterBase>)> {
        let mut params: Vec<(String, Box<dyn ParameterBase>)> =
            vec![("weight".to_string(), Box::new(self.weight.clone()))];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), Box::new(b.clone())));
        }
        params
    }

    fn train(&mut self, mode: bool) {
        self.training = mode;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

impl std::fmt::Debug for Conv3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Conv3d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("dilation", &self.dilation)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_output_shape() {
        // Input: [1, 3, 32, 32], kernel: 3x3, stride: 1, padding: 0
        // Output: [1, 64, 30, 30]
        let conv = Conv2d::new(3, 64, (3, 3));
        let input: Tensor<D4, f32> = Tensor::input([1, 3, 32, 32]);
        let output = conv.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 64, 30, 30]);
    }

    #[test]
    fn test_conv2d_with_padding() {
        // Input: [1, 3, 32, 32], kernel: 3x3, stride: 1, padding: 1
        // Output: [1, 64, 32, 32] (same size)
        let conv = Conv2d::with_options(3, 64, (3, 3), (1, 1), (1, 1), (1, 1), true);
        let input: Tensor<D4, f32> = Tensor::input([1, 3, 32, 32]);
        let output = conv.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 64, 32, 32]);
    }

    #[test]
    fn test_conv2d_with_stride() {
        // Input: [1, 3, 32, 32], kernel: 3x3, stride: 2, padding: 1
        // Output: [1, 64, 16, 16]
        let conv = Conv2d::with_options(3, 64, (3, 3), (2, 2), (1, 1), (1, 1), true);
        let input: Tensor<D4, f32> = Tensor::input([1, 3, 32, 32]);
        let output = conv.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 64, 16, 16]);
    }

    #[test]
    fn test_conv1d_output_shape() {
        // Input: [1, 64, 100], kernel: 5, stride: 1, padding: 0
        // Output: [1, 128, 96]
        let conv = Conv1d::new(64, 128, 5);
        let input: Tensor<D3, f32> = Tensor::input([1, 64, 100]);
        let output = conv.forward_d3(&input);
        assert_eq!(output.shape(), vec![1, 128, 96]);
    }

    #[test]
    fn test_conv1d_with_padding() {
        // Input: [1, 64, 100], kernel: 5, stride: 1, padding: 2
        // Output: [1, 128, 100] (same size)
        let conv = Conv1d::with_options(64, 128, 5, 1, 2, 1, true);
        let input: Tensor<D3, f32> = Tensor::input([1, 64, 100]);
        let output = conv.forward_d3(&input);
        assert_eq!(output.shape(), vec![1, 128, 100]);
    }

    #[test]
    fn test_conv2d_parameters() {
        let conv = Conv2d::new(3, 64, (3, 3));
        let params = conv.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let conv_no_bias = Conv2d::with_options(3, 64, (3, 3), (1, 1), (0, 0), (1, 1), false);
        let params = conv_no_bias.parameters();
        assert_eq!(params.len(), 1); // weight only
    }

    #[test]
    fn test_conv3d_output_shape() {
        // Input: [1, 3, 16, 32, 32], kernel: 3x3x3, stride: 1, padding: 0
        // Output: [1, 64, 14, 30, 30]
        let conv = Conv3d::new(3, 64, (3, 3, 3));
        let input: Tensor<D5, f32> = Tensor::input([1, 3, 16, 32, 32]);
        let output = conv.forward_d5(&input);
        assert_eq!(output.shape(), vec![1, 64, 14, 30, 30]);
    }

    #[test]
    fn test_conv3d_with_padding() {
        // Input: [1, 3, 16, 32, 32], kernel: 3x3x3, stride: 1, padding: 1
        // Output: [1, 64, 16, 32, 32] (same size)
        let conv = Conv3d::with_options(3, 64, (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), true);
        let input: Tensor<D5, f32> = Tensor::input([1, 3, 16, 32, 32]);
        let output = conv.forward_d5(&input);
        assert_eq!(output.shape(), vec![1, 64, 16, 32, 32]);
    }

    #[test]
    fn test_conv3d_with_stride() {
        // Input: [1, 3, 16, 32, 32], kernel: 3x3x3, stride: 2, padding: 1
        // Output: [1, 64, 8, 16, 16]
        let conv = Conv3d::with_options(3, 64, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), true);
        let input: Tensor<D5, f32> = Tensor::input([1, 3, 16, 32, 32]);
        let output = conv.forward_d5(&input);
        assert_eq!(output.shape(), vec![1, 64, 8, 16, 16]);
    }

    #[test]
    fn test_conv3d_parameters() {
        let conv = Conv3d::new(3, 64, (3, 3, 3));
        let params = conv.parameters();
        assert_eq!(params.len(), 2); // weight + bias

        let conv_no_bias =
            Conv3d::with_options(3, 64, (3, 3, 3), (1, 1, 1), (0, 0, 0), (1, 1, 1), false);
        let params = conv_no_bias.parameters();
        assert_eq!(params.len(), 1); // weight only
    }
}

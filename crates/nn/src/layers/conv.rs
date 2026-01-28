//! Convolution layers
//!
//! Implements Conv1d, Conv2d, Conv3d and their transposed variants.
//! The actual computation is performed by the functional module.

use super::{Module, Parameter, ParameterBase};
use crate::functional;
use eclat::tensor::Tensor;
use eclat::tensor::dim::{D1, D3, D4, D5};

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
    /// Create a new Conv2d layer with default stride=1, padding=0, dilation=1, no bias.
    ///
    /// Use builder methods to customize:
    /// ```ignore
    /// let conv = Conv2d::new(3, 64, (3, 3))
    ///     .with_padding((1, 1))
    ///     .with_stride((2, 2))
    ///     .with_bias();
    /// ```
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
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

        Self {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            training: true,
        }
    }

    /// Set the stride. Default is (1, 1).
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

    /// Add a learnable bias. Default is no bias.
    pub fn with_bias(mut self) -> Self {
        if self.bias.is_none() {
            let bias_data = vec![0.0f32; self.out_channels];
            self.bias = Some(Parameter::from_data(
                "bias",
                &bias_data,
                &[self.out_channels],
            ));
        }
        self
    }

    /// Create a new Conv2d layer with all options (legacy API).
    ///
    /// Prefer using the builder pattern instead:
    /// ```ignore
    /// Conv2d::new(in_ch, out_ch, kernel).with_padding(p).with_bias()
    /// ```
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
    ) -> Self {
        let mut conv = Self::new(in_channels, out_channels, kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_dilation(dilation);
        if bias {
            conv = conv.with_bias();
        }
        conv
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, H, W]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, H_out, W_out]
    pub fn forward_d4(&self, input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
        let bias = self.bias.as_ref().map(|b| b.tensor());
        functional::conv2d(
            input,
            &self.weight.tensor(),
            bias.as_deref(),
            self.stride,
            self.padding,
            self.dilation,
        )
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
    /// Create a new Conv1d layer with default stride=1, padding=0, dilation=1, no bias.
    ///
    /// Use builder methods to customize:
    /// ```ignore
    /// let conv = Conv1d::new(3, 64, 3)
    ///     .with_padding(1)
    ///     .with_stride(2)
    ///     .with_bias();
    /// ```
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
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

        Self {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            training: true,
        }
    }

    /// Set the stride. Default is 1.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding. Default is 0.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation. Default is 1.
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Add a learnable bias. Default is no bias.
    pub fn with_bias(mut self) -> Self {
        if self.bias.is_none() {
            let bias_data = vec![0.0f32; self.out_channels];
            self.bias = Some(Parameter::from_data(
                "bias",
                &bias_data,
                &[self.out_channels],
            ));
        }
        self
    }

    /// Create a new Conv1d layer with all options (legacy API).
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Self {
        let mut conv = Self::new(in_channels, out_channels, kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_dilation(dilation);
        if bias {
            conv = conv.with_bias();
        }
        conv
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, L]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, L_out]
    pub fn forward_d3(&self, input: &Tensor<D3, f32>) -> Tensor<D3, f32> {
        let bias = self.bias.as_ref().map(|b| b.tensor());
        functional::conv1d(
            input,
            &self.weight.tensor(),
            bias.as_deref(),
            self.stride,
            self.padding,
            self.dilation,
        )
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
    /// Create a new Conv3d layer with default stride=1, padding=0, dilation=1, no bias.
    ///
    /// Use builder methods to customize:
    /// ```ignore
    /// let conv = Conv3d::new(3, 64, (3, 3, 3))
    ///     .with_padding((1, 1, 1))
    ///     .with_stride((2, 2, 2))
    ///     .with_bias();
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
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

        Self {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            training: true,
        }
    }

    /// Set the stride. Default is (1, 1, 1).
    pub fn with_stride(mut self, stride: (usize, usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding. Default is (0, 0, 0).
    pub fn with_padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Set the dilation. Default is (1, 1, 1).
    pub fn with_dilation(mut self, dilation: (usize, usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Add a learnable bias. Default is no bias.
    pub fn with_bias(mut self) -> Self {
        if self.bias.is_none() {
            let bias_data = vec![0.0f32; self.out_channels];
            self.bias = Some(Parameter::from_data(
                "bias",
                &bias_data,
                &[self.out_channels],
            ));
        }
        self
    }

    /// Create a new Conv3d layer with all options (legacy API).
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        bias: bool,
    ) -> Self {
        let mut conv = Self::new(in_channels, out_channels, kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_dilation(dilation);
        if bias {
            conv = conv.with_bias();
        }
        conv
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, D, H, W]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, D_out, H_out, W_out]
    pub fn forward_d5(&self, input: &Tensor<D5, f32>) -> Tensor<D5, f32> {
        let bias = self.bias.as_ref().map(|b| b.tensor());
        functional::conv3d(
            input,
            &self.weight.tensor(),
            bias.as_deref(),
            self.stride,
            self.padding,
            self.dilation,
        )
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

// ============================================================================
// ConvTranspose2d
// ============================================================================

/// 2D Transposed Convolution layer (Deconvolution).
///
/// Applies a 2D transposed convolution over an input signal composed of several input planes.
/// Also known as fractionally-strided convolution or deconvolution.
///
/// # Shape
/// - Input: `[N, C_in, H, W]`
/// - Output: `[N, C_out, H_out, W_out]`
///
/// Where:
/// - `H_out = (H - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
/// - `W_out = (W - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::ConvTranspose2d;
/// use eclat::tensor::{Tensor, dim::D4};
///
/// let conv_t = ConvTranspose2d::new(64, 3, (3, 3));
/// let input: Tensor<D4, f32> = Tensor::input([1, 64, 30, 30]);
/// let output = conv_t.forward_d4(&input);  // [1, 3, 32, 32]
/// ```
pub struct ConvTranspose2d {
    /// Weight parameter [in_channels, out_channels, kH, kW]
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
    /// Output padding (opH, opW)
    output_padding: (usize, usize),
    /// Dilation (dH, dW)
    dilation: (usize, usize),
    /// Training mode flag
    training: bool,
}

impl ConvTranspose2d {
    /// Create a new ConvTranspose2d layer with default options, no bias.
    ///
    /// Use builder methods to customize:
    /// ```ignore
    /// let conv_t = ConvTranspose2d::new(64, 3, (3, 3))
    ///     .with_stride((2, 2))
    ///     .with_padding((1, 1))
    ///     .with_bias();
    /// ```
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        let (kh, kw) = kernel_size;

        // Kaiming uniform initialization
        let fan_in = in_channels * kh * kw;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Initialize weight: [in_channels, out_channels, kH, kW]
        let weight_size = in_channels * out_channels * kh * kw;
        let weight_data: Vec<f32> = (0..weight_size)
            .map(|i| (i as f32 * 0.1).sin() * bound)
            .collect();
        let weight: Parameter<D4> =
            Parameter::from_data("weight", &weight_data, &[in_channels, out_channels, kh, kw]);

        Self {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            dilation: (1, 1),
            training: true,
        }
    }

    /// Set the stride. Default is (1, 1).
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Set the padding. Default is (0, 0).
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Set the output padding. Default is (0, 0).
    pub fn with_output_padding(mut self, output_padding: (usize, usize)) -> Self {
        self.output_padding = output_padding;
        self
    }

    /// Set the dilation. Default is (1, 1).
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Add a learnable bias. Default is no bias.
    pub fn with_bias(mut self) -> Self {
        if self.bias.is_none() {
            let bias_data = vec![0.0f32; self.out_channels];
            self.bias = Some(Parameter::from_data(
                "bias",
                &bias_data,
                &[self.out_channels],
            ));
        }
        self
    }

    /// Create a new ConvTranspose2d layer with all options (legacy API).
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        bias: bool,
    ) -> Self {
        let mut conv = Self::new(in_channels, out_channels, kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_output_padding(output_padding)
            .with_dilation(dilation);
        if bias {
            conv = conv.with_bias();
        }
        conv
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, H, W]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, H_out, W_out]
    pub fn forward_d4(&self, input: &Tensor<D4, f32>) -> Tensor<D4, f32> {
        let bias = self.bias.as_ref().map(|b| b.tensor());
        functional::conv_transpose2d(
            input,
            &self.weight.tensor(),
            bias.as_deref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
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

    /// Get the output padding.
    pub fn output_padding(&self) -> (usize, usize) {
        self.output_padding
    }

    /// Get the dilation.
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }
}

impl Module for ConvTranspose2d {
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

impl std::fmt::Debug for ConvTranspose2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvTranspose2d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("output_padding", &self.output_padding)
            .field("dilation", &self.dilation)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

// ============================================================================
// ConvTranspose1d
// ============================================================================

/// 1D Transposed Convolution layer (Deconvolution).
///
/// Applies a 1D transposed convolution over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `[N, C_in, L]`
/// - Output: `[N, C_out, L_out]`
///
/// Where:
/// - `L_out = (L - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
pub struct ConvTranspose1d {
    /// Weight parameter [in_channels, out_channels, K]
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
    /// Output padding
    output_padding: usize,
    /// Dilation
    dilation: usize,
    /// Training mode flag
    training: bool,
}

impl ConvTranspose1d {
    /// Create a new ConvTranspose1d layer with default stride=1, padding=0, output_padding=0, dilation=1, no bias.
    ///
    /// Use builder methods to customize:
    /// ```ignore
    /// let conv_t = ConvTranspose1d::new(64, 3, 3)
    ///     .with_stride(2)
    ///     .with_padding(1)
    ///     .with_output_padding(1)
    ///     .with_bias();
    /// ```
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        // Kaiming uniform initialization
        let fan_in = in_channels * kernel_size;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Initialize weight: [in_channels, out_channels, K]
        let weight_size = in_channels * out_channels * kernel_size;
        let weight_data: Vec<f32> = (0..weight_size)
            .map(|i| (i as f32 * 0.1).sin() * bound)
            .collect();
        let weight: Parameter<D3> = Parameter::from_data(
            "weight",
            &weight_data,
            &[in_channels, out_channels, kernel_size],
        );

        Self {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            training: true,
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

    /// Set the output padding.
    pub fn with_output_padding(mut self, output_padding: usize) -> Self {
        self.output_padding = output_padding;
        self
    }

    /// Set the dilation.
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Add a bias parameter.
    pub fn with_bias(mut self) -> Self {
        if self.bias.is_none() {
            let bias_data = vec![0.0f32; self.out_channels];
            self.bias = Some(Parameter::from_data(
                "bias",
                &bias_data,
                &[self.out_channels],
            ));
        }
        self
    }

    /// Create a new ConvTranspose1d layer with all options (legacy API).
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        bias: bool,
    ) -> Self {
        let layer = Self::new(in_channels, out_channels, kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_output_padding(output_padding)
            .with_dilation(dilation);
        if bias { layer.with_bias() } else { layer }
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, L]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, L_out]
    pub fn forward_d3(&self, input: &Tensor<D3, f32>) -> Tensor<D3, f32> {
        let bias = self.bias.as_ref().map(|b| b.tensor());
        functional::conv_transpose1d(
            input,
            &self.weight.tensor(),
            bias.as_deref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
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

impl Module for ConvTranspose1d {
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

impl std::fmt::Debug for ConvTranspose1d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvTranspose1d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("output_padding", &self.output_padding)
            .field("dilation", &self.dilation)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

// ============================================================================
// ConvTranspose3d
// ============================================================================

/// 3D Transposed Convolution layer (Deconvolution).
///
/// Applies a 3D transposed convolution over an input signal composed of several input planes.
///
/// # Shape
/// - Input: `[N, C_in, D, H, W]`
/// - Output: `[N, C_out, D_out, H_out, W_out]`
///
/// Where:
/// - `D_out = (D - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
/// - `H_out = (H - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
/// - `W_out = (W - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
///
/// # Example
///
/// ```ignore
/// use eclat_nn::ConvTranspose3d;
/// use eclat::tensor::{Tensor, dim::D5};
///
/// let conv_t = ConvTranspose3d::new(64, 3, (3, 3, 3));
/// let input: Tensor<D5, f32> = Tensor::input([1, 64, 14, 30, 30]);
/// let output = conv_t.forward_d5(&input);  // [1, 3, 16, 32, 32]
/// ```
pub struct ConvTranspose3d {
    /// Weight parameter [in_channels, out_channels, kD, kH, kW]
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
    /// Output padding (opD, opH, opW)
    output_padding: (usize, usize, usize),
    /// Dilation (dD, dH, dW)
    dilation: (usize, usize, usize),
    /// Training mode flag
    training: bool,
}

impl ConvTranspose3d {
    /// Create a new ConvTranspose3d layer with default stride=1, padding=0, output_padding=0, dilation=1, no bias.
    ///
    /// Use builder methods to customize:
    /// ```ignore
    /// let conv_t = ConvTranspose3d::new(64, 3, (3, 3, 3))
    ///     .with_stride((2, 2, 2))
    ///     .with_padding((1, 1, 1))
    ///     .with_output_padding((1, 1, 1))
    ///     .with_bias();
    /// ```
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Self {
        let (kd, kh, kw) = kernel_size;

        // Kaiming uniform initialization
        let fan_in = in_channels * kd * kh * kw;
        let bound = (1.0 / fan_in as f32).sqrt();

        // Initialize weight: [in_channels, out_channels, kD, kH, kW]
        let weight_size = in_channels * out_channels * kd * kh * kw;
        let weight_data: Vec<f32> = (0..weight_size)
            .map(|i| (i as f32 * 0.1).sin() * bound)
            .collect();
        let weight: Parameter<D5> = Parameter::from_data(
            "weight",
            &weight_data,
            &[in_channels, out_channels, kd, kh, kw],
        );

        Self {
            weight,
            bias: None,
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            dilation: (1, 1, 1),
            training: true,
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

    /// Set the output padding.
    pub fn with_output_padding(mut self, output_padding: (usize, usize, usize)) -> Self {
        self.output_padding = output_padding;
        self
    }

    /// Set the dilation.
    pub fn with_dilation(mut self, dilation: (usize, usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Add a bias parameter.
    pub fn with_bias(mut self) -> Self {
        if self.bias.is_none() {
            let bias_data = vec![0.0f32; self.out_channels];
            self.bias = Some(Parameter::from_data(
                "bias",
                &bias_data,
                &[self.out_channels],
            ));
        }
        self
    }

    /// Create a new ConvTranspose3d layer with all options (legacy API).
    pub fn with_options(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
        bias: bool,
    ) -> Self {
        let layer = Self::new(in_channels, out_channels, kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_output_padding(output_padding)
            .with_dilation(dilation);
        if bias { layer.with_bias() } else { layer }
    }

    /// Forward pass with static dimension types.
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape [N, C_in, D, H, W]
    ///
    /// # Returns
    /// Output tensor of shape [N, C_out, D_out, H_out, W_out]
    pub fn forward_d5(&self, input: &Tensor<D5, f32>) -> Tensor<D5, f32> {
        let bias = self.bias.as_ref().map(|b| b.tensor());
        functional::conv_transpose3d(
            input,
            &self.weight.tensor(),
            bias.as_deref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
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

    /// Get the output padding.
    pub fn output_padding(&self) -> (usize, usize, usize) {
        self.output_padding
    }

    /// Get the dilation.
    pub fn dilation(&self) -> (usize, usize, usize) {
        self.dilation
    }
}

impl Module for ConvTranspose3d {
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

impl std::fmt::Debug for ConvTranspose3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConvTranspose3d")
            .field("in_channels", &self.in_channels)
            .field("out_channels", &self.out_channels)
            .field("kernel_size", &self.kernel_size)
            .field("stride", &self.stride)
            .field("padding", &self.padding)
            .field("output_padding", &self.output_padding)
            .field("dilation", &self.dilation)
            .field("bias", &self.bias.is_some())
            .finish()
    }
}

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
        // new() creates layer without bias by default
        let conv = Conv2d::new(3, 64, (3, 3));
        let params = conv.parameters();
        assert_eq!(params.len(), 1); // weight only

        // with_bias() adds bias
        let conv_with_bias = Conv2d::new(3, 64, (3, 3)).with_bias();
        let params = conv_with_bias.parameters();
        assert_eq!(params.len(), 2); // weight + bias
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
        // new() creates layer without bias by default
        let conv = Conv3d::new(3, 64, (3, 3, 3));
        let params = conv.parameters();
        assert_eq!(params.len(), 1); // weight only

        // with_bias() adds bias
        let conv_with_bias = Conv3d::new(3, 64, (3, 3, 3)).with_bias();
        let params = conv_with_bias.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    // ============================================================================
    // ConvTranspose2d Tests
    // ============================================================================

    #[test]
    fn test_conv_transpose2d_output_shape() {
        // Input: [1, 64, 30, 30], kernel: 3x3, stride: 1, padding: 0
        // Output: [1, 3, 32, 32]
        let conv_t = ConvTranspose2d::new(64, 3, (3, 3));
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 30, 30]);
        let output = conv_t.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_conv_transpose2d_with_stride() {
        // Input: [1, 64, 16, 16], kernel: 3x3, stride: 2, padding: 1, output_padding: 1
        // Output: [1, 3, 32, 32]
        let conv_t =
            ConvTranspose2d::with_options(64, 3, (3, 3), (2, 2), (1, 1), (1, 1), (1, 1), true);
        let input: Tensor<D4, f32> = Tensor::input([1, 64, 16, 16]);
        let output = conv_t.forward_d4(&input);
        assert_eq!(output.shape(), vec![1, 3, 32, 32]);
    }

    #[test]
    fn test_conv_transpose2d_parameters() {
        // new() creates layer without bias by default
        let conv_t = ConvTranspose2d::new(64, 3, (3, 3));
        let params = conv_t.parameters();
        assert_eq!(params.len(), 1); // weight only

        // with_bias() adds bias
        let conv_t_with_bias = ConvTranspose2d::new(64, 3, (3, 3)).with_bias();
        let params = conv_t_with_bias.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }

    // ============================================================================
    // ConvTranspose1d Tests
    // ============================================================================

    #[test]
    fn test_conv_transpose1d_output_shape() {
        // Input: [1, 128, 96], kernel: 5, stride: 1, padding: 0
        // Output: [1, 64, 100]
        let conv_t = ConvTranspose1d::new(128, 64, 5);
        let input: Tensor<D3, f32> = Tensor::input([1, 128, 96]);
        let output = conv_t.forward_d3(&input);
        assert_eq!(output.shape(), vec![1, 64, 100]);
    }

    #[test]
    fn test_conv_transpose1d_with_padding() {
        // Input: [1, 128, 100], kernel: 5, stride: 1, padding: 2
        // Output: [1, 64, 100] (same size)
        let conv_t = ConvTranspose1d::with_options(128, 64, 5, 1, 2, 0, 1, true);
        let input: Tensor<D3, f32> = Tensor::input([1, 128, 100]);
        let output = conv_t.forward_d3(&input);
        assert_eq!(output.shape(), vec![1, 64, 100]);
    }

    // ============================================================================
    // ConvTranspose3d Tests
    // ============================================================================

    #[test]
    fn test_conv_transpose3d_output_shape() {
        // Input: [1, 64, 14, 30, 30], kernel: 3x3x3, stride: 1, padding: 0
        // Output: [1, 3, 16, 32, 32]
        let conv_t = ConvTranspose3d::new(64, 3, (3, 3, 3));
        let input: Tensor<D5, f32> = Tensor::input([1, 64, 14, 30, 30]);
        let output = conv_t.forward_d5(&input);
        assert_eq!(output.shape(), vec![1, 3, 16, 32, 32]);
    }

    #[test]
    fn test_conv_transpose3d_with_stride() {
        // Input: [1, 64, 8, 16, 16], kernel: 3x3x3, stride: 2, padding: 1, output_padding: 1
        // Output: [1, 3, 16, 32, 32]
        let conv_t = ConvTranspose3d::with_options(
            64,
            3,
            (3, 3, 3),
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            true,
        );
        let input: Tensor<D5, f32> = Tensor::input([1, 64, 8, 16, 16]);
        let output = conv_t.forward_d5(&input);
        assert_eq!(output.shape(), vec![1, 3, 16, 32, 32]);
    }

    #[test]
    fn test_conv_transpose3d_parameters() {
        // new() creates layer without bias by default
        let conv_t = ConvTranspose3d::new(64, 3, (3, 3, 3));
        let params = conv_t.parameters();
        assert_eq!(params.len(), 1); // weight only

        // with_bias() adds bias
        let conv_t_with_bias = ConvTranspose3d::new(64, 3, (3, 3, 3)).with_bias();
        let params = conv_t_with_bias.parameters();
        assert_eq!(params.len(), 2); // weight + bias
    }
}

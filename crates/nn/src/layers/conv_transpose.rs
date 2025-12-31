//! 転置畳み込み層（Transposed Convolution Layers）
//!
//! 1D, 2D, 3D転置畳み込み層の実装を提供します。

use std::marker::PhantomData;

use harp::tensor::{Dim1, Dim3, Dim4, Dim5, DimDyn, FloatDType, Tensor};
use typed_builder::TypedBuilder;

use crate::{Module, Parameter, ParameterMut};

// ============================================================================
// ConvTranspose1d
// ============================================================================

/// 1D転置畳み込み層
///
/// 入力: `[N, C_in, L]`
/// 出力: `[N, C_out, L_out]`
///
/// `L_out = (L - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1`
///
/// # Example
///
/// ```ignore
/// let conv_t = ConvTranspose1d::<f32>::new(32, 16, 4)
///     .stride(2)
///     .padding(1)
///     .build();
/// let output = conv_t.forward(&input);
/// ```
pub struct ConvTranspose1d<T: FloatDType = f32> {
    /// 重み [C_in, C_out/groups, kernel_size]
    weight: Parameter<T, Dim3>,
    /// バイアス [C_out]
    bias: Option<Parameter<T, Dim1>>,
    /// ストライド
    stride: usize,
    /// パディング
    padding: usize,
    /// 出力パディング
    output_padding: usize,
    /// ダイレーション
    dilation: usize,
    /// グループ数
    groups: usize,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> ConvTranspose1d<T> {
    /// 新しいConvTranspose1d層のビルダーを作成
    ///
    /// デフォルト値:
    /// - stride: 1
    /// - padding: 0
    /// - output_padding: 0
    /// - dilation: 1
    /// - groups: 1
    /// - bias: true
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::type_complexity)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> ConvTranspose1dConfigBuilder<T, ((usize,), (usize,), (usize,), (), (), (), (), (), ())>
    {
        ConvTranspose1dConfig::builder()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }

    /// 入力チャンネル数
    pub fn in_channels(&self) -> usize {
        self.weight.shape()[0]
    }

    /// 出力チャンネル数
    pub fn out_channels(&self) -> usize {
        self.weight.shape()[1] * self.groups
    }

    /// カーネルサイズ
    pub fn kernel_size(&self) -> usize {
        self.weight.shape()[2]
    }

    /// 順伝播
    ///
    /// 入力: `[N, C_in, L]`
    /// 出力: `[N, C_out, L_out]`
    ///
    /// groups=1もgroups>1も同じロジックで処理
    pub fn forward(&self, input: &Tensor<T, Dim3>) -> Tensor<T, Dim3> {
        let input_shape = input.shape();
        let (n, c_in, l) = (input_shape[0], input_shape[1], input_shape[2]);
        let c_out = self.out_channels();
        let k = self.kernel_size();
        let c_in_per_group = c_in / self.groups;
        let c_out_per_group = c_out / self.groups;

        assert_eq!(
            c_in,
            self.in_channels(),
            "Input channels mismatch: expected {}, got {}",
            self.in_channels(),
            c_in
        );

        // 出力サイズを計算（パディング除去前）
        let l_out_raw = (l - 1) * self.stride + self.dilation * (k - 1) + 1;

        // 最終出力サイズ
        let l_out = l_out_raw - 2 * self.padding + self.output_padding;

        let mut outputs = Vec::with_capacity(self.groups);

        for g in 0..self.groups {
            // 入力チャンネルをスライス
            let input_slice = input
                .slice(&[
                    (0, n),
                    (g * c_in_per_group, (g + 1) * c_in_per_group),
                    (0, l),
                ])
                .contiguous();

            // 重みをスライス
            let weight_slice = self
                .weight
                .as_dyn()
                .slice(&[
                    (g * c_in_per_group, (g + 1) * c_in_per_group),
                    (0, c_out_per_group),
                    (0, k),
                ])
                .contiguous();

            // Reshape input: [N, C_in/G, L] -> [N * L, C_in/G]
            let input_reshaped = input_slice
                .permute(&[0, 2, 1])
                .reshape([n * l, c_in_per_group]);

            // Reshape weight: [C_in/G, C_out/G, k] -> [C_in/G, C_out/G * k]
            let weight_reshaped = weight_slice
                .reshape_dyn(&[c_in_per_group, c_out_per_group * k])
                .into_dim2();

            // Matmul
            let group_output = input_reshaped.matmul2(&weight_reshaped);

            // Reshape and fold
            let group_output = group_output
                .into_dyn()
                .reshape_dyn(&[n, l, c_out_per_group, k])
                .permute(&[0, 2, 1, 3])
                .into_dim4()
                .fold1d_dilated(l_out_raw, self.stride, self.dilation);

            outputs.push(group_output.into_dyn());
        }

        // Concatenate along channel dimension
        let output = Tensor::concat(&outputs.iter().collect::<Vec<_>>(), 1);

        // パディングを除去して最終サイズに調整
        let output = if self.padding > 0 || self.output_padding > 0 {
            output.slice(&[(0, n), (0, c_out), (self.padding, self.padding + l_out)])
        } else {
            output
        };

        // Add bias if present
        let output = if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .as_dyn()
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .expand(output.shape());
            &output + &bias_expanded
        } else {
            output
        };

        output.into_dim3()
    }
}

/// ConvTranspose1d層の設定
#[derive(TypedBuilder)]
#[builder(build_method(into = ConvTranspose1d<T>))]
pub struct ConvTranspose1dConfig<T: FloatDType = f32> {
    /// 入力チャンネル数
    in_channels: usize,
    /// 出力チャンネル数
    out_channels: usize,
    /// カーネルサイズ
    kernel_size: usize,
    /// ストライド（デフォルト: 1）
    #[builder(default = 1)]
    stride: usize,
    /// パディング（デフォルト: 0）
    #[builder(default = 0)]
    padding: usize,
    /// 出力パディング（デフォルト: 0）
    #[builder(default = 0)]
    output_padding: usize,
    /// ダイレーション（デフォルト: 1）
    #[builder(default = 1)]
    dilation: usize,
    /// グループ数（デフォルト: 1）
    #[builder(default = 1)]
    groups: usize,
    /// バイアスの有無（デフォルト: true）
    #[builder(default = true)]
    bias: bool,
    #[builder(default, setter(skip))]
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> From<ConvTranspose1dConfig<T>> for ConvTranspose1d<T> {
    fn from(config: ConvTranspose1dConfig<T>) -> Self {
        assert!(config.in_channels > 0, "in_channels must be positive");
        assert!(config.out_channels > 0, "out_channels must be positive");
        assert!(config.kernel_size > 0, "kernel_size must be positive");
        assert!(config.stride > 0, "stride must be positive");
        assert!(config.dilation > 0, "dilation must be positive");
        assert!(config.groups > 0, "groups must be positive");
        assert!(
            config.in_channels.is_multiple_of(config.groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            config.out_channels.is_multiple_of(config.groups),
            "out_channels must be divisible by groups"
        );
        assert!(
            config.output_padding < config.stride,
            "output_padding must be smaller than stride"
        );

        let c_out_per_group = config.out_channels / config.groups;

        // 重みの初期化 (static dimension)
        let weight =
            Tensor::<T, Dim3>::rand([config.in_channels, c_out_per_group, config.kernel_size]);

        let bias = if config.bias {
            Some(Parameter::new(Tensor::<T, Dim1>::zeros([
                config.out_channels
            ])))
        } else {
            None
        };

        ConvTranspose1d {
            weight: Parameter::new(weight),
            bias,
            stride: config.stride,
            padding: config.padding,
            output_padding: config.output_padding,
            dilation: config.dilation,
            groups: config.groups,
            _dtype: PhantomData,
        }
    }
}

// ============================================================================
// ConvTranspose2d
// ============================================================================

/// 2D転置畳み込み層
///
/// 入力: `[N, C_in, H, W]`
/// 出力: `[N, C_out, H_out, W_out]`
///
/// # Example
///
/// ```ignore
/// let conv_t = ConvTranspose2d::<f32>::new(64, 32, (4, 4))
///     .stride((2, 2))
///     .padding((1, 1))
///     .build();
/// let output = conv_t.forward(&input);
/// ```
pub struct ConvTranspose2d<T: FloatDType = f32> {
    /// 重み [C_in, C_out/groups, kH, kW]
    weight: Parameter<T, Dim4>,
    /// バイアス [C_out]
    bias: Option<Parameter<T, Dim1>>,
    /// ストライド (sH, sW)
    stride: (usize, usize),
    /// パディング (pH, pW)
    padding: (usize, usize),
    /// 出力パディング (opH, opW)
    output_padding: (usize, usize),
    /// ダイレーション (dH, dW)
    dilation: (usize, usize),
    /// グループ数
    groups: usize,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> ConvTranspose2d<T> {
    /// 新しいConvTranspose2d層のビルダーを作成
    ///
    /// デフォルト値:
    /// - stride: (1, 1)
    /// - padding: (0, 0)
    /// - output_padding: (0, 0)
    /// - dilation: (1, 1)
    /// - groups: 1
    /// - bias: true
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::type_complexity)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> ConvTranspose2dConfigBuilder<
        T,
        (
            (usize,),
            (usize,),
            ((usize, usize),),
            (),
            (),
            (),
            (),
            (),
            (),
        ),
    > {
        ConvTranspose2dConfig::builder()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }

    /// 入力チャンネル数
    pub fn in_channels(&self) -> usize {
        self.weight.shape()[0]
    }

    /// 出力チャンネル数
    pub fn out_channels(&self) -> usize {
        self.weight.shape()[1] * self.groups
    }

    /// カーネルサイズ
    pub fn kernel_size(&self) -> (usize, usize) {
        (self.weight.shape()[2], self.weight.shape()[3])
    }

    /// 順伝播
    ///
    /// 入力: `[N, C_in, H, W]`
    /// 出力: `[N, C_out, H_out, W_out]`
    ///
    /// groups=1もgroups>1も同じロジックで処理
    pub fn forward(&self, input: &Tensor<T, Dim4>) -> Tensor<T, Dim4> {
        let input_shape = input.shape();
        let (n, c_in, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        );
        let c_out = self.out_channels();
        let (kh, kw) = self.kernel_size();
        let c_in_per_group = c_in / self.groups;
        let c_out_per_group = c_out / self.groups;

        assert_eq!(
            c_in,
            self.in_channels(),
            "Input channels mismatch: expected {}, got {}",
            self.in_channels(),
            c_in
        );

        // 出力サイズを計算（パディング除去前）
        let h_out_raw = (h - 1) * self.stride.0 + self.dilation.0 * (kh - 1) + 1;
        let w_out_raw = (w - 1) * self.stride.1 + self.dilation.1 * (kw - 1) + 1;

        // 最終出力サイズ
        let h_out = h_out_raw - 2 * self.padding.0 + self.output_padding.0;
        let w_out = w_out_raw - 2 * self.padding.1 + self.output_padding.1;

        let mut outputs = Vec::with_capacity(self.groups);

        for g in 0..self.groups {
            // 入力チャンネルをスライス
            let input_slice = input
                .slice(&[
                    (0, n),
                    (g * c_in_per_group, (g + 1) * c_in_per_group),
                    (0, h),
                    (0, w),
                ])
                .contiguous();

            // 重みをスライス
            let weight_slice = self
                .weight
                .as_dyn()
                .slice(&[
                    (g * c_in_per_group, (g + 1) * c_in_per_group),
                    (0, c_out_per_group),
                    (0, kh),
                    (0, kw),
                ])
                .contiguous();

            // Reshape input: [N, C_in/G, H, W] -> [N * H * W, C_in/G]
            let input_reshaped = input_slice
                .permute(&[0, 2, 3, 1])
                .reshape([n * h * w, c_in_per_group]);

            // Reshape weight: [C_in/G, C_out/G, kH, kW] -> [C_in/G, C_out/G * kH * kW]
            let weight_reshaped = weight_slice
                .reshape_dyn(&[c_in_per_group, c_out_per_group * kh * kw])
                .into_dim2();

            // Matmul
            let group_output = input_reshaped.matmul2(&weight_reshaped);

            // Reshape and fold
            let group_output = group_output
                .into_dyn()
                .reshape_dyn(&[n, h, w, c_out_per_group, kh, kw])
                .permute(&[0, 3, 1, 2, 4, 5])
                .into_dim6()
                .fold2d_dilated((h_out_raw, w_out_raw), self.stride, self.dilation);

            outputs.push(group_output.into_dyn());
        }

        // Concatenate along channel dimension
        let output = Tensor::concat(&outputs.iter().collect::<Vec<_>>(), 1);

        // パディングを除去
        let output = if self.padding.0 > 0
            || self.padding.1 > 0
            || self.output_padding.0 > 0
            || self.output_padding.1 > 0
        {
            output.slice(&[
                (0, n),
                (0, c_out),
                (self.padding.0, self.padding.0 + h_out),
                (self.padding.1, self.padding.1 + w_out),
            ])
        } else {
            output
        };

        // Add bias if present
        let output = if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .as_dyn()
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .expand(output.shape());
            &output + &bias_expanded
        } else {
            output
        };

        output.into_dim4()
    }
}

/// ConvTranspose2d層の設定
#[derive(TypedBuilder)]
#[builder(build_method(into = ConvTranspose2d<T>))]
pub struct ConvTranspose2dConfig<T: FloatDType = f32> {
    /// 入力チャンネル数
    in_channels: usize,
    /// 出力チャンネル数
    out_channels: usize,
    /// カーネルサイズ (kH, kW)
    kernel_size: (usize, usize),
    /// ストライド (sH, sW)（デフォルト: (1, 1)）
    #[builder(default = (1, 1))]
    stride: (usize, usize),
    /// パディング (pH, pW)（デフォルト: (0, 0)）
    #[builder(default = (0, 0))]
    padding: (usize, usize),
    /// 出力パディング (opH, opW)（デフォルト: (0, 0)）
    #[builder(default = (0, 0))]
    output_padding: (usize, usize),
    /// ダイレーション (dH, dW)（デフォルト: (1, 1)）
    #[builder(default = (1, 1))]
    dilation: (usize, usize),
    /// グループ数（デフォルト: 1）
    #[builder(default = 1)]
    groups: usize,
    /// バイアスの有無（デフォルト: true）
    #[builder(default = true)]
    bias: bool,
    #[builder(default, setter(skip))]
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> From<ConvTranspose2dConfig<T>> for ConvTranspose2d<T> {
    fn from(config: ConvTranspose2dConfig<T>) -> Self {
        assert!(config.in_channels > 0, "in_channels must be positive");
        assert!(config.out_channels > 0, "out_channels must be positive");
        assert!(
            config.kernel_size.0 > 0 && config.kernel_size.1 > 0,
            "kernel_size must be positive"
        );
        assert!(
            config.stride.0 > 0 && config.stride.1 > 0,
            "stride must be positive"
        );
        assert!(
            config.dilation.0 > 0 && config.dilation.1 > 0,
            "dilation must be positive"
        );
        assert!(config.groups > 0, "groups must be positive");
        assert!(
            config.in_channels.is_multiple_of(config.groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            config.out_channels.is_multiple_of(config.groups),
            "out_channels must be divisible by groups"
        );
        assert!(
            config.output_padding.0 < config.stride.0 && config.output_padding.1 < config.stride.1,
            "output_padding must be smaller than stride"
        );

        let c_out_per_group = config.out_channels / config.groups;

        // 重みの初期化 (static dimension)
        let weight = Tensor::<T, Dim4>::rand([
            config.in_channels,
            c_out_per_group,
            config.kernel_size.0,
            config.kernel_size.1,
        ]);

        let bias = if config.bias {
            Some(Parameter::new(Tensor::<T, Dim1>::zeros([
                config.out_channels
            ])))
        } else {
            None
        };

        ConvTranspose2d {
            weight: Parameter::new(weight),
            bias,
            stride: config.stride,
            padding: config.padding,
            output_padding: config.output_padding,
            dilation: config.dilation,
            groups: config.groups,
            _dtype: PhantomData,
        }
    }
}

// ============================================================================
// ConvTranspose3d
// ============================================================================

/// 3D転置畳み込み層
///
/// 入力: `[N, C_in, D, H, W]`
/// 出力: `[N, C_out, D_out, H_out, W_out]`
///
/// # Example
///
/// ```ignore
/// let conv_t = ConvTranspose3d::<f32>::new(64, 32, (4, 4, 4))
///     .stride((2, 2, 2))
///     .padding((1, 1, 1))
///     .build();
/// let output = conv_t.forward(&input);
/// ```
pub struct ConvTranspose3d<T: FloatDType = f32> {
    /// 重み [C_in, C_out/groups, kD, kH, kW]
    weight: Parameter<T, Dim5>,
    /// バイアス [C_out]
    bias: Option<Parameter<T, Dim1>>,
    /// ストライド (sD, sH, sW)
    stride: (usize, usize, usize),
    /// パディング (pD, pH, pW)
    padding: (usize, usize, usize),
    /// 出力パディング (opD, opH, opW)
    output_padding: (usize, usize, usize),
    /// ダイレーション (dD, dH, dW)
    dilation: (usize, usize, usize),
    /// グループ数
    groups: usize,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> ConvTranspose3d<T> {
    /// 新しいConvTranspose3d層のビルダーを作成
    ///
    /// デフォルト値:
    /// - stride: (1, 1, 1)
    /// - padding: (0, 0, 0)
    /// - output_padding: (0, 0, 0)
    /// - dilation: (1, 1, 1)
    /// - groups: 1
    /// - bias: true
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::type_complexity)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> ConvTranspose3dConfigBuilder<
        T,
        (
            (usize,),
            (usize,),
            ((usize, usize, usize),),
            (),
            (),
            (),
            (),
            (),
            (),
        ),
    > {
        ConvTranspose3dConfig::builder()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }

    /// 入力チャンネル数
    pub fn in_channels(&self) -> usize {
        self.weight.shape()[0]
    }

    /// 出力チャンネル数
    pub fn out_channels(&self) -> usize {
        self.weight.shape()[1] * self.groups
    }

    /// カーネルサイズ
    pub fn kernel_size(&self) -> (usize, usize, usize) {
        (
            self.weight.shape()[2],
            self.weight.shape()[3],
            self.weight.shape()[4],
        )
    }

    /// 順伝播
    ///
    /// 入力: `[N, C_in, D, H, W]`
    /// 出力: `[N, C_out, D_out, H_out, W_out]`
    ///
    /// groups=1もgroups>1も同じロジックで処理
    pub fn forward(&self, input: &Tensor<T, Dim5>) -> Tensor<T, Dim5> {
        let input_shape = input.shape();
        let (n, c_in, d, h, w) = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
            input_shape[4],
        );
        let c_out = self.out_channels();
        let (kd, kh, kw) = self.kernel_size();
        let c_in_per_group = c_in / self.groups;
        let c_out_per_group = c_out / self.groups;

        assert_eq!(
            c_in,
            self.in_channels(),
            "Input channels mismatch: expected {}, got {}",
            self.in_channels(),
            c_in
        );

        // 出力サイズを計算（パディング除去前）
        let d_out_raw = (d - 1) * self.stride.0 + self.dilation.0 * (kd - 1) + 1;
        let h_out_raw = (h - 1) * self.stride.1 + self.dilation.1 * (kh - 1) + 1;
        let w_out_raw = (w - 1) * self.stride.2 + self.dilation.2 * (kw - 1) + 1;

        // 最終出力サイズ
        let d_out = d_out_raw - 2 * self.padding.0 + self.output_padding.0;
        let h_out = h_out_raw - 2 * self.padding.1 + self.output_padding.1;
        let w_out = w_out_raw - 2 * self.padding.2 + self.output_padding.2;

        let mut outputs = Vec::with_capacity(self.groups);

        for g in 0..self.groups {
            // 入力チャンネルをスライス
            let input_slice = input
                .slice(&[
                    (0, n),
                    (g * c_in_per_group, (g + 1) * c_in_per_group),
                    (0, d),
                    (0, h),
                    (0, w),
                ])
                .contiguous();

            // 重みをスライス
            let weight_slice = self
                .weight
                .as_dyn()
                .slice(&[
                    (g * c_in_per_group, (g + 1) * c_in_per_group),
                    (0, c_out_per_group),
                    (0, kd),
                    (0, kh),
                    (0, kw),
                ])
                .contiguous();

            // Reshape input: [N, C_in/G, D, H, W] -> [N * D * H * W, C_in/G]
            let input_reshaped = input_slice
                .permute(&[0, 2, 3, 4, 1])
                .reshape([n * d * h * w, c_in_per_group]);

            // Reshape weight: [C_in/G, C_out/G, kD, kH, kW] -> [C_in/G, C_out/G * kD * kH * kW]
            let weight_reshaped = weight_slice
                .reshape_dyn(&[c_in_per_group, c_out_per_group * kd * kh * kw])
                .into_dim2();

            // Matmul
            let group_output = input_reshaped.matmul2(&weight_reshaped);

            // Reshape and fold
            let group_output = group_output
                .into_dyn()
                .reshape_dyn(&[n, d, h, w, c_out_per_group, kd, kh, kw])
                .permute(&[0, 4, 1, 2, 3, 5, 6, 7])
                .into_dim8()
                .fold3d_dilated(
                    (d_out_raw, h_out_raw, w_out_raw),
                    self.stride,
                    self.dilation,
                );

            outputs.push(group_output.into_dyn());
        }

        // Concatenate along channel dimension
        let output = Tensor::concat(&outputs.iter().collect::<Vec<_>>(), 1);

        // パディングを除去
        let output = if self.padding.0 > 0
            || self.padding.1 > 0
            || self.padding.2 > 0
            || self.output_padding.0 > 0
            || self.output_padding.1 > 0
            || self.output_padding.2 > 0
        {
            output.slice(&[
                (0, n),
                (0, c_out),
                (self.padding.0, self.padding.0 + d_out),
                (self.padding.1, self.padding.1 + h_out),
                (self.padding.2, self.padding.2 + w_out),
            ])
        } else {
            output
        };

        // Add bias if present
        let output = if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .as_dyn()
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .expand(output.shape());
            &output + &bias_expanded
        } else {
            output
        };

        output.into_dim5()
    }
}

/// ConvTranspose3d層の設定
#[derive(TypedBuilder)]
#[builder(build_method(into = ConvTranspose3d<T>))]
pub struct ConvTranspose3dConfig<T: FloatDType = f32> {
    /// 入力チャンネル数
    in_channels: usize,
    /// 出力チャンネル数
    out_channels: usize,
    /// カーネルサイズ (kD, kH, kW)
    kernel_size: (usize, usize, usize),
    /// ストライド (sD, sH, sW)（デフォルト: (1, 1, 1)）
    #[builder(default = (1, 1, 1))]
    stride: (usize, usize, usize),
    /// パディング (pD, pH, pW)（デフォルト: (0, 0, 0)）
    #[builder(default = (0, 0, 0))]
    padding: (usize, usize, usize),
    /// 出力パディング (opD, opH, opW)（デフォルト: (0, 0, 0)）
    #[builder(default = (0, 0, 0))]
    output_padding: (usize, usize, usize),
    /// ダイレーション (dD, dH, dW)（デフォルト: (1, 1, 1)）
    #[builder(default = (1, 1, 1))]
    dilation: (usize, usize, usize),
    /// グループ数（デフォルト: 1）
    #[builder(default = 1)]
    groups: usize,
    /// バイアスの有無（デフォルト: true）
    #[builder(default = true)]
    bias: bool,
    #[builder(default, setter(skip))]
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> From<ConvTranspose3dConfig<T>> for ConvTranspose3d<T> {
    fn from(config: ConvTranspose3dConfig<T>) -> Self {
        assert!(config.in_channels > 0, "in_channels must be positive");
        assert!(config.out_channels > 0, "out_channels must be positive");
        assert!(
            config.kernel_size.0 > 0 && config.kernel_size.1 > 0 && config.kernel_size.2 > 0,
            "kernel_size must be positive in all dimensions"
        );
        assert!(
            config.stride.0 > 0 && config.stride.1 > 0 && config.stride.2 > 0,
            "stride must be positive in all dimensions"
        );
        assert!(
            config.dilation.0 > 0 && config.dilation.1 > 0 && config.dilation.2 > 0,
            "dilation must be positive in all dimensions"
        );
        assert!(config.groups > 0, "groups must be positive");
        assert!(
            config.in_channels.is_multiple_of(config.groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            config.out_channels.is_multiple_of(config.groups),
            "out_channels must be divisible by groups"
        );
        assert!(
            config.output_padding.0 < config.stride.0
                && config.output_padding.1 < config.stride.1
                && config.output_padding.2 < config.stride.2,
            "output_padding must be smaller than stride"
        );

        let c_out_per_group = config.out_channels / config.groups;

        // 重みの初期化 (static dimension)
        let weight = Tensor::<T, Dim5>::rand([
            config.in_channels,
            c_out_per_group,
            config.kernel_size.0,
            config.kernel_size.1,
            config.kernel_size.2,
        ]);

        let bias = if config.bias {
            Some(Parameter::new(Tensor::<T, Dim1>::zeros([
                config.out_channels
            ])))
        } else {
            None
        };

        ConvTranspose3d {
            weight: Parameter::new(weight),
            bias,
            stride: config.stride,
            padding: config.padding,
            output_padding: config.output_padding,
            dilation: config.dilation,
            groups: config.groups,
            _dtype: PhantomData,
        }
    }
}

// ============================================================================
// Module implementations
// ============================================================================

impl<T: FloatDType> Module<T> for ConvTranspose1d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut dyn ParameterMut<T>> {
        let mut params: std::collections::HashMap<String, &mut dyn ParameterMut<T>> =
            std::collections::HashMap::new();
        params.insert(
            "weight".to_string(),
            &mut self.weight as &mut dyn ParameterMut<T>,
        );
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias as &mut dyn ParameterMut<T>);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Tensor<T, DimDyn>>) {
        if let Some(w) = params.get("weight") {
            ParameterMut::set_dyn(&mut self.weight, w.clone());
        }
        if let Some(b) = params.get("bias")
            && let Some(ref mut bias) = self.bias
        {
            ParameterMut::set_dyn(bias, b.clone());
        }
    }
}

impl<T: FloatDType> Module<T> for ConvTranspose2d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut dyn ParameterMut<T>> {
        let mut params: std::collections::HashMap<String, &mut dyn ParameterMut<T>> =
            std::collections::HashMap::new();
        params.insert(
            "weight".to_string(),
            &mut self.weight as &mut dyn ParameterMut<T>,
        );
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias as &mut dyn ParameterMut<T>);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Tensor<T, DimDyn>>) {
        if let Some(w) = params.get("weight") {
            ParameterMut::set_dyn(&mut self.weight, w.clone());
        }
        if let Some(b) = params.get("bias")
            && let Some(ref mut bias) = self.bias
        {
            ParameterMut::set_dyn(bias, b.clone());
        }
    }
}

impl<T: FloatDType> Module<T> for ConvTranspose3d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut dyn ParameterMut<T>> {
        let mut params: std::collections::HashMap<String, &mut dyn ParameterMut<T>> =
            std::collections::HashMap::new();
        params.insert(
            "weight".to_string(),
            &mut self.weight as &mut dyn ParameterMut<T>,
        );
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias as &mut dyn ParameterMut<T>);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Tensor<T, DimDyn>>) {
        if let Some(w) = params.get("weight") {
            ParameterMut::set_dyn(&mut self.weight, w.clone());
        }
        if let Some(b) = params.get("bias")
            && let Some(ref mut bias) = self.bias
        {
            ParameterMut::set_dyn(bias, b.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Module;

    // =========================================================================
    // ConvTranspose1d Tests
    // =========================================================================

    #[test]
    fn test_conv_transpose1d_creation() {
        let conv = ConvTranspose1d::<f32>::new(32, 16, 4).build();
        assert_eq!(conv.in_channels(), 32);
        assert_eq!(conv.out_channels(), 16);
        assert_eq!(conv.kernel_size(), 4);
    }

    #[test]
    fn test_conv_transpose1d_forward_shape() {
        let conv = ConvTranspose1d::<f32>::new(32, 16, 4)
            .stride(2)
            .padding(1)
            .build();
        let input = Tensor::<f32, Dim3>::rand([2, 32, 50]);
        let output = conv.forward(&input);
        // L_out = (L-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
        // = (50-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 98 - 2 + 3 + 1 = 100
        assert_eq!(output.shape(), &[2, 16, 100]);
    }

    #[test]
    fn test_conv_transpose1d_grouped() {
        let conv = ConvTranspose1d::<f32>::new(32, 16, 4)
            .stride(2)
            .padding(1)
            .groups(4)
            .build();
        let input = Tensor::<f32, Dim3>::rand([2, 32, 50]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 16, 100]);
    }

    #[test]
    fn test_conv_transpose1d_parameters() {
        let mut conv = ConvTranspose1d::<f32>::new(32, 16, 4).build();
        let params = conv.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // =========================================================================
    // ConvTranspose2d Tests
    // =========================================================================

    #[test]
    fn test_conv_transpose2d_creation() {
        let conv = ConvTranspose2d::<f32>::new(64, 32, (4, 4)).build();
        assert_eq!(conv.in_channels(), 64);
        assert_eq!(conv.out_channels(), 32);
        assert_eq!(conv.kernel_size(), (4, 4));
    }

    #[test]
    fn test_conv_transpose2d_forward_shape() {
        let conv = ConvTranspose2d::<f32>::new(64, 32, (4, 4))
            .stride((2, 2))
            .padding((1, 1))
            .build();
        let input = Tensor::<f32, Dim4>::rand([2, 64, 16, 16]);
        let output = conv.forward(&input);
        // H_out = (H-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
        // = (16-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 30 - 2 + 3 + 1 = 32
        assert_eq!(output.shape(), &[2, 32, 32, 32]);
    }

    #[test]
    fn test_conv_transpose2d_grouped() {
        let conv = ConvTranspose2d::<f32>::new(64, 32, (4, 4))
            .stride((2, 2))
            .padding((1, 1))
            .groups(4)
            .build();
        let input = Tensor::<f32, Dim4>::rand([2, 64, 16, 16]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 32, 32, 32]);
    }

    #[test]
    fn test_conv_transpose2d_parameters() {
        let mut conv = ConvTranspose2d::<f32>::new(64, 32, (4, 4)).build();
        let params = conv.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    // =========================================================================
    // ConvTranspose3d Tests
    // =========================================================================

    #[test]
    fn test_conv_transpose3d_creation() {
        let conv = ConvTranspose3d::<f32>::new(64, 32, (4, 4, 4)).build();
        assert_eq!(conv.in_channels(), 64);
        assert_eq!(conv.out_channels(), 32);
        assert_eq!(conv.kernel_size(), (4, 4, 4));
    }

    #[test]
    fn test_conv_transpose3d_forward_shape() {
        let conv = ConvTranspose3d::<f32>::new(64, 32, (4, 4, 4))
            .stride((2, 2, 2))
            .padding((1, 1, 1))
            .build();
        let input = Tensor::<f32, Dim5>::rand([2, 64, 8, 8, 8]);
        let output = conv.forward(&input);
        // D_out = (D-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
        // = (8-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 14 - 2 + 3 + 1 = 16
        assert_eq!(output.shape(), &[2, 32, 16, 16, 16]);
    }

    #[test]
    fn test_conv_transpose3d_grouped() {
        let conv = ConvTranspose3d::<f32>::new(64, 32, (4, 4, 4))
            .stride((2, 2, 2))
            .padding((1, 1, 1))
            .groups(4)
            .build();
        let input = Tensor::<f32, Dim5>::rand([2, 64, 8, 8, 8]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 32, 16, 16, 16]);
    }

    #[test]
    fn test_conv_transpose3d_parameters() {
        let mut conv = ConvTranspose3d::<f32>::new(64, 32, (4, 4, 4)).build();
        let params = conv.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }
}

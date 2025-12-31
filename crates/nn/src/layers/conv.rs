//! 畳み込み層（Convolution Layers）
//!
//! 1D, 2D, 3D畳み込み層の実装を提供します。

use std::marker::PhantomData;

use harp::tensor::{Dim1, Dim3, Dim4, Dim5, DimDyn, FloatDType, Tensor};
use typed_builder::TypedBuilder;

use crate::{Module, Parameter, ParameterMut};

// ============================================================================
// Conv1d
// ============================================================================

/// Conv1d層の設定
#[derive(TypedBuilder)]
#[builder(build_method(into = Conv1d<T>))]
pub struct Conv1dConfig<T: FloatDType = f32> {
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

impl<T: FloatDType> From<Conv1dConfig<T>> for Conv1d<T> {
    fn from(config: Conv1dConfig<T>) -> Self {
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

        let c_in_per_group = config.in_channels / config.groups;
        let weight =
            Tensor::<T, Dim3>::rand([config.out_channels, c_in_per_group, config.kernel_size]);

        let bias = if config.bias {
            Some(Parameter::new(Tensor::<T, Dim1>::zeros([
                config.out_channels
            ])))
        } else {
            None
        };

        Conv1d {
            weight: Parameter::new(weight),
            bias,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            groups: config.groups,
            _dtype: PhantomData,
        }
    }
}

/// 1D畳み込み層
///
/// 入力: `[N, C_in, L]`
/// 出力: `[N, C_out, L_out]`
///
/// `L_out = (L + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1`
///
/// # Example
///
/// ```ignore
/// let conv = Conv1d::<f32>::new(16, 32, 3)
///     .stride(1)
///     .padding(1)
///     .build();
/// let output = conv.forward(&input);
/// ```
pub struct Conv1d<T: FloatDType = f32> {
    /// 重み [C_out, C_in/groups, kernel_size]
    weight: Parameter<T, Dim3>,
    /// バイアス [C_out] (Noneの場合はバイアスなし)
    bias: Option<Parameter<T, Dim1>>,
    /// ストライド
    stride: usize,
    /// パディング
    padding: usize,
    /// ダイレーション
    dilation: usize,
    /// グループ数
    groups: usize,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Conv1d<T> {
    /// Conv1d ビルダーを作成
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::type_complexity)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> Conv1dConfigBuilder<T, ((usize,), (usize,), (usize,), (), (), (), (), ())> {
        Conv1dConfig::builder()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }

    /// 入力チャンネル数
    pub fn in_channels(&self) -> usize {
        self.weight.shape()[1] * self.groups
    }

    /// 出力チャンネル数
    pub fn out_channels(&self) -> usize {
        self.weight.shape()[0]
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
    /// 要素積とsumを使用した統一実装（groups=1も groups>1 も同じロジック）
    pub fn forward(&self, input: &Tensor<T, Dim3>) -> Tensor<T, Dim3> {
        let input_shape = input.shape();
        let (n, c_in, l) = (input_shape[0], input_shape[1], input_shape[2]);
        let c_out = self.out_channels();
        let k = self.kernel_size();
        let groups = self.groups;
        let c_in_per_group = c_in / groups;
        let c_out_per_group = c_out / groups;

        assert_eq!(
            c_in,
            self.in_channels(),
            "Input channels mismatch: expected {}, got {}",
            self.in_channels(),
            c_in
        );

        // パディングを適用
        let padded = if self.padding > 0 {
            input.pad(
                &[(0, 0), (0, 0), (self.padding, self.padding)],
                harp::tensor::ops::PadValue::Zero,
            )
        } else {
            input.clone()
        };

        // 出力サイズを計算
        let l_padded = l + 2 * self.padding;
        let eff_k = (k - 1) * self.dilation + 1;
        let l_out = (l_padded - eff_k) / self.stride + 1;

        // Unfold: [N, C_in, L] -> [N, C_in, L_out, k]
        let unfolded = padded.unfold1d_dilated(k, self.stride, self.dilation);

        // グループ対応のためにreshape
        // [N, C_in, L_out, k] -> [N, groups, C_in/groups, L_out, k]
        let unfolded = unfolded
            .into_dyn()
            .reshape_dyn(&[n, groups, c_in_per_group, l_out, k]);

        // 重みをreshape
        // [C_out, C_in/groups, k] -> [groups, C_out/groups, C_in/groups, k]
        let weight =
            self.weight
                .as_dyn()
                .reshape_dyn(&[groups, c_out_per_group, c_in_per_group, k]);

        // ブロードキャスト用にunsqueeze
        // unfolded: [N, groups, C_in/groups, L_out, k]
        //        -> [N, groups, 1, C_in/groups, L_out, k]
        let unfolded = unfolded.unsqueeze(2);

        // weight: [groups, C_out/groups, C_in/groups, k]
        //      -> [1, groups, C_out/groups, C_in/groups, 1, k]
        let weight = weight.unsqueeze(0).unsqueeze(4);

        // 要素積: [N, groups, C_out/groups, C_in/groups, L_out, k]
        let product = &unfolded * &weight;

        // sum over (C_in/groups, k) -> [N, groups, C_out/groups, L_out]
        let output = product.sum(5).sum(3);

        // reshape: [N, groups, C_out/groups, L_out] -> [N, C_out, L_out]
        let mut output = output.reshape_dyn(&[n, c_out, l_out]);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .as_dyn()
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .expand(output.shape());
            output = &output + &bias_expanded;
        }

        output.into_dim3()
    }
}

// ============================================================================
// Conv2d
// ============================================================================

/// 2D畳み込み層
///
/// 入力: `[N, C_in, H, W]`
/// 出力: `[N, C_out, H_out, W_out]`
///
/// # Example
///
/// ```ignore
/// let conv = Conv2d::<f32>::new(3, 64, (3, 3))
///     .stride((1, 1))
///     .padding((1, 1))
///     .build();
/// let output = conv.forward(&input);
/// ```
pub struct Conv2d<T: FloatDType = f32> {
    /// 重み [C_out, C_in/groups, kH, kW]
    weight: Parameter<T, Dim4>,
    /// バイアス [C_out]
    bias: Option<Parameter<T, Dim1>>,
    /// ストライド (sH, sW)
    stride: (usize, usize),
    /// パディング (pH, pW)
    padding: (usize, usize),
    /// ダイレーション (dH, dW)
    dilation: (usize, usize),
    /// グループ数
    groups: usize,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Conv2d<T> {
    /// 新しいConv2d層のビルダーを作成
    ///
    /// デフォルト値:
    /// - stride: (1, 1)
    /// - padding: (0, 0)
    /// - dilation: (1, 1)
    /// - groups: 1
    /// - bias: true
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::type_complexity)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Conv2dConfigBuilder<T, ((usize,), (usize,), ((usize, usize),), (), (), (), (), ())> {
        Conv2dConfig::builder()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }

    /// 入力チャンネル数
    pub fn in_channels(&self) -> usize {
        self.weight.shape()[1] * self.groups
    }

    /// 出力チャンネル数
    pub fn out_channels(&self) -> usize {
        self.weight.shape()[0]
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
    /// 要素積とsumを使用した統一実装（groups=1も groups>1 も同じロジック）
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
        let groups = self.groups;
        let c_in_per_group = c_in / groups;
        let c_out_per_group = c_out / groups;

        assert_eq!(
            c_in,
            self.in_channels(),
            "Input channels mismatch: expected {}, got {}",
            self.in_channels(),
            c_in
        );

        // パディングを適用
        let padded = if self.padding.0 > 0 || self.padding.1 > 0 {
            input.pad(
                &[
                    (0, 0),
                    (0, 0),
                    (self.padding.0, self.padding.0),
                    (self.padding.1, self.padding.1),
                ],
                harp::tensor::ops::PadValue::Zero,
            )
        } else {
            input.clone()
        };

        // 出力サイズを計算
        let h_padded = h + 2 * self.padding.0;
        let w_padded = w + 2 * self.padding.1;
        let eff_kh = (kh - 1) * self.dilation.0 + 1;
        let eff_kw = (kw - 1) * self.dilation.1 + 1;
        let h_out = (h_padded - eff_kh) / self.stride.0 + 1;
        let w_out = (w_padded - eff_kw) / self.stride.1 + 1;

        // Unfold: [N, C_in, H, W] -> [N, C_in, H_out, W_out, kH, kW]
        let unfolded = padded.unfold2d_dilated((kh, kw), self.stride, self.dilation);

        // グループ対応のためにreshape
        // [N, C_in, H_out, W_out, kH, kW] -> [N, groups, C_in/groups, H_out, W_out, kH, kW]
        let unfolded =
            unfolded
                .into_dyn()
                .reshape_dyn(&[n, groups, c_in_per_group, h_out, w_out, kh, kw]);

        // 重みをreshape
        // [C_out, C_in/groups, kH, kW] -> [groups, C_out/groups, C_in/groups, kH, kW]
        let weight =
            self.weight
                .as_dyn()
                .reshape_dyn(&[groups, c_out_per_group, c_in_per_group, kh, kw]);

        // ブロードキャスト用にunsqueeze
        // unfolded: [N, groups, C_in/groups, H_out, W_out, kH, kW]
        //        -> [N, groups, 1, C_in/groups, H_out, W_out, kH, kW]
        let unfolded = unfolded.unsqueeze(2);

        // weight: [groups, C_out/groups, C_in/groups, kH, kW]
        //      -> [1, groups, C_out/groups, C_in/groups, 1, 1, kH, kW]
        let weight = weight.unsqueeze(0).unsqueeze(4).unsqueeze(5);

        // 要素積: [N, groups, C_out/groups, C_in/groups, H_out, W_out, kH, kW]
        let product = &unfolded * &weight;

        // sum over (C_in/groups, kH, kW) -> [N, groups, C_out/groups, H_out, W_out]
        let output = product.sum(7).sum(6).sum(3);

        // reshape: [N, groups, C_out/groups, H_out, W_out] -> [N, C_out, H_out, W_out]
        let mut output = output.reshape_dyn(&[n, c_out, h_out, w_out]);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .as_dyn()
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .expand(output.shape());
            output = &output + &bias_expanded;
        }

        output.into_dim4()
    }
}

/// Conv2d層の設定
#[derive(TypedBuilder)]
#[builder(build_method(into = Conv2d<T>))]
pub struct Conv2dConfig<T: FloatDType = f32> {
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

impl<T: FloatDType> From<Conv2dConfig<T>> for Conv2d<T> {
    fn from(config: Conv2dConfig<T>) -> Self {
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

        let c_in_per_group = config.in_channels / config.groups;

        // 重みの初期化 (static dimension)
        let weight = Tensor::<T, Dim4>::rand([
            config.out_channels,
            c_in_per_group,
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

        Conv2d {
            weight: Parameter::new(weight),
            bias,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            groups: config.groups,
            _dtype: PhantomData,
        }
    }
}

// ============================================================================
// Conv3d
// ============================================================================

/// 3D畳み込み層
///
/// 入力: `[N, C_in, D, H, W]`
/// 出力: `[N, C_out, D_out, H_out, W_out]`
///
/// # Example
///
/// ```ignore
/// let conv = Conv3d::<f32>::new(3, 64, (3, 3, 3))
///     .stride((1, 1, 1))
///     .padding((1, 1, 1))
///     .build();
/// let output = conv.forward(&input);
/// ```
pub struct Conv3d<T: FloatDType = f32> {
    /// 重み [C_out, C_in/groups, kD, kH, kW]
    weight: Parameter<T, Dim5>,
    /// バイアス [C_out]
    bias: Option<Parameter<T, Dim1>>,
    /// ストライド (sD, sH, sW)
    stride: (usize, usize, usize),
    /// パディング (pD, pH, pW)
    padding: (usize, usize, usize),
    /// ダイレーション (dD, dH, dW)
    dilation: (usize, usize, usize),
    /// グループ数
    groups: usize,
    /// 型マーカー
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Conv3d<T> {
    /// 新しいConv3d層のビルダーを作成
    ///
    /// デフォルト値:
    /// - stride: (1, 1, 1)
    /// - padding: (0, 0, 0)
    /// - dilation: (1, 1, 1)
    /// - groups: 1
    /// - bias: true
    #[allow(clippy::new_ret_no_self)]
    #[allow(clippy::type_complexity)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Conv3dConfigBuilder<
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
        ),
    > {
        Conv3dConfig::builder()
            .in_channels(in_channels)
            .out_channels(out_channels)
            .kernel_size(kernel_size)
    }

    /// 入力チャンネル数
    pub fn in_channels(&self) -> usize {
        self.weight.shape()[1] * self.groups
    }

    /// 出力チャンネル数
    pub fn out_channels(&self) -> usize {
        self.weight.shape()[0]
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
    /// 要素積とsumを使用した統一実装（groups=1も groups>1 も同じロジック）
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
        let groups = self.groups;
        let c_in_per_group = c_in / groups;
        let c_out_per_group = c_out / groups;

        assert_eq!(
            c_in,
            self.in_channels(),
            "Input channels mismatch: expected {}, got {}",
            self.in_channels(),
            c_in
        );

        // パディングを適用
        let padded = if self.padding.0 > 0 || self.padding.1 > 0 || self.padding.2 > 0 {
            input.pad(
                &[
                    (0, 0),
                    (0, 0),
                    (self.padding.0, self.padding.0),
                    (self.padding.1, self.padding.1),
                    (self.padding.2, self.padding.2),
                ],
                harp::tensor::ops::PadValue::Zero,
            )
        } else {
            input.clone()
        };

        // 出力サイズを計算
        let d_padded = d + 2 * self.padding.0;
        let h_padded = h + 2 * self.padding.1;
        let w_padded = w + 2 * self.padding.2;
        let eff_kd = (kd - 1) * self.dilation.0 + 1;
        let eff_kh = (kh - 1) * self.dilation.1 + 1;
        let eff_kw = (kw - 1) * self.dilation.2 + 1;
        let d_out = (d_padded - eff_kd) / self.stride.0 + 1;
        let h_out = (h_padded - eff_kh) / self.stride.1 + 1;
        let w_out = (w_padded - eff_kw) / self.stride.2 + 1;

        // Unfold: [N, C_in, D, H, W] -> [N, C_in, D_out, H_out, W_out, kD, kH, kW]
        let unfolded = padded.unfold3d_dilated((kd, kh, kw), self.stride, self.dilation);

        // グループ対応のためにreshape
        // [N, C_in, D_out, H_out, W_out, kD, kH, kW]
        // -> [N, groups, C_in/groups, D_out, H_out, W_out, kD, kH, kW]
        let unfolded = unfolded.into_dyn().reshape_dyn(&[
            n,
            groups,
            c_in_per_group,
            d_out,
            h_out,
            w_out,
            kd,
            kh,
            kw,
        ]);

        // 重みをreshape
        // [C_out, C_in/groups, kD, kH, kW] -> [groups, C_out/groups, C_in/groups, kD, kH, kW]
        let weight = self.weight.as_dyn().reshape_dyn(&[
            groups,
            c_out_per_group,
            c_in_per_group,
            kd,
            kh,
            kw,
        ]);

        // ブロードキャスト用にunsqueeze
        // unfolded: [N, groups, C_in/groups, D_out, H_out, W_out, kD, kH, kW]
        //        -> [N, groups, 1, C_in/groups, D_out, H_out, W_out, kD, kH, kW]
        let unfolded = unfolded.unsqueeze(2);

        // weight: [groups, C_out/groups, C_in/groups, kD, kH, kW]
        //      -> [1, groups, C_out/groups, C_in/groups, 1, 1, 1, kD, kH, kW]
        let weight = weight.unsqueeze(0).unsqueeze(4).unsqueeze(5).unsqueeze(6);

        // 要素積: [N, groups, C_out/groups, C_in/groups, D_out, H_out, W_out, kD, kH, kW]
        let product = &unfolded * &weight;

        // sum over (C_in/groups, kD, kH, kW) -> [N, groups, C_out/groups, D_out, H_out, W_out]
        let output = product.sum(9).sum(8).sum(7).sum(3);

        // reshape: [N, groups, C_out/groups, D_out, H_out, W_out] -> [N, C_out, D_out, H_out, W_out]
        let mut output = output.reshape_dyn(&[n, c_out, d_out, h_out, w_out]);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .as_dyn()
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .unsqueeze(4)
                .expand(output.shape());
            output = &output + &bias_expanded;
        }

        output.into_dim5()
    }
}

/// Conv3d層の設定
#[derive(TypedBuilder)]
#[builder(build_method(into = Conv3d<T>))]
pub struct Conv3dConfig<T: FloatDType = f32> {
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

impl<T: FloatDType> From<Conv3dConfig<T>> for Conv3d<T> {
    fn from(config: Conv3dConfig<T>) -> Self {
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

        let c_in_per_group = config.in_channels / config.groups;

        // 重みの初期化 (static dimension)
        let weight = Tensor::<T, Dim5>::rand([
            config.out_channels,
            c_in_per_group,
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

        Conv3d {
            weight: Parameter::new(weight),
            bias,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            groups: config.groups,
            _dtype: PhantomData,
        }
    }
}

// ============================================================================
// Module implementations
// ============================================================================

impl<T: FloatDType> Module<T> for Conv1d<T> {
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

impl<T: FloatDType> Module<T> for Conv2d<T> {
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

impl<T: FloatDType> Module<T> for Conv3d<T> {
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
    // Conv1d Tests
    // =========================================================================

    #[test]
    fn test_conv1d_creation() {
        let conv = Conv1d::<f32>::new(16, 32, 3).build();
        assert_eq!(conv.in_channels(), 16);
        assert_eq!(conv.out_channels(), 32);
        assert_eq!(conv.kernel_size(), 3);
    }

    #[test]
    fn test_conv1d_forward_shape() {
        let conv = Conv1d::<f32>::new(16, 32, 3).padding(1).build();
        let input = Tensor::<f32, Dim3>::rand([2, 16, 100]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 32, 100]);
    }

    #[test]
    fn test_conv1d_no_bias() {
        let conv = Conv1d::<f32>::new(16, 32, 3).bias(false).build();
        assert!(conv.bias.is_none());
    }

    #[test]
    fn test_conv1d_parameters() {
        let mut conv = Conv1d::<f32>::new(16, 32, 3).build();
        let params = conv.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    #[test]
    fn test_conv1d_grouped() {
        let conv = Conv1d::<f32>::new(16, 32, 3).groups(4).padding(1).build();
        let input = Tensor::<f32, Dim3>::rand([2, 16, 100]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 32, 100]);
    }

    // =========================================================================
    // Conv2d Tests
    // =========================================================================

    #[test]
    fn test_conv2d_creation() {
        let conv = Conv2d::<f32>::new(3, 64, (3, 3)).build();
        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 64);
        assert_eq!(conv.kernel_size(), (3, 3));
    }

    #[test]
    fn test_conv2d_forward_shape() {
        let conv = Conv2d::<f32>::new(3, 64, (3, 3)).padding((1, 1)).build();
        let input = Tensor::<f32, Dim4>::rand([2, 3, 32, 32]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 32, 32]);
    }

    #[test]
    fn test_conv2d_stride() {
        let conv = Conv2d::<f32>::new(3, 64, (3, 3))
            .stride((2, 2))
            .padding((1, 1))
            .build();
        let input = Tensor::<f32, Dim4>::rand([2, 3, 32, 32]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 16, 16]);
    }

    #[test]
    fn test_conv2d_dilation() {
        let conv = Conv2d::<f32>::new(3, 64, (3, 3))
            .dilation((2, 2))
            .padding((2, 2))
            .build();
        let input = Tensor::<f32, Dim4>::rand([2, 3, 32, 32]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 32, 32]);
    }

    #[test]
    fn test_conv2d_grouped() {
        let conv = Conv2d::<f32>::new(16, 32, (3, 3))
            .groups(4)
            .padding((1, 1))
            .build();
        let input = Tensor::<f32, Dim4>::rand([2, 16, 32, 32]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 32, 32, 32]);
    }

    #[test]
    fn test_conv2d_depthwise() {
        // Depthwise convolution: groups == in_channels == out_channels
        let conv = Conv2d::<f32>::new(16, 16, (3, 3))
            .groups(16)
            .padding((1, 1))
            .build();
        let input = Tensor::<f32, Dim4>::rand([2, 16, 32, 32]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 16, 32, 32]);
    }

    #[test]
    fn test_conv2d_parameters() {
        let mut conv = Conv2d::<f32>::new(3, 64, (3, 3)).build();
        let params = conv.parameters();
        assert!(params.contains_key("weight"));
        assert!(params.contains_key("bias"));
    }

    #[test]
    fn test_conv2d_num_parameters() {
        let mut conv = Conv2d::<f32>::new(3, 64, (3, 3)).build();
        // weight: 64 * 3 * 3 * 3 = 1728, bias: 64
        assert_eq!(conv.num_parameters(), 1728 + 64);
    }

    // =========================================================================
    // Conv3d Tests
    // =========================================================================

    #[test]
    fn test_conv3d_creation() {
        let conv = Conv3d::<f32>::new(3, 64, (3, 3, 3)).build();
        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 64);
        assert_eq!(conv.kernel_size(), (3, 3, 3));
    }

    #[test]
    fn test_conv3d_forward_shape() {
        let conv = Conv3d::<f32>::new(3, 64, (3, 3, 3))
            .padding((1, 1, 1))
            .build();
        let input = Tensor::<f32, Dim5>::rand([2, 3, 16, 16, 16]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 16, 16, 16]);
    }

    #[test]
    fn test_conv3d_grouped() {
        let conv = Conv3d::<f32>::new(8, 16, (3, 3, 3))
            .groups(4)
            .padding((1, 1, 1))
            .build();
        let input = Tensor::<f32, Dim5>::rand([2, 8, 16, 16, 16]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 16, 16, 16, 16]);
    }
}

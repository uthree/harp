//! 畳み込み層（Convolution Layers）
//!
//! 1D, 2D, 3D畳み込み層の実装を提供します。

use std::marker::PhantomData;

use harp::tensor::{Dim1, Dim3, Dim4, Dim5, DimDyn, FloatDType, Tensor};

use crate::{Module, Parameter};

// ============================================================================
// Conv1d
// ============================================================================

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
    weight: Parameter<T>,
    /// バイアス [C_out] (Noneの場合はバイアスなし)
    bias: Option<Parameter<T>>,
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
    /// Conv1dBuilder を作成
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Conv1dBuilder<T> {
        Conv1dBuilder::new(in_channels, out_channels, kernel_size)
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
    /// 実装: unfold + broadcast multiply + sum (groups統一)
    pub fn forward(&self, input: &Tensor<T, Dim3>) -> Tensor<T, Dim3> {
        let input_shape = input.shape();
        let (n, c_in, _l) = (input_shape[0], input_shape[1], input_shape[2]);
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

        // Unfold: [N, C_in, L_padded] -> [N, C_in, L_out, k]
        let unfolded = padded.unfold1d_dilated(k, self.stride, self.dilation);
        let l_out = unfolded.shape()[2];

        // Reshape to introduce groups: [N, C_in, L_out, k] -> [N, groups, C_in/groups, L_out, k]
        let unfolded_grouped =
            unfolded
                .into_dyn()
                .reshape_dyn(&[n, groups, c_in_per_group, l_out, k]);

        // Add C_out/groups dimension and explicit expand: [N, groups, C_out/groups, C_in/groups, L_out, k]
        let unfolded_expanded = unfolded_grouped.unsqueeze(2).expand(&[
            n,
            groups,
            c_out_per_group,
            c_in_per_group,
            l_out,
            k,
        ]);

        // Weight: [C_out, C_in/groups, k] -> [groups, C_out/groups, C_in/groups, k]
        // -> explicit expand to [N, groups, C_out/groups, C_in/groups, L_out, k]
        let weight_expanded = self
            .weight
            .clone()
            .reshape_dyn(&[groups, c_out_per_group, c_in_per_group, k])
            .unsqueeze(0)
            .unsqueeze(4)
            .expand(&[n, groups, c_out_per_group, c_in_per_group, l_out, k]);

        // Elementwise multiply (same shape, no implicit broadcast)
        let product = &unfolded_expanded * &weight_expanded;

        // Sum over C_in/groups (axis 3) and k (axis 5): [N, groups, C_out/groups, L_out]
        let summed: harp::tensor::Tensor<T, DimDyn> = product.sum(5).sum(3);

        // Reshape to [N, C_out, L_out]
        let mut output = summed.reshape_dyn(&[n, c_out, l_out]);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .expand(output.shape());
            output = &output + &bias_expanded;
        }

        output.into_dim3()
    }
}

/// Conv1dのビルダー
pub struct Conv1dBuilder<T: FloatDType = f32> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Conv1dBuilder<T> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(kernel_size > 0, "kernel_size must be positive");

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            _dtype: PhantomData,
        }
    }

    /// ストライドを設定
    pub fn stride(mut self, stride: usize) -> Self {
        assert!(stride > 0, "stride must be positive");
        self.stride = stride;
        self
    }

    /// パディングを設定
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// ダイレーションを設定
    pub fn dilation(mut self, dilation: usize) -> Self {
        assert!(dilation > 0, "dilation must be positive");
        self.dilation = dilation;
        self
    }

    /// グループ数を設定
    pub fn groups(mut self, groups: usize) -> Self {
        assert!(groups > 0, "groups must be positive");
        assert!(
            self.in_channels.is_multiple_of(groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            self.out_channels.is_multiple_of(groups),
            "out_channels must be divisible by groups"
        );
        self.groups = groups;
        self
    }

    /// バイアスの有無を設定
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Conv1dを構築
    pub fn build(self) -> Conv1d<T> {
        let c_in_per_group = self.in_channels / self.groups;

        // 重みの初期化
        let weight = Tensor::<T, Dim3>::rand([self.out_channels, c_in_per_group, self.kernel_size])
            .into_dyn();

        let bias = if self.bias {
            Some(Parameter::new(
                Tensor::<T, Dim1>::zeros([self.out_channels]).into_dyn(),
            ))
        } else {
            None
        };

        Conv1d {
            weight: Parameter::new(weight),
            bias,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
            _dtype: PhantomData,
        }
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
    weight: Parameter<T>,
    /// バイアス [C_out]
    bias: Option<Parameter<T>>,
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
    /// Conv2dBuilder を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Conv2dBuilder<T> {
        Conv2dBuilder::new(in_channels, out_channels, kernel_size)
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
    /// 実装: unfold + broadcast multiply + sum (groups統一)
    pub fn forward(&self, input: &Tensor<T, Dim4>) -> Tensor<T, Dim4> {
        let input_shape = input.shape();
        let (n, c_in, _h, _w) = (
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

        // Unfold: [N, C_in, H, W] -> [N, C_in, H_out, W_out, kH, kW]
        let unfolded = padded.unfold2d_dilated((kh, kw), self.stride, self.dilation);
        let h_out = unfolded.shape()[2];
        let w_out = unfolded.shape()[3];

        // Reshape to introduce groups: [N, C_in, H_out, W_out, kH, kW] -> [N, groups, C_in/groups, H_out, W_out, kH, kW]
        let unfolded_grouped =
            unfolded
                .into_dyn()
                .reshape_dyn(&[n, groups, c_in_per_group, h_out, w_out, kh, kw]);

        // Add C_out/groups dimension and explicit expand: [N, groups, C_out/groups, C_in/groups, H_out, W_out, kH, kW]
        let unfolded_expanded = unfolded_grouped.unsqueeze(2).expand(&[
            n,
            groups,
            c_out_per_group,
            c_in_per_group,
            h_out,
            w_out,
            kh,
            kw,
        ]);

        // Weight: [C_out, C_in/groups, kH, kW] -> [groups, C_out/groups, C_in/groups, kH, kW]
        // -> explicit expand to [N, groups, C_out/groups, C_in/groups, H_out, W_out, kH, kW]
        let weight_expanded = self
            .weight
            .clone()
            .reshape_dyn(&[groups, c_out_per_group, c_in_per_group, kh, kw])
            .unsqueeze(0)
            .unsqueeze(4)
            .unsqueeze(5)
            .expand(&[
                n,
                groups,
                c_out_per_group,
                c_in_per_group,
                h_out,
                w_out,
                kh,
                kw,
            ]);

        // Elementwise multiply (same shape, no implicit broadcast)
        let product = &unfolded_expanded * &weight_expanded;

        // Sum over C_in/groups (axis 3), kH (axis 6), kW (axis 7): [N, groups, C_out/groups, H_out, W_out]
        let summed: harp::tensor::Tensor<T, DimDyn> = product.sum(7).sum(6).sum(3);

        // Reshape to [N, C_out, H_out, W_out]
        let mut output = summed.reshape_dyn(&[n, c_out, h_out, w_out]);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
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

/// Conv2dのビルダー
pub struct Conv2dBuilder<T: FloatDType = f32> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    bias: bool,
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Conv2dBuilder<T> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel_size must be positive"
        );

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            bias: true,
            _dtype: PhantomData,
        }
    }

    /// ストライドを設定
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        assert!(
            stride.0 > 0 && stride.1 > 0,
            "stride must be positive in both dimensions"
        );
        self.stride = stride;
        self
    }

    /// パディングを設定
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// ダイレーションを設定
    pub fn dilation(mut self, dilation: (usize, usize)) -> Self {
        assert!(
            dilation.0 > 0 && dilation.1 > 0,
            "dilation must be positive in both dimensions"
        );
        self.dilation = dilation;
        self
    }

    /// グループ数を設定
    pub fn groups(mut self, groups: usize) -> Self {
        assert!(groups > 0, "groups must be positive");
        assert!(
            self.in_channels.is_multiple_of(groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            self.out_channels.is_multiple_of(groups),
            "out_channels must be divisible by groups"
        );
        self.groups = groups;
        self
    }

    /// バイアスの有無を設定
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Conv2dを構築
    pub fn build(self) -> Conv2d<T> {
        let c_in_per_group = self.in_channels / self.groups;

        // 重みの初期化
        let weight = Tensor::<T, Dim4>::rand([
            self.out_channels,
            c_in_per_group,
            self.kernel_size.0,
            self.kernel_size.1,
        ])
        .into_dyn();

        let bias = if self.bias {
            Some(Parameter::new(
                Tensor::<T, Dim1>::zeros([self.out_channels]).into_dyn(),
            ))
        } else {
            None
        };

        Conv2d {
            weight: Parameter::new(weight),
            bias,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
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
    weight: Parameter<T>,
    /// バイアス [C_out]
    bias: Option<Parameter<T>>,
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
    /// Conv3dBuilder を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> Conv3dBuilder<T> {
        Conv3dBuilder::new(in_channels, out_channels, kernel_size)
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
    /// 実装: unfold + broadcast multiply + sum (groups統一)
    pub fn forward(&self, input: &Tensor<T, Dim5>) -> Tensor<T, Dim5> {
        let input_shape = input.shape();
        let (n, c_in, _d, _h, _w) = (
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

        // Unfold: [N, C_in, D, H, W] -> [N, C_in, D_out, H_out, W_out, kD, kH, kW]
        let unfolded = padded.unfold3d_dilated((kd, kh, kw), self.stride, self.dilation);
        let d_out = unfolded.shape()[2];
        let h_out = unfolded.shape()[3];
        let w_out = unfolded.shape()[4];

        // Reshape to introduce groups: [N, C_in, D_out, H_out, W_out, kD, kH, kW]
        //                           -> [N, groups, C_in/groups, D_out, H_out, W_out, kD, kH, kW]
        let unfolded_grouped = unfolded.into_dyn().reshape_dyn(&[
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

        // Add C_out/groups dimension and explicit expand:
        // [N, groups, C_out/groups, C_in/groups, D_out, H_out, W_out, kD, kH, kW]
        let unfolded_expanded = unfolded_grouped.unsqueeze(2).expand(&[
            n,
            groups,
            c_out_per_group,
            c_in_per_group,
            d_out,
            h_out,
            w_out,
            kd,
            kh,
            kw,
        ]);

        // Weight: [C_out, C_in/groups, kD, kH, kW] -> [groups, C_out/groups, C_in/groups, kD, kH, kW]
        // -> explicit expand to [N, groups, C_out/groups, C_in/groups, D_out, H_out, W_out, kD, kH, kW]
        let weight_expanded = self
            .weight
            .clone()
            .reshape_dyn(&[groups, c_out_per_group, c_in_per_group, kd, kh, kw])
            .unsqueeze(0)
            .unsqueeze(4)
            .unsqueeze(5)
            .unsqueeze(6)
            .expand(&[
                n,
                groups,
                c_out_per_group,
                c_in_per_group,
                d_out,
                h_out,
                w_out,
                kd,
                kh,
                kw,
            ]);

        // Elementwise multiply (same shape, no implicit broadcast)
        let product = &unfolded_expanded * &weight_expanded;

        // Sum over C_in/groups (axis 3), kD (axis 7), kH (axis 8), kW (axis 9)
        let summed: harp::tensor::Tensor<T, DimDyn> = product.sum(9).sum(8).sum(7).sum(3);

        // Reshape to [N, C_out, D_out, H_out, W_out]
        let mut output = summed.reshape_dyn(&[n, c_out, d_out, h_out, w_out]);

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
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

/// Conv3dのビルダー
pub struct Conv3dBuilder<T: FloatDType = f32> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    bias: bool,
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> Conv3dBuilder<T> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize, usize)) -> Self {
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0 && kernel_size.2 > 0,
            "kernel_size must be positive in all dimensions"
        );

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
            bias: true,
            _dtype: PhantomData,
        }
    }

    /// ストライドを設定
    pub fn stride(mut self, stride: (usize, usize, usize)) -> Self {
        assert!(
            stride.0 > 0 && stride.1 > 0 && stride.2 > 0,
            "stride must be positive in all dimensions"
        );
        self.stride = stride;
        self
    }

    /// パディングを設定
    pub fn padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// ダイレーションを設定
    pub fn dilation(mut self, dilation: (usize, usize, usize)) -> Self {
        assert!(
            dilation.0 > 0 && dilation.1 > 0 && dilation.2 > 0,
            "dilation must be positive in all dimensions"
        );
        self.dilation = dilation;
        self
    }

    /// グループ数を設定
    pub fn groups(mut self, groups: usize) -> Self {
        assert!(groups > 0, "groups must be positive");
        assert!(
            self.in_channels.is_multiple_of(groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            self.out_channels.is_multiple_of(groups),
            "out_channels must be divisible by groups"
        );
        self.groups = groups;
        self
    }

    /// バイアスの有無を設定
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Conv3dを構築
    pub fn build(self) -> Conv3d<T> {
        let c_in_per_group = self.in_channels / self.groups;

        // 重みの初期化
        let weight = Tensor::<T, Dim5>::rand([
            self.out_channels,
            c_in_per_group,
            self.kernel_size.0,
            self.kernel_size.1,
            self.kernel_size.2,
        ])
        .into_dyn();

        let bias = if self.bias {
            Some(Parameter::new(
                Tensor::<T, Dim1>::zeros([self.out_channels]).into_dyn(),
            ))
        } else {
            None
        };

        Conv3d {
            weight: Parameter::new(weight),
            bias,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
            _dtype: PhantomData,
        }
    }
}

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
/// let conv_t = ConvTranspose1d::<f32>::new(32, 16, 3)
///     .stride(2)
///     .padding(1)
///     .build();
/// let output = conv_t.forward(&input);
/// ```
pub struct ConvTranspose1d<T: FloatDType = f32> {
    /// 重み [C_in, C_out/groups, kernel_size]
    weight: Parameter<T>,
    /// バイアス [C_out]
    bias: Option<Parameter<T>>,
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
    /// ConvTranspose1dBuilder を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    ) -> ConvTranspose1dBuilder<T> {
        ConvTranspose1dBuilder::new(in_channels, out_channels, kernel_size)
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
    pub fn forward(&self, input: &Tensor<T, Dim3>) -> Tensor<T, Dim3> {
        let input_shape = input.shape();
        let (n, c_in, l) = (input_shape[0], input_shape[1], input_shape[2]);
        let c_out = self.out_channels();
        let k = self.kernel_size();

        assert_eq!(
            c_in,
            self.in_channels(),
            "Input channels mismatch: expected {}, got {}",
            self.in_channels(),
            c_in
        );

        // 出力サイズを計算
        // raw: fold操作の直接出力サイズ（パディング除去前）
        let l_out_raw = (l - 1) * self.stride + self.dilation * (k - 1) + 1;
        // final: 最終出力サイズ（パディング除去後）
        let l_out = (l - 1) * self.stride - 2 * self.padding
            + self.dilation * (k - 1)
            + self.output_padding
            + 1;

        if self.groups == 1 {
            self.forward_no_groups(input, n, c_in, c_out, l, l_out_raw, l_out, k)
        } else {
            self.forward_grouped(input, n, c_in, c_out, l, l_out_raw, l_out, k)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_no_groups(
        &self,
        input: &Tensor<T, Dim3>,
        n: usize,
        c_in: usize,
        c_out: usize,
        l: usize,
        l_out_raw: usize,
        l_out: usize,
        k: usize,
    ) -> Tensor<T, Dim3> {
        // Input: [N, C_in, L] -> [N, L, C_in] -> [N * L, C_in]
        let input_reshaped = input.permute(&[0, 2, 1]).reshape([n * l, c_in]);

        // Weight: [C_in, C_out, k] -> [C_in, C_out * k]
        let weight_reshaped = self
            .weight
            .clone()
            .reshape_dyn(&[c_in, c_out * k])
            .into_dim2();

        // Matmul: [N * L, C_in] @ [C_in, C_out * k] -> [N * L, C_out * k]
        let output = input_reshaped.matmul2(&weight_reshaped);

        // Reshape: [N * L, C_out * k] -> [N, L, C_out, k] -> [N, C_out, L, k]
        let output = output
            .into_dyn()
            .reshape_dyn(&[n, l, c_out, k])
            .permute(&[0, 2, 1, 3])
            .into_dim4();

        // Fold: [N, C_out, L, k] -> [N, C_out, L_out_raw]
        let mut output = output
            .fold1d_dilated(l_out_raw, self.stride, self.dilation)
            .into_dyn();

        // パディングを除去し、最終出力サイズに調整
        if self.padding > 0 || self.output_padding > 0 {
            output = output.slice(&[(0, n), (0, c_out), (self.padding, self.padding + l_out)]);
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .expand(output.shape());
            output = &output + &bias_expanded;
        }

        output.into_dim3()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_grouped(
        &self,
        input: &Tensor<T, Dim3>,
        n: usize,
        c_in: usize,
        c_out: usize,
        l: usize,
        l_out_raw: usize,
        l_out: usize,
        k: usize,
    ) -> Tensor<T, Dim3> {
        let c_in_per_group = c_in / self.groups;
        let c_out_per_group = c_out / self.groups;

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

        // パディングを除去し、最終出力サイズに調整
        let output = if self.padding > 0 || self.output_padding > 0 {
            output.slice(&[(0, n), (0, c_out), (self.padding, self.padding + l_out)])
        } else {
            output
        };

        // Add bias if present
        let output = if let Some(ref bias) = self.bias {
            let bias_expanded = bias
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

/// ConvTranspose1dのビルダー
pub struct ConvTranspose1dBuilder<T: FloatDType = f32> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> ConvTranspose1dBuilder<T> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: usize) -> Self {
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(kernel_size > 0, "kernel_size must be positive");

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
            padding: 0,
            output_padding: 0,
            dilation: 1,
            groups: 1,
            bias: true,
            _dtype: PhantomData,
        }
    }

    /// ストライドを設定
    pub fn stride(mut self, stride: usize) -> Self {
        assert!(stride > 0, "stride must be positive");
        self.stride = stride;
        self
    }

    /// パディングを設定
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// 出力パディングを設定
    pub fn output_padding(mut self, output_padding: usize) -> Self {
        assert!(
            output_padding < self.stride,
            "output_padding must be smaller than stride"
        );
        self.output_padding = output_padding;
        self
    }

    /// ダイレーションを設定
    pub fn dilation(mut self, dilation: usize) -> Self {
        assert!(dilation > 0, "dilation must be positive");
        self.dilation = dilation;
        self
    }

    /// グループ数を設定
    pub fn groups(mut self, groups: usize) -> Self {
        assert!(groups > 0, "groups must be positive");
        assert!(
            self.in_channels.is_multiple_of(groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            self.out_channels.is_multiple_of(groups),
            "out_channels must be divisible by groups"
        );
        self.groups = groups;
        self
    }

    /// バイアスの有無を設定
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// ConvTranspose1dを構築
    pub fn build(self) -> ConvTranspose1d<T> {
        let c_out_per_group = self.out_channels / self.groups;

        // 重みの初期化
        let weight = Tensor::<T, Dim3>::rand([self.in_channels, c_out_per_group, self.kernel_size])
            .into_dyn();

        let bias = if self.bias {
            Some(Parameter::new(
                Tensor::<T, Dim1>::zeros([self.out_channels]).into_dyn(),
            ))
        } else {
            None
        };

        ConvTranspose1d {
            weight: Parameter::new(weight),
            bias,
            stride: self.stride,
            padding: self.padding,
            output_padding: self.output_padding,
            dilation: self.dilation,
            groups: self.groups,
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
    weight: Parameter<T>,
    /// バイアス [C_out]
    bias: Option<Parameter<T>>,
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
    /// ConvTranspose2dBuilder を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> ConvTranspose2dBuilder<T> {
        ConvTranspose2dBuilder::new(in_channels, out_channels, kernel_size)
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

        if self.groups == 1 {
            self.forward_no_groups(
                input, n, c_in, c_out, h, w, h_out, w_out, h_out_raw, w_out_raw, kh, kw,
            )
        } else {
            self.forward_grouped(
                input, n, c_in, c_out, h, w, h_out, w_out, h_out_raw, w_out_raw, kh, kw,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_no_groups(
        &self,
        input: &Tensor<T, Dim4>,
        n: usize,
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        h_out: usize,
        w_out: usize,
        h_out_raw: usize,
        w_out_raw: usize,
        kh: usize,
        kw: usize,
    ) -> Tensor<T, Dim4> {
        // Input: [N, C_in, H, W] -> [N, H, W, C_in] -> [N * H * W, C_in]
        let input_reshaped = input.permute(&[0, 2, 3, 1]).reshape([n * h * w, c_in]);

        // Weight: [C_in, C_out, kH, kW] -> [C_in, C_out * kH * kW]
        let weight_reshaped = self
            .weight
            .clone()
            .reshape_dyn(&[c_in, c_out * kh * kw])
            .into_dim2();

        // Matmul: [N * H * W, C_in] @ [C_in, C_out * kH * kW] -> [N * H * W, C_out * kH * kW]
        let output = input_reshaped.matmul2(&weight_reshaped);

        // Reshape: [N * H * W, C_out * kH * kW] -> [N, H, W, C_out, kH, kW] -> [N, C_out, H, W, kH, kW]
        let output = output
            .into_dyn()
            .reshape_dyn(&[n, h, w, c_out, kh, kw])
            .permute(&[0, 3, 1, 2, 4, 5])
            .into_dim6();

        // Fold: [N, C_out, H, W, kH, kW] -> [N, C_out, H_out_raw, W_out_raw]
        let output = output.fold2d_dilated((h_out_raw, w_out_raw), self.stride, self.dilation);

        // パディングを除去して最終サイズに調整
        let mut output = output.into_dyn();
        if self.padding.0 > 0
            || self.padding.1 > 0
            || self.output_padding.0 > 0
            || self.output_padding.1 > 0
        {
            output = output.slice(&[
                (0, n),
                (0, c_out),
                (self.padding.0, self.padding.0 + h_out),
                (self.padding.1, self.padding.1 + w_out),
            ]);
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
                .clone()
                .unsqueeze(0)
                .unsqueeze(2)
                .unsqueeze(3)
                .expand(output.shape());
            output = &output + &bias_expanded;
        }

        output.into_dim4()
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_grouped(
        &self,
        input: &Tensor<T, Dim4>,
        n: usize,
        c_in: usize,
        c_out: usize,
        h: usize,
        w: usize,
        h_out: usize,
        w_out: usize,
        h_out_raw: usize,
        w_out_raw: usize,
        kh: usize,
        kw: usize,
    ) -> Tensor<T, Dim4> {
        let c_in_per_group = c_in / self.groups;
        let c_out_per_group = c_out / self.groups;

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

/// ConvTranspose2dのビルダー
pub struct ConvTranspose2dBuilder<T: FloatDType = f32> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    bias: bool,
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> ConvTranspose2dBuilder<T> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel_size must be positive"
        );

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            bias: true,
            _dtype: PhantomData,
        }
    }

    /// ストライドを設定
    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        assert!(
            stride.0 > 0 && stride.1 > 0,
            "stride must be positive in both dimensions"
        );
        self.stride = stride;
        self
    }

    /// パディングを設定
    pub fn padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// 出力パディングを設定
    pub fn output_padding(mut self, output_padding: (usize, usize)) -> Self {
        assert!(
            output_padding.0 < self.stride.0 && output_padding.1 < self.stride.1,
            "output_padding must be smaller than stride"
        );
        self.output_padding = output_padding;
        self
    }

    /// ダイレーションを設定
    pub fn dilation(mut self, dilation: (usize, usize)) -> Self {
        assert!(
            dilation.0 > 0 && dilation.1 > 0,
            "dilation must be positive in both dimensions"
        );
        self.dilation = dilation;
        self
    }

    /// グループ数を設定
    pub fn groups(mut self, groups: usize) -> Self {
        assert!(groups > 0, "groups must be positive");
        assert!(
            self.in_channels.is_multiple_of(groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            self.out_channels.is_multiple_of(groups),
            "out_channels must be divisible by groups"
        );
        self.groups = groups;
        self
    }

    /// バイアスの有無を設定
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// ConvTranspose2dを構築
    pub fn build(self) -> ConvTranspose2d<T> {
        let c_out_per_group = self.out_channels / self.groups;

        // 重みの初期化
        let weight = Tensor::<T, Dim4>::rand([
            self.in_channels,
            c_out_per_group,
            self.kernel_size.0,
            self.kernel_size.1,
        ])
        .into_dyn();

        let bias = if self.bias {
            Some(Parameter::new(
                Tensor::<T, Dim1>::zeros([self.out_channels]).into_dyn(),
            ))
        } else {
            None
        };

        ConvTranspose2d {
            weight: Parameter::new(weight),
            bias,
            stride: self.stride,
            padding: self.padding,
            output_padding: self.output_padding,
            dilation: self.dilation,
            groups: self.groups,
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
    weight: Parameter<T>,
    /// バイアス [C_out]
    bias: Option<Parameter<T>>,
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
    /// ConvTranspose3dBuilder を作成
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
    ) -> ConvTranspose3dBuilder<T> {
        ConvTranspose3dBuilder::new(in_channels, out_channels, kernel_size)
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

        if self.groups == 1 {
            self.forward_no_groups(
                input, n, c_in, c_out, d, h, w, d_out, h_out, w_out, d_out_raw, h_out_raw,
                w_out_raw, kd, kh, kw,
            )
        } else {
            self.forward_grouped(
                input, n, c_in, c_out, d, h, w, d_out, h_out, w_out, d_out_raw, h_out_raw,
                w_out_raw, kd, kh, kw,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_no_groups(
        &self,
        input: &Tensor<T, Dim5>,
        n: usize,
        c_in: usize,
        c_out: usize,
        d: usize,
        h: usize,
        w: usize,
        d_out: usize,
        h_out: usize,
        w_out: usize,
        d_out_raw: usize,
        h_out_raw: usize,
        w_out_raw: usize,
        kd: usize,
        kh: usize,
        kw: usize,
    ) -> Tensor<T, Dim5> {
        // Input: [N, C_in, D, H, W] -> [N, D, H, W, C_in] -> [N * D * H * W, C_in]
        let input_reshaped = input
            .permute(&[0, 2, 3, 4, 1])
            .reshape([n * d * h * w, c_in]);

        // Weight: [C_in, C_out, kD, kH, kW] -> [C_in, C_out * kD * kH * kW]
        let weight_reshaped = self
            .weight
            .clone()
            .reshape_dyn(&[c_in, c_out * kd * kh * kw])
            .into_dim2();

        // Matmul: [N * D * H * W, C_in] @ [C_in, C_out * kD * kH * kW] -> [N * D * H * W, C_out * kD * kH * kW]
        let output = input_reshaped.matmul2(&weight_reshaped);

        // Reshape: -> [N, D, H, W, C_out, kD, kH, kW] -> [N, C_out, D, H, W, kD, kH, kW]
        let output = output
            .into_dyn()
            .reshape_dyn(&[n, d, h, w, c_out, kd, kh, kw])
            .permute(&[0, 4, 1, 2, 3, 5, 6, 7])
            .into_dim8();

        // Fold: [N, C_out, D, H, W, kD, kH, kW] -> [N, C_out, D_out_raw, H_out_raw, W_out_raw]
        let output = output.fold3d_dilated(
            (d_out_raw, h_out_raw, w_out_raw),
            self.stride,
            self.dilation,
        );

        // パディングを除去して最終サイズに調整
        let mut output = output.into_dyn();
        if self.padding.0 > 0
            || self.padding.1 > 0
            || self.padding.2 > 0
            || self.output_padding.0 > 0
            || self.output_padding.1 > 0
            || self.output_padding.2 > 0
        {
            output = output.slice(&[
                (0, n),
                (0, c_out),
                (self.padding.0, self.padding.0 + d_out),
                (self.padding.1, self.padding.1 + h_out),
                (self.padding.2, self.padding.2 + w_out),
            ]);
        }

        // Add bias if present
        if let Some(ref bias) = self.bias {
            let bias_expanded = bias
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

    #[allow(clippy::too_many_arguments)]
    fn forward_grouped(
        &self,
        input: &Tensor<T, Dim5>,
        n: usize,
        c_in: usize,
        c_out: usize,
        d: usize,
        h: usize,
        w: usize,
        d_out: usize,
        h_out: usize,
        w_out: usize,
        d_out_raw: usize,
        h_out_raw: usize,
        w_out_raw: usize,
        kd: usize,
        kh: usize,
        kw: usize,
    ) -> Tensor<T, Dim5> {
        let c_in_per_group = c_in / self.groups;
        let c_out_per_group = c_out / self.groups;

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

/// ConvTranspose3dのビルダー
pub struct ConvTranspose3dBuilder<T: FloatDType = f32> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
    groups: usize,
    bias: bool,
    _dtype: PhantomData<T>,
}

impl<T: FloatDType> ConvTranspose3dBuilder<T> {
    fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize, usize)) -> Self {
        assert!(in_channels > 0, "in_channels must be positive");
        assert!(out_channels > 0, "out_channels must be positive");
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0 && kernel_size.2 > 0,
            "kernel_size must be positive in all dimensions"
        );

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            dilation: (1, 1, 1),
            groups: 1,
            bias: true,
            _dtype: PhantomData,
        }
    }

    /// ストライドを設定
    pub fn stride(mut self, stride: (usize, usize, usize)) -> Self {
        assert!(
            stride.0 > 0 && stride.1 > 0 && stride.2 > 0,
            "stride must be positive in all dimensions"
        );
        self.stride = stride;
        self
    }

    /// パディングを設定
    pub fn padding(mut self, padding: (usize, usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// 出力パディングを設定
    pub fn output_padding(mut self, output_padding: (usize, usize, usize)) -> Self {
        assert!(
            output_padding.0 < self.stride.0
                && output_padding.1 < self.stride.1
                && output_padding.2 < self.stride.2,
            "output_padding must be smaller than stride"
        );
        self.output_padding = output_padding;
        self
    }

    /// ダイレーションを設定
    pub fn dilation(mut self, dilation: (usize, usize, usize)) -> Self {
        assert!(
            dilation.0 > 0 && dilation.1 > 0 && dilation.2 > 0,
            "dilation must be positive in all dimensions"
        );
        self.dilation = dilation;
        self
    }

    /// グループ数を設定
    pub fn groups(mut self, groups: usize) -> Self {
        assert!(groups > 0, "groups must be positive");
        assert!(
            self.in_channels.is_multiple_of(groups),
            "in_channels must be divisible by groups"
        );
        assert!(
            self.out_channels.is_multiple_of(groups),
            "out_channels must be divisible by groups"
        );
        self.groups = groups;
        self
    }

    /// バイアスの有無を設定
    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// ConvTranspose3dを構築
    pub fn build(self) -> ConvTranspose3d<T> {
        let c_out_per_group = self.out_channels / self.groups;

        // 重みの初期化
        let weight = Tensor::<T, Dim5>::rand([
            self.in_channels,
            c_out_per_group,
            self.kernel_size.0,
            self.kernel_size.1,
            self.kernel_size.2,
        ])
        .into_dyn();

        let bias = if self.bias {
            Some(Parameter::new(
                Tensor::<T, Dim1>::zeros([self.out_channels]).into_dyn(),
            ))
        } else {
            None
        };

        ConvTranspose3d {
            weight: Parameter::new(weight),
            bias,
            stride: self.stride,
            padding: self.padding,
            output_padding: self.output_padding,
            dilation: self.dilation,
            groups: self.groups,
            _dtype: PhantomData,
        }
    }
}

// ============================================================================
// Module implementations
// ============================================================================

impl<T: FloatDType> Module<T> for Conv1d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut Parameter<T>> {
        let mut params = std::collections::HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Parameter<T>>) {
        if let Some(w) = params.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = params.get("bias") {
            self.bias = Some(b.clone());
        }
    }
}

impl<T: FloatDType> Module<T> for Conv2d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut Parameter<T>> {
        let mut params = std::collections::HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Parameter<T>>) {
        if let Some(w) = params.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = params.get("bias") {
            self.bias = Some(b.clone());
        }
    }
}

impl<T: FloatDType> Module<T> for Conv3d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut Parameter<T>> {
        let mut params = std::collections::HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Parameter<T>>) {
        if let Some(w) = params.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = params.get("bias") {
            self.bias = Some(b.clone());
        }
    }
}

impl<T: FloatDType> Module<T> for ConvTranspose1d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut Parameter<T>> {
        let mut params = std::collections::HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Parameter<T>>) {
        if let Some(w) = params.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = params.get("bias") {
            self.bias = Some(b.clone());
        }
    }
}

impl<T: FloatDType> Module<T> for ConvTranspose2d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut Parameter<T>> {
        let mut params = std::collections::HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Parameter<T>>) {
        if let Some(w) = params.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = params.get("bias") {
            self.bias = Some(b.clone());
        }
    }
}

impl<T: FloatDType> Module<T> for ConvTranspose3d<T> {
    fn parameters(&mut self) -> std::collections::HashMap<String, &mut Parameter<T>> {
        let mut params = std::collections::HashMap::new();
        params.insert("weight".to_string(), &mut self.weight);
        if let Some(ref mut bias) = self.bias {
            params.insert("bias".to_string(), bias);
        }
        params
    }

    fn load_parameters(&mut self, params: std::collections::HashMap<String, Parameter<T>>) {
        if let Some(w) = params.get("weight") {
            self.weight = w.clone();
        }
        if let Some(b) = params.get("bias") {
            self.bias = Some(b.clone());
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

    // ConvTranspose1d tests
    #[test]
    fn test_conv_transpose1d_creation() {
        let conv = ConvTranspose1d::<f32>::new(3, 64, 3).build();
        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 64);
        assert_eq!(conv.kernel_size(), 3);
    }

    #[test]
    fn test_conv_transpose1d_forward_shape() {
        let conv = ConvTranspose1d::<f32>::new(3, 64, 3)
            .stride(2)
            .padding(1)
            .build();
        // Input: [batch, channels, length]
        // output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding
        // = (8 - 1) * 2 - 2 * 1 + 3 + 0 = 7 * 2 - 2 + 3 = 14 - 2 + 3 = 15
        let input = Tensor::<f32, Dim3>::rand([2, 3, 8]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 15]);
    }

    #[test]
    fn test_conv_transpose1d_parameters() {
        let mut conv = ConvTranspose1d::<f32>::new(3, 64, 3).build();
        let params = conv.parameters();
        // weight + bias
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_conv_transpose1d_no_bias() {
        let mut conv = ConvTranspose1d::<f32>::new(3, 64, 3).bias(false).build();
        let params = conv.parameters();
        // weight only
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_conv_transpose1d_grouped() {
        let conv = ConvTranspose1d::<f32>::new(8, 16, 3)
            .groups(4)
            .stride(2)
            .padding(1)
            .build();
        let input = Tensor::<f32, Dim3>::rand([2, 8, 8]);
        let output = conv.forward(&input);
        // output_length = (8 - 1) * 2 - 2 * 1 + 3 + 0 = 15
        assert_eq!(output.shape(), &[2, 16, 15]);
    }

    // ConvTranspose2d tests
    #[test]
    fn test_conv_transpose2d_creation() {
        let conv = ConvTranspose2d::<f32>::new(3, 64, (3, 3)).build();
        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 64);
        assert_eq!(conv.kernel_size(), (3, 3));
    }

    #[test]
    fn test_conv_transpose2d_forward_shape() {
        let conv = ConvTranspose2d::<f32>::new(3, 64, (3, 3))
            .stride((2, 2))
            .padding((1, 1))
            .build();
        // Input: [batch, channels, height, width]
        // output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
        // = (8 - 1) * 2 - 2 * 1 + 3 + 0 = 15
        let input = Tensor::<f32, Dim4>::rand([2, 3, 8, 8]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 15, 15]);
    }

    #[test]
    fn test_conv_transpose2d_parameters() {
        let mut conv = ConvTranspose2d::<f32>::new(3, 64, (3, 3)).build();
        let params = conv.parameters();
        // weight + bias
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_conv_transpose2d_no_bias() {
        let mut conv = ConvTranspose2d::<f32>::new(3, 64, (3, 3))
            .bias(false)
            .build();
        let params = conv.parameters();
        // weight only
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn test_conv_transpose2d_grouped() {
        let conv = ConvTranspose2d::<f32>::new(8, 16, (3, 3))
            .groups(4)
            .stride((2, 2))
            .padding((1, 1))
            .build();
        let input = Tensor::<f32, Dim4>::rand([2, 8, 8, 8]);
        let output = conv.forward(&input);
        // output_size = (8 - 1) * 2 - 2 * 1 + 3 + 0 = 15
        assert_eq!(output.shape(), &[2, 16, 15, 15]);
    }

    #[test]
    fn test_conv_transpose2d_output_padding() {
        let conv = ConvTranspose2d::<f32>::new(3, 64, (3, 3))
            .stride((2, 2))
            .padding((1, 1))
            .output_padding((1, 1))
            .build();
        // output_size = (8 - 1) * 2 - 2 * 1 + 3 + 1 = 16
        let input = Tensor::<f32, Dim4>::rand([2, 3, 8, 8]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 16, 16]);
    }

    // ConvTranspose3d tests
    #[test]
    fn test_conv_transpose3d_creation() {
        let conv = ConvTranspose3d::<f32>::new(3, 64, (3, 3, 3)).build();
        assert_eq!(conv.in_channels(), 3);
        assert_eq!(conv.out_channels(), 64);
        assert_eq!(conv.kernel_size(), (3, 3, 3));
    }

    #[test]
    fn test_conv_transpose3d_forward_shape() {
        let conv = ConvTranspose3d::<f32>::new(3, 64, (3, 3, 3))
            .stride((2, 2, 2))
            .padding((1, 1, 1))
            .build();
        // output_size = (8 - 1) * 2 - 2 * 1 + 3 + 0 = 15
        let input = Tensor::<f32, Dim5>::rand([2, 3, 8, 8, 8]);
        let output = conv.forward(&input);
        assert_eq!(output.shape(), &[2, 64, 15, 15, 15]);
    }

    #[test]
    fn test_conv_transpose3d_grouped() {
        let conv = ConvTranspose3d::<f32>::new(8, 16, (3, 3, 3))
            .groups(4)
            .stride((2, 2, 2))
            .padding((1, 1, 1))
            .build();
        let input = Tensor::<f32, Dim5>::rand([2, 8, 8, 8, 8]);
        let output = conv.forward(&input);
        // output_size = (8 - 1) * 2 - 2 * 1 + 3 + 0 = 15
        assert_eq!(output.shape(), &[2, 16, 15, 15, 15]);
    }

    // Gradient propagation tests
    #[test]
    fn test_conv2d_gradient() {
        let conv = Conv2d::<f32>::new(3, 8, (3, 3)).padding((1, 1)).build();
        let input = Tensor::<f32, Dim4>::rand([1, 3, 8, 8]).set_requires_grad(true);
        let output = conv.forward(&input);
        output.backward();

        // Input should have gradients
        assert!(input.grad().is_some());
    }

    #[test]
    fn test_conv_transpose2d_gradient() {
        let conv = ConvTranspose2d::<f32>::new(3, 8, (3, 3))
            .stride((2, 2))
            .padding((1, 1))
            .build();
        let input = Tensor::<f32, Dim4>::rand([1, 3, 4, 4]).set_requires_grad(true);
        let output = conv.forward(&input);
        output.backward();

        // Input should have gradients
        assert!(input.grad().is_some());
    }
}

//! 畳み込み層（Convolution Layers）
//!
//! 1D, 2D, 3D畳み込み層の実装を提供します。

use std::marker::PhantomData;

use harp::tensor::{Dim3, Dim4, Dim5, DimDyn, FloatDType, Tensor};

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

        if self.groups == 1 {
            // グループなし: 通常の畳み込み
            self.forward_no_groups(&padded, n, c_in, c_out, l_out, k)
        } else {
            // グループ畳み込み
            self.forward_grouped(&padded, n, c_in, c_out, l_out, k)
        }
    }

    fn forward_no_groups(
        &self,
        input: &Tensor<T, Dim3>,
        n: usize,
        c_in: usize,
        c_out: usize,
        l_out: usize,
        k: usize,
    ) -> Tensor<T, Dim3> {
        // Unfold: [N, C_in, L] -> [N, C_in, L_out, k]
        let unfolded = input.unfold1d_dilated(k, self.stride, self.dilation);

        // Reshape unfolded: [N, C_in, L_out, k] -> [N, L_out, C_in * k]
        let unfolded_reshaped = unfolded
            .permute(&[0, 2, 1, 3])
            .reshape([n, l_out, c_in * k]);

        // Reshape weight: [C_out, C_in, k] -> [C_in * k, C_out]
        let weight_reshaped = self
            .weight
            .clone()
            .permute(&[1, 2, 0])
            .reshape_dyn(&[c_in * k, c_out])
            .into_dim2();

        // Matmul: [N, L_out, C_in * k] @ [C_in * k, C_out] -> [N, L_out, C_out]
        let mut output = unfolded_reshaped
            .into_dyn()
            .reshape_dyn(&[n * l_out, c_in * k])
            .into_dim2()
            .matmul2(&weight_reshaped)
            .into_dyn()
            .reshape_dyn(&[n, l_out, c_out]);

        // Permute to [N, C_out, L_out]
        output = output.permute(&[0, 2, 1]);

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

    fn forward_grouped(
        &self,
        input: &Tensor<T, Dim3>,
        n: usize,
        c_in: usize,
        c_out: usize,
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
                    (0, input.shape()[2]),
                ])
                .contiguous();

            // 重みをスライス
            let weight_slice = self
                .weight
                .slice(&[
                    (g * c_out_per_group, (g + 1) * c_out_per_group),
                    (0, c_in_per_group),
                    (0, k),
                ])
                .contiguous();

            // Unfold
            let unfolded = input_slice.unfold1d_dilated(k, self.stride, self.dilation);

            // Reshape and matmul
            let unfolded_reshaped =
                unfolded
                    .permute(&[0, 2, 1, 3])
                    .reshape([n, l_out, c_in_per_group * k]);

            let weight_reshaped = weight_slice
                .permute(&[1, 2, 0])
                .reshape_dyn(&[c_in_per_group * k, c_out_per_group])
                .into_dim2();

            let group_output = unfolded_reshaped
                .into_dyn()
                .reshape_dyn(&[n * l_out, c_in_per_group * k])
                .into_dim2()
                .matmul2(&weight_reshaped)
                .into_dyn()
                .reshape_dyn(&[n, l_out, c_out_per_group])
                .permute(&[0, 2, 1]);

            outputs.push(group_output);
        }

        // Concatenate along channel dimension
        let output = Tensor::concat(&outputs.iter().collect::<Vec<_>>(), 1);

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

        // 重みの初期化: He initialization
        let weight =
            Tensor::<T, DimDyn>::rand_dyn(&[self.out_channels, c_in_per_group, self.kernel_size]);

        let bias = if self.bias {
            Some(Parameter::new(Tensor::<T, DimDyn>::zeros_dyn(&[
                self.out_channels
            ])))
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

        if self.groups == 1 {
            self.forward_no_groups(&padded, n, c_in, c_out, h_out, w_out, kh, kw)
        } else {
            self.forward_grouped(&padded, n, c_in, c_out, h_out, w_out, kh, kw)
        }
    }

    fn forward_no_groups(
        &self,
        input: &Tensor<T, Dim4>,
        n: usize,
        c_in: usize,
        c_out: usize,
        h_out: usize,
        w_out: usize,
        kh: usize,
        kw: usize,
    ) -> Tensor<T, Dim4> {
        // Unfold: [N, C_in, H, W] -> [N, C_in, H_out, W_out, kH, kW]
        let unfolded = input.unfold2d_dilated((kh, kw), self.stride, self.dilation);

        // Reshape unfolded: [N, C_in, H_out, W_out, kH, kW] -> [N * H_out * W_out, C_in * kH * kW]
        let unfolded_reshaped = unfolded
            .permute(&[0, 2, 3, 1, 4, 5])
            .reshape([n * h_out * w_out, c_in * kh * kw]);

        // Reshape weight: [C_out, C_in, kH, kW] -> [C_in * kH * kW, C_out]
        let weight_reshaped = self
            .weight
            .clone()
            .permute(&[1, 2, 3, 0])
            .reshape_dyn(&[c_in * kh * kw, c_out])
            .into_dim2();

        // Matmul: [N * H_out * W_out, C_in * kH * kW] @ [C_in * kH * kW, C_out]
        //       -> [N * H_out * W_out, C_out]
        let output = unfolded_reshaped.matmul2(&weight_reshaped).into_dyn();

        // Reshape to [N, H_out, W_out, C_out] then permute to [N, C_out, H_out, W_out]
        let mut output = output
            .reshape_dyn(&[n, h_out, w_out, c_out])
            .permute(&[0, 3, 1, 2]);

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

    fn forward_grouped(
        &self,
        input: &Tensor<T, Dim4>,
        n: usize,
        c_in: usize,
        c_out: usize,
        h_out: usize,
        w_out: usize,
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
                    (0, input.shape()[2]),
                    (0, input.shape()[3]),
                ])
                .contiguous();

            // 重みをスライス
            let weight_slice = self
                .weight
                .slice(&[
                    (g * c_out_per_group, (g + 1) * c_out_per_group),
                    (0, c_in_per_group),
                    (0, kh),
                    (0, kw),
                ])
                .contiguous();

            // Unfold
            let unfolded = input_slice.unfold2d_dilated((kh, kw), self.stride, self.dilation);

            // Reshape and matmul
            let unfolded_reshaped = unfolded
                .permute(&[0, 2, 3, 1, 4, 5])
                .reshape([n * h_out * w_out, c_in_per_group * kh * kw]);

            let weight_reshaped = weight_slice
                .permute(&[1, 2, 3, 0])
                .reshape_dyn(&[c_in_per_group * kh * kw, c_out_per_group])
                .into_dim2();

            let group_output = unfolded_reshaped
                .matmul2(&weight_reshaped)
                .into_dyn()
                .reshape_dyn(&[n, h_out, w_out, c_out_per_group])
                .permute(&[0, 3, 1, 2]);

            outputs.push(group_output);
        }

        // Concatenate along channel dimension
        let output = Tensor::concat(&outputs.iter().collect::<Vec<_>>(), 1);

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
        let weight = Tensor::<T, DimDyn>::rand_dyn(&[
            self.out_channels,
            c_in_per_group,
            self.kernel_size.0,
            self.kernel_size.1,
        ]);

        let bias = if self.bias {
            Some(Parameter::new(Tensor::<T, DimDyn>::zeros_dyn(&[
                self.out_channels
            ])))
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

        if self.groups == 1 {
            self.forward_no_groups(&padded, n, c_in, c_out, d_out, h_out, w_out, kd, kh, kw)
        } else {
            self.forward_grouped(&padded, n, c_in, c_out, d_out, h_out, w_out, kd, kh, kw)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_no_groups(
        &self,
        input: &Tensor<T, Dim5>,
        n: usize,
        c_in: usize,
        c_out: usize,
        d_out: usize,
        h_out: usize,
        w_out: usize,
        kd: usize,
        kh: usize,
        kw: usize,
    ) -> Tensor<T, Dim5> {
        // Unfold: [N, C_in, D, H, W] -> [N, C_in, D_out, H_out, W_out, kD, kH, kW]
        let unfolded = input.unfold3d_dilated((kd, kh, kw), self.stride, self.dilation);

        // Reshape unfolded: [N, C_in, D_out, H_out, W_out, kD, kH, kW]
        //                -> [N * D_out * H_out * W_out, C_in * kD * kH * kW]
        let unfolded_reshaped = unfolded
            .permute(&[0, 2, 3, 4, 1, 5, 6, 7])
            .reshape([n * d_out * h_out * w_out, c_in * kd * kh * kw]);

        // Reshape weight: [C_out, C_in, kD, kH, kW] -> [C_in * kD * kH * kW, C_out]
        let weight_reshaped = self
            .weight
            .clone()
            .permute(&[1, 2, 3, 4, 0])
            .reshape_dyn(&[c_in * kd * kh * kw, c_out])
            .into_dim2();

        // Matmul
        let output = unfolded_reshaped.matmul2(&weight_reshaped).into_dyn();

        // Reshape to [N, D_out, H_out, W_out, C_out] then permute to [N, C_out, D_out, H_out, W_out]
        let mut output = output
            .reshape_dyn(&[n, d_out, h_out, w_out, c_out])
            .permute(&[0, 4, 1, 2, 3]);

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
        d_out: usize,
        h_out: usize,
        w_out: usize,
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
                    (0, input.shape()[2]),
                    (0, input.shape()[3]),
                    (0, input.shape()[4]),
                ])
                .contiguous();

            // 重みをスライス
            let weight_slice = self
                .weight
                .slice(&[
                    (g * c_out_per_group, (g + 1) * c_out_per_group),
                    (0, c_in_per_group),
                    (0, kd),
                    (0, kh),
                    (0, kw),
                ])
                .contiguous();

            // Unfold
            let unfolded = input_slice.unfold3d_dilated((kd, kh, kw), self.stride, self.dilation);

            // Reshape and matmul
            let unfolded_reshaped = unfolded
                .permute(&[0, 2, 3, 4, 1, 5, 6, 7])
                .reshape([n * d_out * h_out * w_out, c_in_per_group * kd * kh * kw]);

            let weight_reshaped = weight_slice
                .permute(&[1, 2, 3, 4, 0])
                .reshape_dyn(&[c_in_per_group * kd * kh * kw, c_out_per_group])
                .into_dim2();

            let group_output = unfolded_reshaped
                .matmul2(&weight_reshaped)
                .into_dyn()
                .reshape_dyn(&[n, d_out, h_out, w_out, c_out_per_group])
                .permute(&[0, 4, 1, 2, 3]);

            outputs.push(group_output);
        }

        // Concatenate along channel dimension
        let output = Tensor::concat(&outputs.iter().collect::<Vec<_>>(), 1);

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
        let weight = Tensor::<T, DimDyn>::rand_dyn(&[
            self.out_channels,
            c_in_per_group,
            self.kernel_size.0,
            self.kernel_size.1,
            self.kernel_size.2,
        ]);

        let bias = if self.bias {
            Some(Parameter::new(Tensor::<T, DimDyn>::zeros_dyn(&[
                self.out_channels
            ])))
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

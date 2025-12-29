//! Unfold (sliding window) operations for convolution preprocessing

use std::marker::PhantomData;
use std::sync::Arc;

use super::super::binary::with_grad_fn_generic;
use super::backward::{Unfold1dBackward, Unfold2dBackward, Unfold3dBackward};
use crate::tensor::shape::Expr;
use crate::tensor::{
    Dim3, Dim4, Dim5, Dim6, Dim8, FloatDType, GradFn, Tensor, TensorInner, TensorOp,
};

impl<T: FloatDType> Tensor<T, Dim3> {
    /// 1D unfold (sliding window) - [N, C, L] → [N, C, out_L, k]
    ///
    /// Extracts sliding windows along the last dimension.
    /// This is useful for im2col in 1D convolution.
    ///
    /// # Arguments
    /// * `size` - Window size (kernel size)
    /// * `stride` - Stride for sliding window
    ///
    /// # Output Shape
    /// `[N, C, (L - size) / stride + 1, size]`
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim3>::ones([2, 3, 10]);
    /// let unfolded = input.unfold1d(3, 1);
    /// assert_eq!(unfolded.shape(), &[2, 3, 8, 3]);
    /// ```
    pub fn unfold1d(&self, size: usize, stride: usize) -> Tensor<T, Dim4> {
        self.unfold1d_dilated(size, stride, 1)
    }

    /// 1D unfold with dilation - [N, C, L] → [N, C, out_L, k]
    ///
    /// Extracts sliding windows along the last dimension with dilation support.
    ///
    /// # Arguments
    /// * `size` - Window size (kernel size)
    /// * `stride` - Stride for sliding window
    /// * `dilation` - Spacing between kernel elements
    ///
    /// # Output Shape
    /// `[N, C, (L - effective_size) / stride + 1, size]`
    /// where `effective_size = (size - 1) * dilation + 1`
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim3>::ones([2, 3, 10]);
    /// // dilation=2: kernel elements are 2 apart, effective size = (3-1)*2+1 = 5
    /// let unfolded = input.unfold1d_dilated(3, 1, 2);
    /// assert_eq!(unfolded.shape(), &[2, 3, 6, 3]); // (10-5)/1+1 = 6
    /// ```
    pub fn unfold1d_dilated(&self, size: usize, stride: usize, dilation: usize) -> Tensor<T, Dim4> {
        assert!(size > 0, "size must be positive");
        assert!(stride > 0, "stride must be positive");
        assert!(dilation > 0, "dilation must be positive");

        let l = self.shape()[2];
        let effective_size = (size - 1) * dilation + 1;
        assert!(
            l >= effective_size,
            "input size {} must be >= effective window size {}",
            l,
            effective_size
        );

        // Calculate output shape
        let out_l = (l - effective_size) / stride + 1;
        let new_shape = vec![self.shape()[0], self.shape()[1], out_l, size];

        // Apply unfold to view
        let new_view = self.inner.view.clone().unfold(
            &[2],
            &[Expr::from(size as i64)],
            &[Expr::from(stride as i64)],
            &[Expr::from(dilation as i64)],
        );

        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, new_view, new_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Register gradient function
        let grad_fn = if self.requires_grad() {
            Some(Arc::new(Unfold1dBackward::new(
                self.clone().into_dyn(),
                l, // output_size (original L)
                size,
                stride,
                dilation,
            )) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }
}

impl<T: FloatDType> Tensor<T, Dim4> {
    /// 2D unfold (sliding window) - [N, C, H, W] → [N, C, out_H, out_W, kH, kW]
    ///
    /// Extracts sliding windows along the spatial dimensions (H, W).
    /// This is useful for im2col in 2D convolution.
    ///
    /// # Arguments
    /// * `sizes` - Window sizes (kH, kW)
    /// * `strides` - Strides for sliding window (stride_h, stride_w)
    ///
    /// # Output Shape
    /// `[N, C, (H - kH) / stride_h + 1, (W - kW) / stride_w + 1, kH, kW]`
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim4>::ones([2, 3, 28, 28]);
    /// let unfolded = input.unfold2d((3, 3), (1, 1));
    /// assert_eq!(unfolded.shape(), &[2, 3, 26, 26, 3, 3]);
    /// ```
    pub fn unfold2d(&self, sizes: (usize, usize), strides: (usize, usize)) -> Tensor<T, Dim6> {
        self.unfold2d_dilated(sizes, strides, (1, 1))
    }

    /// 2D unfold with dilation - [N, C, H, W] → [N, C, out_H, out_W, kH, kW]
    ///
    /// Extracts sliding windows along the spatial dimensions with dilation support.
    ///
    /// # Arguments
    /// * `sizes` - Window sizes (kH, kW)
    /// * `strides` - Strides for sliding window (stride_h, stride_w)
    /// * `dilations` - Spacing between kernel elements (dilation_h, dilation_w)
    ///
    /// # Output Shape
    /// `[N, C, (H - eff_kH) / stride_h + 1, (W - eff_kW) / stride_w + 1, kH, kW]`
    /// where `eff_k = (k - 1) * dilation + 1`
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim4>::ones([2, 3, 28, 28]);
    /// // dilation=2: effective kernel size = (3-1)*2+1 = 5
    /// let unfolded = input.unfold2d_dilated((3, 3), (1, 1), (2, 2));
    /// assert_eq!(unfolded.shape(), &[2, 3, 24, 24, 3, 3]); // (28-5)/1+1 = 24
    /// ```
    pub fn unfold2d_dilated(
        &self,
        sizes: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Tensor<T, Dim6> {
        let (kh, kw) = sizes;
        let (sh, sw) = strides;
        let (dh, dw) = dilations;

        assert!(kh > 0 && kw > 0, "sizes must be positive");
        assert!(sh > 0 && sw > 0, "strides must be positive");
        assert!(dh > 0 && dw > 0, "dilations must be positive");

        let h = self.shape()[2];
        let w = self.shape()[3];

        // Calculate effective kernel sizes
        let eff_kh = (kh - 1) * dh + 1;
        let eff_kw = (kw - 1) * dw + 1;

        assert!(
            h >= eff_kh,
            "input height {} must be >= effective kernel height {}",
            h,
            eff_kh
        );
        assert!(
            w >= eff_kw,
            "input width {} must be >= effective kernel width {}",
            w,
            eff_kw
        );

        // Calculate output shape
        let out_h = (h - eff_kh) / sh + 1;
        let out_w = (w - eff_kw) / sw + 1;
        let new_shape = vec![self.shape()[0], self.shape()[1], out_h, out_w, kh, kw];

        // Apply unfold to view
        let new_view = self.inner.view.clone().unfold(
            &[2, 3],
            &[Expr::from(kh as i64), Expr::from(kw as i64)],
            &[Expr::from(sh as i64), Expr::from(sw as i64)],
            &[Expr::from(dh as i64), Expr::from(dw as i64)],
        );

        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, new_view, new_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Register gradient function
        let grad_fn = if self.requires_grad() {
            Some(Arc::new(Unfold2dBackward::new(
                self.clone().into_dyn(),
                (h, w), // output_size (original H, W)
                (kh, kw),
                (sh, sw),
                (dh, dw),
            )) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }
}

impl<T: FloatDType> Tensor<T, Dim5> {
    /// 3D unfold (sliding window) - [N, C, H, W, D] → [N, C, out_H, out_W, out_D, kH, kW, kD]
    ///
    /// Extracts sliding windows along the spatial dimensions (H, W, D).
    /// This is useful for im2col in 3D convolution.
    ///
    /// # Arguments
    /// * `sizes` - Window sizes (kH, kW, kD)
    /// * `strides` - Strides for sliding window (stride_h, stride_w, stride_d)
    ///
    /// # Output Shape
    /// `[N, C, (H - kH) / sH + 1, (W - kW) / sW + 1, (D - kD) / sD + 1, kH, kW, kD]`
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim5>::ones([2, 3, 16, 16, 16]);
    /// let unfolded = input.unfold3d((3, 3, 3), (1, 1, 1));
    /// assert_eq!(unfolded.shape(), &[2, 3, 14, 14, 14, 3, 3, 3]);
    /// ```
    pub fn unfold3d(
        &self,
        sizes: (usize, usize, usize),
        strides: (usize, usize, usize),
    ) -> Tensor<T, Dim8> {
        self.unfold3d_dilated(sizes, strides, (1, 1, 1))
    }

    /// 3D unfold with dilation - [N, C, H, W, D] → [N, C, out_H, out_W, out_D, kH, kW, kD]
    ///
    /// Extracts sliding windows along the spatial dimensions with dilation support.
    ///
    /// # Arguments
    /// * `sizes` - Window sizes (kH, kW, kD)
    /// * `strides` - Strides for sliding window (stride_h, stride_w, stride_d)
    /// * `dilations` - Spacing between kernel elements (dilation_h, dilation_w, dilation_d)
    ///
    /// # Output Shape
    /// `[N, C, (H - eff_kH) / sH + 1, (W - eff_kW) / sW + 1, (D - eff_kD) / sD + 1, kH, kW, kD]`
    /// where `eff_k = (k - 1) * dilation + 1`
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim5>::ones([2, 3, 16, 16, 16]);
    /// // dilation=2: effective kernel size = (3-1)*2+1 = 5
    /// let unfolded = input.unfold3d_dilated((3, 3, 3), (1, 1, 1), (2, 2, 2));
    /// assert_eq!(unfolded.shape(), &[2, 3, 12, 12, 12, 3, 3, 3]); // (16-5)/1+1 = 12
    /// ```
    pub fn unfold3d_dilated(
        &self,
        sizes: (usize, usize, usize),
        strides: (usize, usize, usize),
        dilations: (usize, usize, usize),
    ) -> Tensor<T, Dim8> {
        let (kh, kw, kd) = sizes;
        let (sh, sw, sd) = strides;
        let (dil_h, dil_w, dil_d) = dilations;

        assert!(kh > 0 && kw > 0 && kd > 0, "sizes must be positive");
        assert!(sh > 0 && sw > 0 && sd > 0, "strides must be positive");
        assert!(
            dil_h > 0 && dil_w > 0 && dil_d > 0,
            "dilations must be positive"
        );

        let h = self.shape()[2];
        let w = self.shape()[3];
        let d = self.shape()[4];

        // Calculate effective kernel sizes
        let eff_kh = (kh - 1) * dil_h + 1;
        let eff_kw = (kw - 1) * dil_w + 1;
        let eff_kd = (kd - 1) * dil_d + 1;

        assert!(
            h >= eff_kh,
            "input height {} must be >= effective kernel height {}",
            h,
            eff_kh
        );
        assert!(
            w >= eff_kw,
            "input width {} must be >= effective kernel width {}",
            w,
            eff_kw
        );
        assert!(
            d >= eff_kd,
            "input depth {} must be >= effective kernel depth {}",
            d,
            eff_kd
        );

        // Calculate output shape
        let out_h = (h - eff_kh) / sh + 1;
        let out_w = (w - eff_kw) / sw + 1;
        let out_d = (d - eff_kd) / sd + 1;
        let new_shape = vec![
            self.shape()[0],
            self.shape()[1],
            out_h,
            out_w,
            out_d,
            kh,
            kw,
            kd,
        ];

        // Apply unfold to view
        let new_view = self.inner.view.clone().unfold(
            &[2, 3, 4],
            &[
                Expr::from(kh as i64),
                Expr::from(kw as i64),
                Expr::from(kd as i64),
            ],
            &[
                Expr::from(sh as i64),
                Expr::from(sw as i64),
                Expr::from(sd as i64),
            ],
            &[
                Expr::from(dil_h as i64),
                Expr::from(dil_w as i64),
                Expr::from(dil_d as i64),
            ],
        );

        let input = self.as_input_ref();
        let inner = TensorInner::new(TensorOp::View { input }, new_view, new_shape, T::DTYPE);

        let result = Tensor {
            inner: Arc::new(inner),
            _dtype: PhantomData,
            _dim: PhantomData,
        };

        // Register gradient function
        let grad_fn = if self.requires_grad() {
            Some(Arc::new(Unfold3dBackward::new(
                self.clone().into_dyn(),
                (h, w, d), // output_size (original H, W, D)
                (kh, kw, kd),
                (sh, sw, sd),
                (dil_h, dil_w, dil_d),
            )) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }
}

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
        assert!(size > 0, "size must be positive");
        assert!(stride > 0, "stride must be positive");
        let l = self.shape()[2];
        assert!(
            l >= size,
            "input size {} must be >= window size {}",
            l,
            size
        );

        // Calculate output shape
        let out_l = (l - size) / stride + 1;
        let new_shape = vec![self.shape()[0], self.shape()[1], out_l, size];

        // Apply unfold to view
        let new_view = self.inner.view.clone().unfold(
            &[2],
            &[Expr::from(size as i64)],
            &[Expr::from(stride as i64)],
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
        let (kh, kw) = sizes;
        let (sh, sw) = strides;

        assert!(kh > 0 && kw > 0, "sizes must be positive");
        assert!(sh > 0 && sw > 0, "strides must be positive");

        let h = self.shape()[2];
        let w = self.shape()[3];

        assert!(
            h >= kh,
            "input height {} must be >= kernel height {}",
            h,
            kh
        );
        assert!(w >= kw, "input width {} must be >= kernel width {}", w, kw);

        // Calculate output shape
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let new_shape = vec![self.shape()[0], self.shape()[1], out_h, out_w, kh, kw];

        // Apply unfold to view
        let new_view = self.inner.view.clone().unfold(
            &[2, 3],
            &[Expr::from(kh as i64), Expr::from(kw as i64)],
            &[Expr::from(sh as i64), Expr::from(sw as i64)],
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
        let (kh, kw, kd) = sizes;
        let (sh, sw, sd) = strides;

        assert!(kh > 0 && kw > 0 && kd > 0, "sizes must be positive");
        assert!(sh > 0 && sw > 0 && sd > 0, "strides must be positive");

        let h = self.shape()[2];
        let w = self.shape()[3];
        let d = self.shape()[4];

        assert!(
            h >= kh,
            "input height {} must be >= kernel height {}",
            h,
            kh
        );
        assert!(w >= kw, "input width {} must be >= kernel width {}", w, kw);
        assert!(d >= kd, "input depth {} must be >= kernel depth {}", d, kd);

        // Calculate output shape
        let out_h = (h - kh) / sh + 1;
        let out_w = (w - kw) / sw + 1;
        let out_d = (d - kd) / sd + 1;
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
            )) as Arc<dyn GradFn<T>>)
        } else {
            None
        };

        with_grad_fn_generic(result, grad_fn)
    }
}

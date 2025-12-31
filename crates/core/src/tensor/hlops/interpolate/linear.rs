//! Linear interpolation operations
//!
//! Provides linear, bilinear, and trilinear interpolation for resizing tensors.
//! Composed from primops: arange, floor, cast, gather.
//!
//! Supported dimensions:
//! - 1D: `Dim3` (N, C, W) - linear1d
//! - 2D: `Dim4` (N, C, H, W) - bilinear2d
//! - 3D: `Dim5` (N, C, D, H, W) - trilinear3d

use crate::tensor::{Dim1, Dim3, Dim4, Dim5, DimDyn, FloatDType, Floor, Tensor};

// ============================================================================
// Helper: Generate indices and weights for linear interpolation
// ============================================================================

/// Generate gather indices and interpolation weights for one spatial dimension.
///
/// Returns (x0, x1, weight) where:
/// - x0: floor indices (left neighbor)
/// - x1: ceil indices (right neighbor, clamped to max)
/// - weight: interpolation weight (fractional part, 0.0 to 1.0)
fn make_linear_indices(
    out_size: usize,
    in_size: usize,
) -> (Tensor<f32, Dim1>, Tensor<f32, Dim1>, Tensor<f32, Dim1>) {
    let scale = in_size as f32 / out_size as f32;
    let coords = Tensor::<f32, Dim1>::arange(out_size);
    let scaled = &coords * scale;

    // Boundary-safe indices
    let max_idx = (in_size.saturating_sub(1)) as f32;
    let x0 = scaled.clamp(0.0, max_idx).floor();
    let x1 = (&x0 + 1.0).clamp(0.0, max_idx);

    // Weight (fractional part)
    let weight = &scaled - &x0;
    let weight = weight.clamp(0.0, 1.0);

    (x0, x1, weight)
}

// ============================================================================
// 1D Linear Interpolation (NCW format)
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim3> {
    /// Linear interpolation for 1D signals (NCW format)
    ///
    /// Resizes the spatial dimension (W) using linear interpolation.
    ///
    /// # Arguments
    /// * `size` - Target width
    ///
    /// # Returns
    /// Resized tensor with shape [N, C, size]
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim3>::ones([1, 3, 100]);
    /// let output = input.linear1d(200);
    /// assert_eq!(output.shape(), &[1, 3, 200]);
    /// ```
    pub fn linear1d(&self, size: usize) -> Self {
        let shape = self.shape();
        let (batch, channels, in_w) = (shape[0], shape[1], shape[2]);
        let out_w = size;

        // Generate indices and weights for W axis
        let (w0, w1, weight_w) = make_linear_indices(out_w, in_w);

        // Reshape weight to [1, 1, out_w] for broadcasting
        let weight_w_3d = weight_w.reshape_dyn(&[1, 1, out_w]);
        let weight_w_broadcast = weight_w_3d.expand(&[batch, channels, out_w]);

        // Prepare indices for gather
        let w0_3d = w0.reshape_dyn(&[1, 1, out_w]);
        let w0_broadcast = w0_3d.expand(&[batch, channels, out_w]);
        let w0_idx: Tensor<i64, DimDyn> = w0_broadcast.cast();

        let w1_3d = w1.reshape_dyn(&[1, 1, out_w]);
        let w1_broadcast = w1_3d.expand(&[batch, channels, out_w]);
        let w1_idx: Tensor<i64, DimDyn> = w1_broadcast.cast();

        // Gather values at both positions
        let self_dyn = self.clone().into_dyn();
        let v0 = self_dyn.gather(2, &w0_idx);
        let v1 = self_dyn.gather(2, &w1_idx);

        // Linear interpolation: v0 * (1 - weight) + v1 * weight
        let one_minus_w: Tensor<T, DimDyn> = (1.0 - &weight_w_broadcast).cast();
        let weight_t: Tensor<T, DimDyn> = weight_w_broadcast.cast();

        let result_dyn = &v0 * &one_minus_w + &v1 * &weight_t;

        // Convert back to Dim3
        result_dyn.reshape([batch, channels, out_w])
    }
}

// ============================================================================
// 2D Bilinear Interpolation (NCHW format)
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim4> {
    /// Bilinear interpolation for 2D images (NCHW format)
    ///
    /// Resizes the spatial dimensions (H, W) using bilinear interpolation.
    ///
    /// # Arguments
    /// * `size` - Target size (height, width)
    ///
    /// # Returns
    /// Resized tensor with shape [N, C, size.0, size.1]
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim4>::ones([1, 3, 4, 4]);
    /// let output = input.bilinear2d((8, 8));
    /// assert_eq!(output.shape(), &[1, 3, 8, 8]);
    /// ```
    pub fn bilinear2d(&self, size: (usize, usize)) -> Self {
        let shape = self.shape();
        let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = size;

        // Generate indices and weights for H and W axes
        let (h0, h1, weight_h) = make_linear_indices(out_h, in_h);
        let (w0, w1, weight_w) = make_linear_indices(out_w, in_w);

        // Reshape weights for broadcasting
        // weight_h: [1, 1, out_h, 1]
        // weight_w: [1, 1, 1, out_w]
        let weight_h_4d = weight_h.reshape_dyn(&[1, 1, out_h, 1]);
        let weight_h_broadcast = weight_h_4d.expand(&[batch, channels, out_h, out_w]);

        let weight_w_4d = weight_w.reshape_dyn(&[1, 1, 1, out_w]);
        let weight_w_broadcast = weight_w_4d.expand(&[batch, channels, out_h, out_w]);

        // Prepare H indices for first gather (shape: [batch, channels, out_h, in_w])
        let h0_4d = h0.reshape_dyn(&[1, 1, out_h, 1]);
        let h0_broadcast = h0_4d.expand(&[batch, channels, out_h, in_w]);
        let h0_idx: Tensor<i64, DimDyn> = h0_broadcast.cast();

        let h1_4d = h1.reshape_dyn(&[1, 1, out_h, 1]);
        let h1_broadcast = h1_4d.expand(&[batch, channels, out_h, in_w]);
        let h1_idx: Tensor<i64, DimDyn> = h1_broadcast.cast();

        // Gather along H axis (dim=2)
        let self_dyn = self.clone().into_dyn();
        let gathered_h0 = self_dyn.gather(2, &h0_idx); // [batch, channels, out_h, in_w]
        let gathered_h1 = self_dyn.gather(2, &h1_idx); // [batch, channels, out_h, in_w]

        // Prepare W indices for second gather (shape: [batch, channels, out_h, out_w])
        let w0_4d = w0.reshape_dyn(&[1, 1, 1, out_w]);
        let w0_broadcast = w0_4d.expand(&[batch, channels, out_h, out_w]);
        let w0_idx: Tensor<i64, DimDyn> = w0_broadcast.cast();

        let w1_4d = w1.reshape_dyn(&[1, 1, 1, out_w]);
        let w1_broadcast = w1_4d.expand(&[batch, channels, out_h, out_w]);
        let w1_idx: Tensor<i64, DimDyn> = w1_broadcast.cast();

        // Gather along W axis (dim=3) for all 4 corner values
        let v00 = gathered_h0.gather(3, &w0_idx); // top-left
        let v01 = gathered_h0.gather(3, &w1_idx); // top-right
        let v10 = gathered_h1.gather(3, &w0_idx); // bottom-left
        let v11 = gathered_h1.gather(3, &w1_idx); // bottom-right

        // Bilinear interpolation:
        // Interpolate along W first, then along H
        // top = v00 * (1 - wx) + v01 * wx
        // bottom = v10 * (1 - wx) + v11 * wx
        // result = top * (1 - wy) + bottom * wy

        let one_minus_wh: Tensor<T, DimDyn> = (1.0 - &weight_h_broadcast).cast();
        let one_minus_ww: Tensor<T, DimDyn> = (1.0 - &weight_w_broadcast).cast();
        let wh: Tensor<T, DimDyn> = weight_h_broadcast.cast();
        let ww: Tensor<T, DimDyn> = weight_w_broadcast.cast();

        // Interpolate along W
        let top = &v00 * &one_minus_ww + &v01 * &ww;
        let bottom = &v10 * &one_minus_ww + &v11 * &ww;

        // Interpolate along H
        let result_dyn = &top * &one_minus_wh + &bottom * &wh;

        // Convert back to Dim4
        result_dyn.reshape([batch, channels, out_h, out_w])
    }
}

// ============================================================================
// 3D Trilinear Interpolation (NCDHW format)
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim5> {
    /// Trilinear interpolation for 3D volumes (NCDHW format)
    ///
    /// Resizes the spatial dimensions (D, H, W) using trilinear interpolation.
    ///
    /// # Arguments
    /// * `size` - Target size (depth, height, width)
    ///
    /// # Returns
    /// Resized tensor with shape [N, C, size.0, size.1, size.2]
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim5>::ones([1, 3, 8, 8, 8]);
    /// let output = input.trilinear3d((16, 16, 16));
    /// assert_eq!(output.shape(), &[1, 3, 16, 16, 16]);
    /// ```
    pub fn trilinear3d(&self, size: (usize, usize, usize)) -> Self {
        let shape = self.shape();
        let (batch, channels, in_d, in_h, in_w) =
            (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let (out_d, out_h, out_w) = size;

        // Generate indices and weights for D, H, W axes
        let (d0, d1, weight_d) = make_linear_indices(out_d, in_d);
        let (h0, h1, weight_h) = make_linear_indices(out_h, in_h);
        let (w0, w1, weight_w) = make_linear_indices(out_w, in_w);

        // Reshape weights for broadcasting to [batch, channels, out_d, out_h, out_w]
        let weight_d_5d = weight_d.reshape_dyn(&[1, 1, out_d, 1, 1]);
        let weight_d_broadcast = weight_d_5d.expand(&[batch, channels, out_d, out_h, out_w]);

        let weight_h_5d = weight_h.reshape_dyn(&[1, 1, 1, out_h, 1]);
        let weight_h_broadcast = weight_h_5d.expand(&[batch, channels, out_d, out_h, out_w]);

        let weight_w_5d = weight_w.reshape_dyn(&[1, 1, 1, 1, out_w]);
        let weight_w_broadcast = weight_w_5d.expand(&[batch, channels, out_d, out_h, out_w]);

        // Step 1: Gather along D axis (dim=2)
        // D indices: [batch, channels, out_d, in_h, in_w]
        let d0_5d = d0.reshape_dyn(&[1, 1, out_d, 1, 1]);
        let d0_broadcast = d0_5d.expand(&[batch, channels, out_d, in_h, in_w]);
        let d0_idx: Tensor<i64, DimDyn> = d0_broadcast.cast();

        let d1_5d = d1.reshape_dyn(&[1, 1, out_d, 1, 1]);
        let d1_broadcast = d1_5d.expand(&[batch, channels, out_d, in_h, in_w]);
        let d1_idx: Tensor<i64, DimDyn> = d1_broadcast.cast();

        let self_dyn = self.clone().into_dyn();
        let gathered_d0 = self_dyn.gather(2, &d0_idx); // [batch, channels, out_d, in_h, in_w]
        let gathered_d1 = self_dyn.gather(2, &d1_idx);

        // Step 2: Gather along H axis (dim=3)
        // H indices: [batch, channels, out_d, out_h, in_w]
        let h0_5d = h0.reshape_dyn(&[1, 1, 1, out_h, 1]);
        let h0_broadcast = h0_5d.expand(&[batch, channels, out_d, out_h, in_w]);
        let h0_idx: Tensor<i64, DimDyn> = h0_broadcast.cast();

        let h1_5d = h1.reshape_dyn(&[1, 1, 1, out_h, 1]);
        let h1_broadcast = h1_5d.expand(&[batch, channels, out_d, out_h, in_w]);
        let h1_idx: Tensor<i64, DimDyn> = h1_broadcast.cast();

        let gathered_d0h0 = gathered_d0.gather(3, &h0_idx);
        let gathered_d0h1 = gathered_d0.gather(3, &h1_idx);
        let gathered_d1h0 = gathered_d1.gather(3, &h0_idx);
        let gathered_d1h1 = gathered_d1.gather(3, &h1_idx);

        // Step 3: Gather along W axis (dim=4)
        // W indices: [batch, channels, out_d, out_h, out_w]
        let w0_5d = w0.reshape_dyn(&[1, 1, 1, 1, out_w]);
        let w0_broadcast = w0_5d.expand(&[batch, channels, out_d, out_h, out_w]);
        let w0_idx: Tensor<i64, DimDyn> = w0_broadcast.cast();

        let w1_5d = w1.reshape_dyn(&[1, 1, 1, 1, out_w]);
        let w1_broadcast = w1_5d.expand(&[batch, channels, out_d, out_h, out_w]);
        let w1_idx: Tensor<i64, DimDyn> = w1_broadcast.cast();

        // All 8 corner values
        let v000 = gathered_d0h0.gather(4, &w0_idx);
        let v001 = gathered_d0h0.gather(4, &w1_idx);
        let v010 = gathered_d0h1.gather(4, &w0_idx);
        let v011 = gathered_d0h1.gather(4, &w1_idx);
        let v100 = gathered_d1h0.gather(4, &w0_idx);
        let v101 = gathered_d1h0.gather(4, &w1_idx);
        let v110 = gathered_d1h1.gather(4, &w0_idx);
        let v111 = gathered_d1h1.gather(4, &w1_idx);

        // Trilinear interpolation: interpolate along W, then H, then D
        let one_minus_wd: Tensor<T, DimDyn> = (1.0 - &weight_d_broadcast).cast();
        let one_minus_wh: Tensor<T, DimDyn> = (1.0 - &weight_h_broadcast).cast();
        let one_minus_ww: Tensor<T, DimDyn> = (1.0 - &weight_w_broadcast).cast();
        let wd: Tensor<T, DimDyn> = weight_d_broadcast.cast();
        let wh: Tensor<T, DimDyn> = weight_h_broadcast.cast();
        let ww: Tensor<T, DimDyn> = weight_w_broadcast.cast();

        // Interpolate along W (8 → 4)
        let c00 = &v000 * &one_minus_ww + &v001 * &ww;
        let c01 = &v010 * &one_minus_ww + &v011 * &ww;
        let c10 = &v100 * &one_minus_ww + &v101 * &ww;
        let c11 = &v110 * &one_minus_ww + &v111 * &ww;

        // Interpolate along H (4 → 2)
        let c0 = &c00 * &one_minus_wh + &c01 * &wh;
        let c1 = &c10 * &one_minus_wh + &c11 * &wh;

        // Interpolate along D (2 → 1)
        let result_dyn = &c0 * &one_minus_wd + &c1 * &wd;

        // Convert back to Dim5
        result_dyn.reshape([batch, channels, out_d, out_h, out_w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // 1D Tests (NCW)
    // ========================================================================

    #[test]
    fn test_linear1d_upsample() {
        let input = Tensor::<f32, Dim3>::ones([1, 3, 100]);
        let output = input.linear1d(200);
        assert_eq!(output.shape(), &[1, 3, 200]);
    }

    #[test]
    fn test_linear1d_downsample() {
        let input = Tensor::<f32, Dim3>::ones([2, 1, 100]);
        let output = input.linear1d(50);
        assert_eq!(output.shape(), &[2, 1, 50]);
    }

    #[test]
    fn test_linear1d_same_size() {
        let input = Tensor::<f32, Dim3>::ones([1, 1, 10]);
        let output = input.linear1d(10);
        assert_eq!(output.shape(), &[1, 1, 10]);
    }

    // ========================================================================
    // 2D Tests (NCHW)
    // ========================================================================

    #[test]
    fn test_bilinear2d_upsample() {
        let input = Tensor::<f32, Dim4>::ones([1, 3, 4, 4]);
        let output = input.bilinear2d((8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn test_bilinear2d_downsample() {
        let input = Tensor::<f32, Dim4>::ones([2, 1, 8, 8]);
        let output = input.bilinear2d((4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4]);
    }

    #[test]
    fn test_bilinear2d_asymmetric() {
        let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 6]);
        let output = input.bilinear2d((8, 3));
        assert_eq!(output.shape(), &[1, 1, 8, 3]);
    }

    #[test]
    fn test_bilinear2d_f64() {
        let input = Tensor::<f64, Dim4>::ones([1, 2, 4, 4]);
        let output = input.bilinear2d((8, 8));
        assert_eq!(output.shape(), &[1, 2, 8, 8]);
    }

    // ========================================================================
    // 3D Tests (NCDHW)
    // ========================================================================

    #[test]
    fn test_trilinear3d_upsample() {
        let input = Tensor::<f32, Dim5>::ones([1, 3, 4, 4, 4]);
        let output = input.trilinear3d((8, 8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8, 8]);
    }

    #[test]
    fn test_trilinear3d_downsample() {
        let input = Tensor::<f32, Dim5>::ones([2, 1, 8, 8, 8]);
        let output = input.trilinear3d((4, 4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4, 4]);
    }

    #[test]
    fn test_trilinear3d_asymmetric() {
        let input = Tensor::<f32, Dim5>::ones([1, 1, 2, 4, 8]);
        let output = input.trilinear3d((4, 8, 4));
        assert_eq!(output.shape(), &[1, 1, 4, 8, 4]);
    }
}

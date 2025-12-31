//! Nearest-neighbor interpolation operations
//!
//! Provides nearest-neighbor interpolation for resizing tensors.
//! Composed from primops: arange, floor, cast, gather.
//!
//! Supported dimensions:
//! - 1D: `Dim3` (N, C, W) - e.g., audio signals
//! - 2D: `Dim4` (N, C, H, W) - e.g., images
//! - 3D: `Dim5` (N, C, D, H, W) - e.g., video/volumetric data

use crate::tensor::{Dim1, Dim3, Dim4, Dim5, DimDyn, FloatDType, Floor, Tensor};

// ============================================================================
// Helper: Generate indices for a single spatial dimension
// ============================================================================

/// Generate gather indices for one spatial dimension
fn make_indices(out_size: usize, in_size: usize) -> Tensor<f32, Dim1> {
    let scale = in_size as f32 / out_size as f32;
    let coords = Tensor::<f32, Dim1>::arange(out_size);
    let scaled = &coords * scale;
    let clamped = scaled.clamp(0.0, (in_size.saturating_sub(1)) as f32);
    clamped.floor()
}

// ============================================================================
// 1D Interpolation (NCW format)
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim3> {
    /// Nearest-neighbor interpolation for 1D signals (NCW format)
    ///
    /// Resizes the spatial dimension (W) using nearest-neighbor sampling.
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
    /// let output = input.nearest1d(200);
    /// assert_eq!(output.shape(), &[1, 3, 200]);
    /// ```
    pub fn nearest1d(&self, size: usize) -> Self {
        let shape = self.shape();
        let (batch, channels, in_w) = (shape[0], shape[1], shape[2]);
        let out_w = size;

        // Generate indices for W axis
        let w_floor = make_indices(out_w, in_w);

        // Reshape to [1, 1, out_w] and expand to [batch, channels, out_w]
        let w_idx_3d = w_floor.reshape_dyn(&[1, 1, out_w]);
        let w_idx_broadcast = w_idx_3d.expand(&[batch, channels, out_w]);
        let w_idx: Tensor<i64, DimDyn> = w_idx_broadcast.cast();

        // Gather along W axis (dim=2)
        let self_dyn = self.clone().into_dyn();
        let result_dyn = self_dyn.gather(2, &w_idx);

        // Convert back to Dim3
        result_dyn.reshape([batch, channels, out_w])
    }
}

// ============================================================================
// 2D Interpolation (NCHW format)
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim4> {
    /// Nearest-neighbor interpolation for 2D images (NCHW format)
    ///
    /// Resizes the spatial dimensions (H, W) using nearest-neighbor sampling.
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
    /// let output = input.nearest2d((8, 8));
    /// assert_eq!(output.shape(), &[1, 3, 8, 8]);
    /// ```
    pub fn nearest2d(&self, size: (usize, usize)) -> Self {
        let shape = self.shape();
        let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = size;

        // Generate indices for H axis
        let h_floor = make_indices(out_h, in_h);
        let h_idx_4d = h_floor.reshape_dyn(&[1, 1, out_h, 1]);
        let h_idx_broadcast = h_idx_4d.expand(&[batch, channels, out_h, in_w]);
        let h_idx: Tensor<i64, DimDyn> = h_idx_broadcast.cast();

        // Gather along H axis (dim=2)
        let self_dyn = self.clone().into_dyn();
        let gathered_h = self_dyn.gather(2, &h_idx);

        // Generate indices for W axis
        let w_floor = make_indices(out_w, in_w);
        let w_idx_4d = w_floor.reshape_dyn(&[1, 1, 1, out_w]);
        let w_idx_broadcast = w_idx_4d.expand(&[batch, channels, out_h, out_w]);
        let w_idx: Tensor<i64, DimDyn> = w_idx_broadcast.cast();

        // Gather along W axis (dim=3)
        let result_dyn = gathered_h.gather(3, &w_idx);

        // Convert back to Dim4
        result_dyn.reshape([batch, channels, out_h, out_w])
    }
}

// ============================================================================
// 3D Interpolation (NCDHW format)
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim5> {
    /// Nearest-neighbor interpolation for 3D volumes (NCDHW format)
    ///
    /// Resizes the spatial dimensions (D, H, W) using nearest-neighbor sampling.
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
    /// let output = input.nearest3d((16, 16, 16));
    /// assert_eq!(output.shape(), &[1, 3, 16, 16, 16]);
    /// ```
    pub fn nearest3d(&self, size: (usize, usize, usize)) -> Self {
        let shape = self.shape();
        let (batch, channels, in_d, in_h, in_w) =
            (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let (out_d, out_h, out_w) = size;

        // Gather along D axis (dim=2)
        let d_floor = make_indices(out_d, in_d);
        let d_idx_5d = d_floor.reshape_dyn(&[1, 1, out_d, 1, 1]);
        let d_idx_broadcast = d_idx_5d.expand(&[batch, channels, out_d, in_h, in_w]);
        let d_idx: Tensor<i64, DimDyn> = d_idx_broadcast.cast();

        let self_dyn = self.clone().into_dyn();
        let gathered_d = self_dyn.gather(2, &d_idx);

        // Gather along H axis (dim=3)
        let h_floor = make_indices(out_h, in_h);
        let h_idx_5d = h_floor.reshape_dyn(&[1, 1, 1, out_h, 1]);
        let h_idx_broadcast = h_idx_5d.expand(&[batch, channels, out_d, out_h, in_w]);
        let h_idx: Tensor<i64, DimDyn> = h_idx_broadcast.cast();

        let gathered_h = gathered_d.gather(3, &h_idx);

        // Gather along W axis (dim=4)
        let w_floor = make_indices(out_w, in_w);
        let w_idx_5d = w_floor.reshape_dyn(&[1, 1, 1, 1, out_w]);
        let w_idx_broadcast = w_idx_5d.expand(&[batch, channels, out_d, out_h, out_w]);
        let w_idx: Tensor<i64, DimDyn> = w_idx_broadcast.cast();

        let result_dyn = gathered_h.gather(4, &w_idx);

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
    fn test_nearest1d_upsample() {
        let input = Tensor::<f32, Dim3>::ones([1, 3, 100]);
        let output = input.nearest1d(200);
        assert_eq!(output.shape(), &[1, 3, 200]);
    }

    #[test]
    fn test_nearest1d_downsample() {
        let input = Tensor::<f32, Dim3>::ones([2, 1, 100]);
        let output = input.nearest1d(50);
        assert_eq!(output.shape(), &[2, 1, 50]);
    }

    #[test]
    fn test_nearest1d_same_size() {
        let input = Tensor::<f32, Dim3>::ones([1, 1, 10]);
        let output = input.nearest1d(10);
        assert_eq!(output.shape(), &[1, 1, 10]);
    }

    // ========================================================================
    // 2D Tests (NCHW)
    // ========================================================================

    #[test]
    fn test_nearest2d_upsample() {
        let input = Tensor::<f32, Dim4>::ones([1, 3, 4, 4]);
        let output = input.nearest2d((8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn test_nearest2d_downsample() {
        let input = Tensor::<f32, Dim4>::ones([2, 1, 8, 8]);
        let output = input.nearest2d((4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4]);
    }

    #[test]
    fn test_nearest2d_asymmetric() {
        let input = Tensor::<f32, Dim4>::ones([1, 1, 4, 6]);
        let output = input.nearest2d((8, 3));
        assert_eq!(output.shape(), &[1, 1, 8, 3]);
    }

    #[test]
    fn test_nearest2d_f64() {
        let input = Tensor::<f64, Dim4>::ones([1, 2, 4, 4]);
        let output = input.nearest2d((8, 8));
        assert_eq!(output.shape(), &[1, 2, 8, 8]);
    }

    // ========================================================================
    // 3D Tests (NCDHW)
    // ========================================================================

    #[test]
    fn test_nearest3d_upsample() {
        let input = Tensor::<f32, Dim5>::ones([1, 3, 4, 4, 4]);
        let output = input.nearest3d((8, 8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8, 8]);
    }

    #[test]
    fn test_nearest3d_downsample() {
        let input = Tensor::<f32, Dim5>::ones([2, 1, 8, 8, 8]);
        let output = input.nearest3d((4, 4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4, 4]);
    }

    #[test]
    fn test_nearest3d_asymmetric() {
        let input = Tensor::<f32, Dim5>::ones([1, 1, 2, 4, 8]);
        let output = input.nearest3d((4, 8, 4));
        assert_eq!(output.shape(), &[1, 1, 4, 8, 4]);
    }
}

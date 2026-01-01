//! Interpolation operations
//!
//! Provides interpolation functions for resizing tensors.
//!
//! ## Supported modes
//! - `nearest`: Nearest-neighbor interpolation
//! - `linear`: Linear interpolation (1D), bilinear (2D), trilinear (3D)
//!
//! ## Supported dimensions
//! - 1D: `Dim3` (N, C, W) - e.g., audio signals
//! - 2D: `Dim4` (N, C, H, W) - e.g., images
//! - 3D: `Dim5` (N, C, D, H, W) - e.g., video/volumetric data

use harp::tensor::{Dim1, Dim3, Dim4, Dim5, DimDyn, FloatDType, Floor, Tensor};

// ============================================================================
// Helper: Generate indices for nearest-neighbor interpolation
// ============================================================================

/// Generate gather indices for one spatial dimension (nearest-neighbor)
fn make_nearest_indices(out_size: usize, in_size: usize) -> Tensor<f32, Dim1> {
    let scale = in_size as f32 / out_size as f32;
    let coords = Tensor::<f32, Dim1>::arange(out_size);
    let scaled = &coords * scale;
    let clamped = scaled.clamp(0.0, (in_size.saturating_sub(1)) as f32);
    clamped.floor()
}

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
// 1D Nearest-Neighbor Interpolation (NCW format)
// ============================================================================

/// Nearest-neighbor interpolation for 1D signals (NCW format)
///
/// Resizes the spatial dimension (W) using nearest-neighbor sampling.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, W]
/// * `size` - Target width
///
/// # Returns
/// Resized tensor with shape [N, C, size]
///
/// # Example
/// ```ignore
/// use harp::tensor::{Tensor, Dim3};
/// use harp_nn::functional::interpolate::nearest1d;
///
/// let input = Tensor::<f32, Dim3>::ones([1, 3, 100]);
/// let output = nearest1d(&input, 200);
/// assert_eq!(output.shape(), &[1, 3, 200]);
/// ```
pub fn nearest1d<T: FloatDType>(input: &Tensor<T, Dim3>, size: usize) -> Tensor<T, Dim3> {
    let shape = input.shape();
    let (batch, channels, in_w) = (shape[0], shape[1], shape[2]);
    let out_w = size;

    let w_floor = make_nearest_indices(out_w, in_w);
    let w_idx_3d = w_floor.reshape_dyn(&[1, 1, out_w]);
    let w_idx_broadcast = w_idx_3d.expand(&[batch, channels, out_w]);
    let w_idx: Tensor<i64, DimDyn> = w_idx_broadcast.cast();

    let input_dyn = input.clone().into_dyn();
    let result_dyn = input_dyn.gather(2, &w_idx);

    result_dyn.reshape([batch, channels, out_w])
}

// ============================================================================
// 2D Nearest-Neighbor Interpolation (NCHW format)
// ============================================================================

/// Nearest-neighbor interpolation for 2D images (NCHW format)
///
/// Resizes the spatial dimensions (H, W) using nearest-neighbor sampling.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `size` - Target size (height, width)
///
/// # Returns
/// Resized tensor with shape [N, C, size.0, size.1]
///
/// # Example
/// ```ignore
/// use harp::tensor::{Tensor, Dim4};
/// use harp_nn::functional::interpolate::nearest2d;
///
/// let input = Tensor::<f32, Dim4>::ones([1, 3, 4, 4]);
/// let output = nearest2d(&input, (8, 8));
/// assert_eq!(output.shape(), &[1, 3, 8, 8]);
/// ```
pub fn nearest2d<T: FloatDType>(input: &Tensor<T, Dim4>, size: (usize, usize)) -> Tensor<T, Dim4> {
    let shape = input.shape();
    let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let (out_h, out_w) = size;

    // Gather along H axis
    let h_floor = make_nearest_indices(out_h, in_h);
    let h_idx_4d = h_floor.reshape_dyn(&[1, 1, out_h, 1]);
    let h_idx_broadcast = h_idx_4d.expand(&[batch, channels, out_h, in_w]);
    let h_idx: Tensor<i64, DimDyn> = h_idx_broadcast.cast();

    let input_dyn = input.clone().into_dyn();
    let gathered_h = input_dyn.gather(2, &h_idx);

    // Gather along W axis
    let w_floor = make_nearest_indices(out_w, in_w);
    let w_idx_4d = w_floor.reshape_dyn(&[1, 1, 1, out_w]);
    let w_idx_broadcast = w_idx_4d.expand(&[batch, channels, out_h, out_w]);
    let w_idx: Tensor<i64, DimDyn> = w_idx_broadcast.cast();

    let result_dyn = gathered_h.gather(3, &w_idx);

    result_dyn.reshape([batch, channels, out_h, out_w])
}

// ============================================================================
// 3D Nearest-Neighbor Interpolation (NCDHW format)
// ============================================================================

/// Nearest-neighbor interpolation for 3D volumes (NCDHW format)
///
/// Resizes the spatial dimensions (D, H, W) using nearest-neighbor sampling.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `size` - Target size (depth, height, width)
///
/// # Returns
/// Resized tensor with shape [N, C, size.0, size.1, size.2]
///
/// # Example
/// ```ignore
/// use harp::tensor::{Tensor, Dim5};
/// use harp_nn::functional::interpolate::nearest3d;
///
/// let input = Tensor::<f32, Dim5>::ones([1, 3, 8, 8, 8]);
/// let output = nearest3d(&input, (16, 16, 16));
/// assert_eq!(output.shape(), &[1, 3, 16, 16, 16]);
/// ```
pub fn nearest3d<T: FloatDType>(
    input: &Tensor<T, Dim5>,
    size: (usize, usize, usize),
) -> Tensor<T, Dim5> {
    let shape = input.shape();
    let (batch, channels, in_d, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let (out_d, out_h, out_w) = size;

    // Gather along D axis
    let d_floor = make_nearest_indices(out_d, in_d);
    let d_idx_5d = d_floor.reshape_dyn(&[1, 1, out_d, 1, 1]);
    let d_idx_broadcast = d_idx_5d.expand(&[batch, channels, out_d, in_h, in_w]);
    let d_idx: Tensor<i64, DimDyn> = d_idx_broadcast.cast();

    let input_dyn = input.clone().into_dyn();
    let gathered_d = input_dyn.gather(2, &d_idx);

    // Gather along H axis
    let h_floor = make_nearest_indices(out_h, in_h);
    let h_idx_5d = h_floor.reshape_dyn(&[1, 1, 1, out_h, 1]);
    let h_idx_broadcast = h_idx_5d.expand(&[batch, channels, out_d, out_h, in_w]);
    let h_idx: Tensor<i64, DimDyn> = h_idx_broadcast.cast();

    let gathered_h = gathered_d.gather(3, &h_idx);

    // Gather along W axis
    let w_floor = make_nearest_indices(out_w, in_w);
    let w_idx_5d = w_floor.reshape_dyn(&[1, 1, 1, 1, out_w]);
    let w_idx_broadcast = w_idx_5d.expand(&[batch, channels, out_d, out_h, out_w]);
    let w_idx: Tensor<i64, DimDyn> = w_idx_broadcast.cast();

    let result_dyn = gathered_h.gather(4, &w_idx);

    result_dyn.reshape([batch, channels, out_d, out_h, out_w])
}

// ============================================================================
// 1D Linear Interpolation (NCW format)
// ============================================================================

/// Linear interpolation for 1D signals (NCW format)
///
/// Resizes the spatial dimension (W) using linear interpolation.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, W]
/// * `size` - Target width
///
/// # Returns
/// Resized tensor with shape [N, C, size]
///
/// # Example
/// ```ignore
/// use harp::tensor::{Tensor, Dim3};
/// use harp_nn::functional::interpolate::linear1d;
///
/// let input = Tensor::<f32, Dim3>::ones([1, 3, 100]);
/// let output = linear1d(&input, 200);
/// assert_eq!(output.shape(), &[1, 3, 200]);
/// ```
pub fn linear1d<T: FloatDType>(input: &Tensor<T, Dim3>, size: usize) -> Tensor<T, Dim3> {
    let shape = input.shape();
    let (batch, channels, in_w) = (shape[0], shape[1], shape[2]);
    let out_w = size;

    let (w0, w1, weight_w) = make_linear_indices(out_w, in_w);

    // Reshape weight for broadcasting
    let weight_w_3d = weight_w.reshape_dyn(&[1, 1, out_w]);
    let weight_w_broadcast = weight_w_3d.expand(&[batch, channels, out_w]);

    // Prepare indices
    let w0_3d = w0.reshape_dyn(&[1, 1, out_w]);
    let w0_broadcast = w0_3d.expand(&[batch, channels, out_w]);
    let w0_idx: Tensor<i64, DimDyn> = w0_broadcast.cast();

    let w1_3d = w1.reshape_dyn(&[1, 1, out_w]);
    let w1_broadcast = w1_3d.expand(&[batch, channels, out_w]);
    let w1_idx: Tensor<i64, DimDyn> = w1_broadcast.cast();

    // Gather values
    let input_dyn = input.clone().into_dyn();
    let v0 = input_dyn.gather(2, &w0_idx);
    let v1 = input_dyn.gather(2, &w1_idx);

    // Linear interpolation
    let one_minus_w: Tensor<T, DimDyn> = (1.0 - &weight_w_broadcast).cast();
    let weight_t: Tensor<T, DimDyn> = weight_w_broadcast.cast();

    let result_dyn = &v0 * &one_minus_w + &v1 * &weight_t;

    result_dyn.reshape([batch, channels, out_w])
}

// ============================================================================
// 2D Bilinear Interpolation (NCHW format)
// ============================================================================

/// Bilinear interpolation for 2D images (NCHW format)
///
/// Resizes the spatial dimensions (H, W) using bilinear interpolation.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, H, W]
/// * `size` - Target size (height, width)
///
/// # Returns
/// Resized tensor with shape [N, C, size.0, size.1]
///
/// # Example
/// ```ignore
/// use harp::tensor::{Tensor, Dim4};
/// use harp_nn::functional::interpolate::bilinear2d;
///
/// let input = Tensor::<f32, Dim4>::ones([1, 3, 4, 4]);
/// let output = bilinear2d(&input, (8, 8));
/// assert_eq!(output.shape(), &[1, 3, 8, 8]);
/// ```
pub fn bilinear2d<T: FloatDType>(input: &Tensor<T, Dim4>, size: (usize, usize)) -> Tensor<T, Dim4> {
    let shape = input.shape();
    let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
    let (out_h, out_w) = size;

    let (h0, h1, weight_h) = make_linear_indices(out_h, in_h);
    let (w0, w1, weight_w) = make_linear_indices(out_w, in_w);

    // Reshape weights for broadcasting
    let weight_h_4d = weight_h.reshape_dyn(&[1, 1, out_h, 1]);
    let weight_h_broadcast = weight_h_4d.expand(&[batch, channels, out_h, out_w]);

    let weight_w_4d = weight_w.reshape_dyn(&[1, 1, 1, out_w]);
    let weight_w_broadcast = weight_w_4d.expand(&[batch, channels, out_h, out_w]);

    // Prepare H indices
    let h0_4d = h0.reshape_dyn(&[1, 1, out_h, 1]);
    let h0_broadcast = h0_4d.expand(&[batch, channels, out_h, in_w]);
    let h0_idx: Tensor<i64, DimDyn> = h0_broadcast.cast();

    let h1_4d = h1.reshape_dyn(&[1, 1, out_h, 1]);
    let h1_broadcast = h1_4d.expand(&[batch, channels, out_h, in_w]);
    let h1_idx: Tensor<i64, DimDyn> = h1_broadcast.cast();

    // Gather along H axis
    let input_dyn = input.clone().into_dyn();
    let gathered_h0 = input_dyn.gather(2, &h0_idx);
    let gathered_h1 = input_dyn.gather(2, &h1_idx);

    // Prepare W indices
    let w0_4d = w0.reshape_dyn(&[1, 1, 1, out_w]);
    let w0_broadcast = w0_4d.expand(&[batch, channels, out_h, out_w]);
    let w0_idx: Tensor<i64, DimDyn> = w0_broadcast.cast();

    let w1_4d = w1.reshape_dyn(&[1, 1, 1, out_w]);
    let w1_broadcast = w1_4d.expand(&[batch, channels, out_h, out_w]);
    let w1_idx: Tensor<i64, DimDyn> = w1_broadcast.cast();

    // Gather all 4 corner values
    let v00 = gathered_h0.gather(3, &w0_idx);
    let v01 = gathered_h0.gather(3, &w1_idx);
    let v10 = gathered_h1.gather(3, &w0_idx);
    let v11 = gathered_h1.gather(3, &w1_idx);

    // Bilinear interpolation
    let one_minus_wh: Tensor<T, DimDyn> = (1.0 - &weight_h_broadcast).cast();
    let one_minus_ww: Tensor<T, DimDyn> = (1.0 - &weight_w_broadcast).cast();
    let wh: Tensor<T, DimDyn> = weight_h_broadcast.cast();
    let ww: Tensor<T, DimDyn> = weight_w_broadcast.cast();

    let top = &v00 * &one_minus_ww + &v01 * &ww;
    let bottom = &v10 * &one_minus_ww + &v11 * &ww;

    let result_dyn = &top * &one_minus_wh + &bottom * &wh;

    result_dyn.reshape([batch, channels, out_h, out_w])
}

// ============================================================================
// 3D Trilinear Interpolation (NCDHW format)
// ============================================================================

/// Trilinear interpolation for 3D volumes (NCDHW format)
///
/// Resizes the spatial dimensions (D, H, W) using trilinear interpolation.
///
/// # Arguments
/// * `input` - Input tensor of shape [N, C, D, H, W]
/// * `size` - Target size (depth, height, width)
///
/// # Returns
/// Resized tensor with shape [N, C, size.0, size.1, size.2]
///
/// # Example
/// ```ignore
/// use harp::tensor::{Tensor, Dim5};
/// use harp_nn::functional::interpolate::trilinear3d;
///
/// let input = Tensor::<f32, Dim5>::ones([1, 3, 8, 8, 8]);
/// let output = trilinear3d(&input, (16, 16, 16));
/// assert_eq!(output.shape(), &[1, 3, 16, 16, 16]);
/// ```
pub fn trilinear3d<T: FloatDType>(
    input: &Tensor<T, Dim5>,
    size: (usize, usize, usize),
) -> Tensor<T, Dim5> {
    let shape = input.shape();
    let (batch, channels, in_d, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let (out_d, out_h, out_w) = size;

    let (d0, d1, weight_d) = make_linear_indices(out_d, in_d);
    let (h0, h1, weight_h) = make_linear_indices(out_h, in_h);
    let (w0, w1, weight_w) = make_linear_indices(out_w, in_w);

    // Reshape weights for broadcasting
    let weight_d_5d = weight_d.reshape_dyn(&[1, 1, out_d, 1, 1]);
    let weight_d_broadcast = weight_d_5d.expand(&[batch, channels, out_d, out_h, out_w]);

    let weight_h_5d = weight_h.reshape_dyn(&[1, 1, 1, out_h, 1]);
    let weight_h_broadcast = weight_h_5d.expand(&[batch, channels, out_d, out_h, out_w]);

    let weight_w_5d = weight_w.reshape_dyn(&[1, 1, 1, 1, out_w]);
    let weight_w_broadcast = weight_w_5d.expand(&[batch, channels, out_d, out_h, out_w]);

    // Prepare D indices
    let d0_5d = d0.reshape_dyn(&[1, 1, out_d, 1, 1]);
    let d0_broadcast = d0_5d.expand(&[batch, channels, out_d, in_h, in_w]);
    let d0_idx: Tensor<i64, DimDyn> = d0_broadcast.cast();

    let d1_5d = d1.reshape_dyn(&[1, 1, out_d, 1, 1]);
    let d1_broadcast = d1_5d.expand(&[batch, channels, out_d, in_h, in_w]);
    let d1_idx: Tensor<i64, DimDyn> = d1_broadcast.cast();

    // Gather along D axis
    let input_dyn = input.clone().into_dyn();
    let gathered_d0 = input_dyn.gather(2, &d0_idx);
    let gathered_d1 = input_dyn.gather(2, &d1_idx);

    // Prepare H indices
    let h0_5d = h0.reshape_dyn(&[1, 1, 1, out_h, 1]);
    let h0_broadcast = h0_5d.expand(&[batch, channels, out_d, out_h, in_w]);
    let h0_idx: Tensor<i64, DimDyn> = h0_broadcast.cast();

    let h1_5d = h1.reshape_dyn(&[1, 1, 1, out_h, 1]);
    let h1_broadcast = h1_5d.expand(&[batch, channels, out_d, out_h, in_w]);
    let h1_idx: Tensor<i64, DimDyn> = h1_broadcast.cast();

    // Gather along H axis
    let gathered_d0h0 = gathered_d0.gather(3, &h0_idx);
    let gathered_d0h1 = gathered_d0.gather(3, &h1_idx);
    let gathered_d1h0 = gathered_d1.gather(3, &h0_idx);
    let gathered_d1h1 = gathered_d1.gather(3, &h1_idx);

    // Prepare W indices
    let w0_5d = w0.reshape_dyn(&[1, 1, 1, 1, out_w]);
    let w0_broadcast = w0_5d.expand(&[batch, channels, out_d, out_h, out_w]);
    let w0_idx: Tensor<i64, DimDyn> = w0_broadcast.cast();

    let w1_5d = w1.reshape_dyn(&[1, 1, 1, 1, out_w]);
    let w1_broadcast = w1_5d.expand(&[batch, channels, out_d, out_h, out_w]);
    let w1_idx: Tensor<i64, DimDyn> = w1_broadcast.cast();

    // Gather all 8 corner values
    let v000 = gathered_d0h0.gather(4, &w0_idx);
    let v001 = gathered_d0h0.gather(4, &w1_idx);
    let v010 = gathered_d0h1.gather(4, &w0_idx);
    let v011 = gathered_d0h1.gather(4, &w1_idx);
    let v100 = gathered_d1h0.gather(4, &w0_idx);
    let v101 = gathered_d1h0.gather(4, &w1_idx);
    let v110 = gathered_d1h1.gather(4, &w0_idx);
    let v111 = gathered_d1h1.gather(4, &w1_idx);

    // Trilinear interpolation
    let one_minus_wd: Tensor<T, DimDyn> = (1.0 - &weight_d_broadcast).cast();
    let one_minus_wh: Tensor<T, DimDyn> = (1.0 - &weight_h_broadcast).cast();
    let one_minus_ww: Tensor<T, DimDyn> = (1.0 - &weight_w_broadcast).cast();
    let wd: Tensor<T, DimDyn> = weight_d_broadcast.cast();
    let wh: Tensor<T, DimDyn> = weight_h_broadcast.cast();
    let ww: Tensor<T, DimDyn> = weight_w_broadcast.cast();

    // Interpolate along W
    let c00 = &v000 * &one_minus_ww + &v001 * &ww;
    let c01 = &v010 * &one_minus_ww + &v011 * &ww;
    let c10 = &v100 * &one_minus_ww + &v101 * &ww;
    let c11 = &v110 * &one_minus_ww + &v111 * &ww;

    // Interpolate along H
    let c0 = &c00 * &one_minus_wh + &c01 * &wh;
    let c1 = &c10 * &one_minus_wh + &c11 * &wh;

    // Interpolate along D
    let result_dyn = &c0 * &one_minus_wd + &c1 * &wd;

    result_dyn.reshape([batch, channels, out_d, out_h, out_w])
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // 1D Nearest Tests
    // ========================================================================

    #[test]
    fn test_nearest1d_upsample() {
        let input = Tensor::<f32, Dim3>::ones([1, 3, 100]);
        let output = nearest1d(&input, 200);
        assert_eq!(output.shape(), &[1, 3, 200]);
    }

    #[test]
    fn test_nearest1d_downsample() {
        let input = Tensor::<f32, Dim3>::ones([2, 1, 100]);
        let output = nearest1d(&input, 50);
        assert_eq!(output.shape(), &[2, 1, 50]);
    }

    // ========================================================================
    // 2D Nearest Tests
    // ========================================================================

    #[test]
    fn test_nearest2d_upsample() {
        let input = Tensor::<f32, Dim4>::ones([1, 3, 4, 4]);
        let output = nearest2d(&input, (8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn test_nearest2d_downsample() {
        let input = Tensor::<f32, Dim4>::ones([2, 1, 8, 8]);
        let output = nearest2d(&input, (4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4]);
    }

    // ========================================================================
    // 3D Nearest Tests
    // ========================================================================

    #[test]
    fn test_nearest3d_upsample() {
        let input = Tensor::<f32, Dim5>::ones([1, 3, 4, 4, 4]);
        let output = nearest3d(&input, (8, 8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8, 8]);
    }

    #[test]
    fn test_nearest3d_downsample() {
        let input = Tensor::<f32, Dim5>::ones([2, 1, 8, 8, 8]);
        let output = nearest3d(&input, (4, 4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4, 4]);
    }

    // ========================================================================
    // 1D Linear Tests
    // ========================================================================

    #[test]
    fn test_linear1d_upsample() {
        let input = Tensor::<f32, Dim3>::ones([1, 3, 100]);
        let output = linear1d(&input, 200);
        assert_eq!(output.shape(), &[1, 3, 200]);
    }

    #[test]
    fn test_linear1d_downsample() {
        let input = Tensor::<f32, Dim3>::ones([2, 1, 100]);
        let output = linear1d(&input, 50);
        assert_eq!(output.shape(), &[2, 1, 50]);
    }

    // ========================================================================
    // 2D Bilinear Tests
    // ========================================================================

    #[test]
    fn test_bilinear2d_upsample() {
        let input = Tensor::<f32, Dim4>::ones([1, 3, 4, 4]);
        let output = bilinear2d(&input, (8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8]);
    }

    #[test]
    fn test_bilinear2d_downsample() {
        let input = Tensor::<f32, Dim4>::ones([2, 1, 8, 8]);
        let output = bilinear2d(&input, (4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4]);
    }

    // ========================================================================
    // 3D Trilinear Tests
    // ========================================================================

    #[test]
    fn test_trilinear3d_upsample() {
        let input = Tensor::<f32, Dim5>::ones([1, 3, 4, 4, 4]);
        let output = trilinear3d(&input, (8, 8, 8));
        assert_eq!(output.shape(), &[1, 3, 8, 8, 8]);
    }

    #[test]
    fn test_trilinear3d_downsample() {
        let input = Tensor::<f32, Dim5>::ones([2, 1, 8, 8, 8]);
        let output = trilinear3d(&input, (4, 4, 4));
        assert_eq!(output.shape(), &[2, 1, 4, 4, 4]);
    }
}

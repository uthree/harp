//! Fold (col2im) operations - inverse of unfold for gradient computation
//!
//! Fold is the transpose operation of unfold:
//! - Unfold: extracts sliding windows (many-to-one read, expressible as View)
//! - Fold: accumulates sliding windows back (one-to-many write with sum)
//!
//! Implementation uses only View operations (slice, pad) and existing binary ops,
//! which means gradients are handled automatically by existing backward implementations.

use super::PadValue;
use crate::tensor::{Dim3, Dim4, Dim5, Dim6, Dim8, DimDyn, FloatDType, Tensor};

// ============================================================================
// 2D Fold: [N, C, out_H, out_W, kH, kW] -> [N, C, H, W]
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim6> {
    /// 2D fold (col2im) - [N, C, out_H, out_W, kH, kW] -> [N, C, H, W]
    ///
    /// The inverse operation of unfold2d. Sums sliding window contributions
    /// back to the original spatial positions.
    ///
    /// # Arguments
    /// * `output_size` - Target output size (H, W)
    /// * `strides` - Strides used in the original unfold operation
    ///
    /// # Note
    /// Currently only stride=1 is supported for overlapping windows.
    /// For stride == kernel_size, a fast reshape path is used.
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim4>::ones([2, 3, 4, 4]);
    /// let unfolded = input.unfold2d((2, 2), (1, 1));  // [2, 3, 3, 3, 2, 2]
    /// let folded = unfolded.fold2d((4, 4), (1, 1));   // [2, 3, 4, 4]
    /// // Note: values are not identical due to overlap accumulation
    /// ```
    pub fn fold2d(&self, output_size: (usize, usize), strides: (usize, usize)) -> Tensor<T, Dim4> {
        self.fold2d_dilated(output_size, strides, (1, 1))
    }

    /// 2D fold with dilation - [N, C, out_H, out_W, kH, kW] -> [N, C, H, W]
    ///
    /// The inverse operation of unfold2d_dilated. Sums sliding window contributions
    /// back to the original spatial positions, accounting for dilation.
    ///
    /// # Arguments
    /// * `output_size` - Target output size (H, W)
    /// * `strides` - Strides used in the original unfold operation
    /// * `dilations` - Dilations used in the original unfold operation
    ///
    /// # Note
    /// Currently only stride=1 or stride=effective_kernel_size is supported.
    ///
    /// # Example
    /// ```ignore
    /// let input = Tensor::<f32, Dim4>::ones([2, 3, 8, 8]);
    /// let unfolded = input.unfold2d_dilated((3, 3), (1, 1), (2, 2));  // effective_k=5
    /// let folded = unfolded.fold2d_dilated((8, 8), (1, 1), (2, 2));
    /// ```
    pub fn fold2d_dilated(
        &self,
        output_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Tensor<T, Dim4> {
        let shape = self.shape();
        let (_n, _c, out_h, out_w, kh, kw) =
            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        let (target_h, target_w) = output_size;
        let (sh, sw) = strides;
        let (dh, dw) = dilations;

        // Calculate effective kernel sizes
        let eff_kh = (kh - 1) * dh + 1;
        let eff_kw = (kw - 1) * dw + 1;

        // Validate output size matches expected dimensions
        let expected_h = (out_h - 1) * sh + eff_kh;
        let expected_w = (out_w - 1) * sw + eff_kw;
        assert!(
            target_h == expected_h && target_w == expected_w,
            "Output size ({}, {}) doesn't match expected ({}, {}) for kernel ({}, {}), stride ({}, {}), dilation ({}, {})",
            target_h,
            target_w,
            expected_h,
            expected_w,
            kh,
            kw,
            sh,
            sw,
            dh,
            dw
        );

        // Unified implementation using interleave + pad + sum pattern
        // This handles all cases uniformly: any stride, any dilation
        self.fold2d_impl(output_size, strides, dilations)
    }

    /// Unified fold implementation using interleave + pad + sum pattern
    fn fold2d_impl(
        &self,
        output_size: (usize, usize),
        strides: (usize, usize),
        dilations: (usize, usize),
    ) -> Tensor<T, Dim4> {
        let shape = self.shape();
        let (n, c, out_h, out_w, kh, kw) =
            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        let (h, w) = output_size;
        let (sh, sw) = strides;
        let (dh, dw) = dilations;

        // Initialize result with zeros
        let mut result: Tensor<T, Dim4> = Tensor::<T, DimDyn>::zeros_dyn(&[n, c, h, w]).into_dim4();

        // Loop over kernel positions
        for ki in 0..kh {
            for kj in 0..kw {
                // Extract contribution at kernel position (ki, kj)
                // input[:, :, :, :, ki, kj] -> [N, C, out_H, out_W]
                let contrib = self
                    .slice(&[
                        (0, n),
                        (0, c),
                        (0, out_h),
                        (0, out_w),
                        (ki, ki + 1),
                        (kj, kj + 1),
                    ])
                    .squeeze(5)
                    .squeeze(4);

                // For stride > 1, interleave to expand contrib to stride intervals
                // [N, C, out_H, out_W] -> [N, C, 1+(out_H-1)*sh, 1+(out_W-1)*sw]
                let expanded = if sh > 1 || sw > 1 {
                    contrib.interleave(&[2, 3], &[sh, sw]).into_dim4()
                } else {
                    contrib
                };

                // Calculate expanded sizes
                let expanded_h = if out_h > 0 { 1 + (out_h - 1) * sh } else { 0 };
                let expanded_w = if out_w > 0 { 1 + (out_w - 1) * sw } else { 0 };

                // Pad to align contribution to correct position in output
                // Position (oh, ow) in contrib maps to (oh * sh + ki * dh, ow * sw + kj * dw) in output
                let pad_before_h = ki * dh;
                let pad_after_h = h - expanded_h - ki * dh;
                let pad_before_w = kj * dw;
                let pad_after_w = w - expanded_w - kj * dw;

                let padded = expanded.pad(
                    &[
                        (0, 0),
                        (0, 0),
                        (pad_before_h, pad_after_h),
                        (pad_before_w, pad_after_w),
                    ],
                    PadValue::Zero,
                );

                // Accumulate by summing
                result = result + padded;
            }
        }

        result
    }
}

// ============================================================================
// 1D Fold: [N, C, out_L, k] -> [N, C, L]
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim4> {
    /// 1D fold (col2im) - [N, C, out_L, k] -> [N, C, L]
    ///
    /// The inverse operation of unfold1d. Sums sliding window contributions
    /// back to the original positions.
    ///
    /// # Arguments
    /// * `output_size` - Target output length L
    /// * `stride` - Stride used in the original unfold operation
    pub fn fold1d(&self, output_size: usize, stride: usize) -> Tensor<T, Dim3> {
        self.fold1d_dilated(output_size, stride, 1)
    }

    /// 1D fold with dilation - [N, C, out_L, k] -> [N, C, L]
    ///
    /// The inverse operation of unfold1d_dilated. Sums sliding window contributions
    /// back to the original positions, accounting for dilation.
    ///
    /// # Arguments
    /// * `output_size` - Target output length L
    /// * `stride` - Stride used in the original unfold operation
    /// * `dilation` - Dilation used in the original unfold operation
    pub fn fold1d_dilated(
        &self,
        output_size: usize,
        stride: usize,
        dilation: usize,
    ) -> Tensor<T, Dim3> {
        let shape = self.shape();
        let (n, c, out_l, k) = (shape[0], shape[1], shape[2], shape[3]);
        let l = output_size;

        // Calculate effective kernel size
        let eff_k = (k - 1) * dilation + 1;

        // Validate output size
        let expected_l = (out_l - 1) * stride + eff_k;
        assert!(
            l == expected_l,
            "Output size {} doesn't match expected {} for kernel {} stride {} dilation {}",
            l,
            expected_l,
            k,
            stride,
            dilation
        );

        // Unified implementation using interleave + pad + sum pattern
        // This handles all cases uniformly: any stride, any dilation

        // Initialize result
        let mut result: Tensor<T, Dim3> = Tensor::<T, DimDyn>::zeros_dyn(&[n, c, l]).into_dim3();

        // Loop over kernel positions
        for ki in 0..k {
            // Extract contribution at kernel position ki
            // input[:, :, :, ki] -> [N, C, out_L]
            let contrib = self
                .slice(&[(0, n), (0, c), (0, out_l), (ki, ki + 1)])
                .squeeze(3);

            // For stride > 1, interleave to expand contrib to stride intervals
            // [N, C, out_L] -> [N, C, 1+(out_L-1)*stride]
            let expanded = if stride > 1 {
                contrib.interleave(&[2], &[stride]).into_dim3()
            } else {
                contrib
            };

            // Calculate expanded size
            let expanded_l = if out_l > 0 {
                1 + (out_l - 1) * stride
            } else {
                0
            };

            // Pad to align contribution (accounting for dilation)
            // Position ol in contrib maps to ol * stride + ki * dilation
            let pad_before = ki * dilation;
            let pad_after = l - expanded_l - ki * dilation;

            let padded = expanded.pad(&[(0, 0), (0, 0), (pad_before, pad_after)], PadValue::Zero);

            // Accumulate
            result = result + padded;
        }

        result
    }
}

// ============================================================================
// 3D Fold: [N, C, out_H, out_W, out_D, kH, kW, kD] -> [N, C, H, W, D]
// ============================================================================

impl<T: FloatDType> Tensor<T, Dim8> {
    /// 3D fold (col2im) - [N, C, out_H, out_W, out_D, kH, kW, kD] -> [N, C, H, W, D]
    ///
    /// The inverse operation of unfold3d. Sums sliding window contributions
    /// back to the original positions.
    ///
    /// # Arguments
    /// * `output_size` - Target output size (H, W, D)
    /// * `strides` - Strides used in the original unfold operation
    pub fn fold3d(
        &self,
        output_size: (usize, usize, usize),
        strides: (usize, usize, usize),
    ) -> Tensor<T, Dim5> {
        self.fold3d_dilated(output_size, strides, (1, 1, 1))
    }

    /// 3D fold with dilation - [N, C, out_H, out_W, out_D, kH, kW, kD] -> [N, C, H, W, D]
    ///
    /// The inverse operation of unfold3d_dilated. Sums sliding window contributions
    /// back to the original positions, accounting for dilation.
    ///
    /// # Arguments
    /// * `output_size` - Target output size (H, W, D)
    /// * `strides` - Strides used in the original unfold operation
    /// * `dilations` - Dilations used in the original unfold operation
    pub fn fold3d_dilated(
        &self,
        output_size: (usize, usize, usize),
        strides: (usize, usize, usize),
        dilations: (usize, usize, usize),
    ) -> Tensor<T, Dim5> {
        let shape = self.shape();
        let (n, c) = (shape[0], shape[1]);
        let (out_h, out_w, out_d) = (shape[2], shape[3], shape[4]);
        let (kh, kw, kd) = (shape[5], shape[6], shape[7]);
        let (target_h, target_w, target_d) = output_size;
        let (sh, sw, sd) = strides;
        let (dh, dw, dd) = dilations;

        // Calculate effective kernel sizes
        let eff_kh = (kh - 1) * dh + 1;
        let eff_kw = (kw - 1) * dw + 1;
        let eff_kd = (kd - 1) * dd + 1;

        // Validate output size
        let expected_h = (out_h - 1) * sh + eff_kh;
        let expected_w = (out_w - 1) * sw + eff_kw;
        let expected_d = (out_d - 1) * sd + eff_kd;
        assert!(
            target_h == expected_h && target_w == expected_w && target_d == expected_d,
            "Output size ({}, {}, {}) doesn't match expected ({}, {}, {}) for kernel ({}, {}, {}), stride ({}, {}, {}), dilation ({}, {}, {})",
            target_h,
            target_w,
            target_d,
            expected_h,
            expected_w,
            expected_d,
            kh,
            kw,
            kd,
            sh,
            sw,
            sd,
            dh,
            dw,
            dd
        );

        // Unified implementation using interleave + pad + sum pattern
        // This handles all cases uniformly: any stride, any dilation

        // Initialize result
        let (h, w, d) = output_size;
        let mut result: Tensor<T, Dim5> =
            Tensor::<T, DimDyn>::zeros_dyn(&[n, c, h, w, d]).into_dim5();

        // Loop over kernel positions
        for ki in 0..kh {
            for kj in 0..kw {
                for kk in 0..kd {
                    // Extract contribution at kernel position (ki, kj, kk)
                    let contrib = self
                        .slice(&[
                            (0, n),
                            (0, c),
                            (0, out_h),
                            (0, out_w),
                            (0, out_d),
                            (ki, ki + 1),
                            (kj, kj + 1),
                            (kk, kk + 1),
                        ])
                        .squeeze(7)
                        .squeeze(6)
                        .squeeze(5);

                    // For stride > 1, interleave to expand contrib to stride intervals
                    // [N, C, out_H, out_W, out_D] -> [N, C, 1+(out_H-1)*sh, 1+(out_W-1)*sw, 1+(out_D-1)*sd]
                    let expanded = if sh > 1 || sw > 1 || sd > 1 {
                        contrib.interleave(&[2, 3, 4], &[sh, sw, sd]).into_dim5()
                    } else {
                        contrib
                    };

                    // Calculate expanded sizes
                    let expanded_h = if out_h > 0 { 1 + (out_h - 1) * sh } else { 0 };
                    let expanded_w = if out_w > 0 { 1 + (out_w - 1) * sw } else { 0 };
                    let expanded_d = if out_d > 0 { 1 + (out_d - 1) * sd } else { 0 };

                    // Pad to align contribution (accounting for dilation)
                    // Position (oh, ow, od) in contrib maps to (oh * sh + ki * dh, ow * sw + kj * dw, od * sd + kk * dd)
                    let pad_before_h = ki * dh;
                    let pad_after_h = h - expanded_h - ki * dh;
                    let pad_before_w = kj * dw;
                    let pad_after_w = w - expanded_w - kj * dw;
                    let pad_before_d = kk * dd;
                    let pad_after_d = d - expanded_d - kk * dd;

                    let padded = expanded.pad(
                        &[
                            (0, 0),
                            (0, 0),
                            (pad_before_h, pad_after_h),
                            (pad_before_w, pad_after_w),
                            (pad_before_d, pad_after_d),
                        ],
                        PadValue::Zero,
                    );

                    // Accumulate
                    result = result + padded;
                }
            }
        }

        result
    }
}

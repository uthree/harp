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
        let shape = self.shape();
        let (_n, _c, out_h, out_w, kh, kw) =
            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        let (target_h, target_w) = output_size;
        let (sh, sw) = strides;

        // Validate output size matches expected dimensions
        let expected_h = (out_h - 1) * sh + kh;
        let expected_w = (out_w - 1) * sw + kw;
        assert!(
            target_h == expected_h && target_w == expected_w,
            "Output size ({}, {}) doesn't match expected ({}, {}) for kernel ({}, {}) stride ({}, {})",
            target_h,
            target_w,
            expected_h,
            expected_w,
            kh,
            kw,
            sh,
            sw
        );

        // Fast path: non-overlapping (stride == kernel_size)
        if sh == kh && sw == kw {
            return self.fold2d_non_overlapping(output_size);
        }

        // Currently only stride=1 is fully supported for overlapping
        assert!(
            sh == 1 && sw == 1,
            "fold2d: only stride=1 or stride=kernel_size currently supported, got ({}, {})",
            sh,
            sw
        );

        self.fold2d_overlapping(output_size)
    }

    /// Fast path for non-overlapping fold (stride == kernel_size)
    #[allow(unused_variables)]
    fn fold2d_non_overlapping(&self, output_size: (usize, usize)) -> Tensor<T, Dim4> {
        let shape = self.shape();
        let (n, c, _out_h, _out_w, _kh, _kw) =
            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        let (h, w) = output_size;

        // [N, C, out_H, out_W, kH, kW] -> [N, C, out_H, kH, out_W, kW]
        let permuted = self.permute(&[0, 1, 2, 4, 3, 5]);

        // [N, C, out_H, kH, out_W, kW] -> [N, C, H, W]
        permuted.reshape([n, c, h, w])
    }

    /// Overlapping fold using slice + pad + sum pattern
    fn fold2d_overlapping(&self, output_size: (usize, usize)) -> Tensor<T, Dim4> {
        let shape = self.shape();
        let (n, c, out_h, out_w, kh, kw) =
            (shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        let (h, w) = output_size;

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

                // Pad to align contribution to correct position in output
                // For stride=1: position (oh, ow) in contrib maps to (oh + ki, ow + kj) in output
                let pad_before_h = ki;
                let pad_after_h = h - out_h - ki;
                let pad_before_w = kj;
                let pad_after_w = w - out_w - kj;

                let padded = contrib.pad(
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
        let shape = self.shape();
        let (n, c, out_l, k) = (shape[0], shape[1], shape[2], shape[3]);
        let l = output_size;

        // Validate output size
        let expected_l = (out_l - 1) * stride + k;
        assert!(
            l == expected_l,
            "Output size {} doesn't match expected {} for kernel {} stride {}",
            l,
            expected_l,
            k,
            stride
        );

        // Fast path: non-overlapping
        if stride == k {
            // [N, C, out_L, k] -> [N, C, L]
            return self.reshape([n, c, l]);
        }

        // Currently only stride=1 is fully supported
        assert!(
            stride == 1,
            "fold1d: only stride=1 or stride=kernel_size currently supported, got {}",
            stride
        );

        // Initialize result
        let mut result: Tensor<T, Dim3> = Tensor::<T, DimDyn>::zeros_dyn(&[n, c, l]).into_dim3();

        // Loop over kernel positions
        for ki in 0..k {
            // Extract contribution at kernel position ki
            // input[:, :, :, ki] -> [N, C, out_L]
            let contrib = self
                .slice(&[(0, n), (0, c), (0, out_l), (ki, ki + 1)])
                .squeeze(3);

            // Pad to align contribution
            let pad_before = ki;
            let pad_after = l - out_l - ki;

            let padded = contrib.pad(&[(0, 0), (0, 0), (pad_before, pad_after)], PadValue::Zero);

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
        let shape = self.shape();
        let (n, c) = (shape[0], shape[1]);
        let (out_h, out_w, out_d) = (shape[2], shape[3], shape[4]);
        let (kh, kw, kd) = (shape[5], shape[6], shape[7]);
        let (target_h, target_w, target_d) = output_size;
        let (sh, sw, sd) = strides;

        // Validate output size
        let expected_h = (out_h - 1) * sh + kh;
        let expected_w = (out_w - 1) * sw + kw;
        let expected_d = (out_d - 1) * sd + kd;
        assert!(
            target_h == expected_h && target_w == expected_w && target_d == expected_d,
            "Output size ({}, {}, {}) doesn't match expected ({}, {}, {})",
            target_h,
            target_w,
            target_d,
            expected_h,
            expected_w,
            expected_d
        );

        // Fast path: non-overlapping
        if sh == kh && sw == kw && sd == kd {
            // [N, C, out_H, out_W, out_D, kH, kW, kD] -> permute -> reshape
            let permuted = self.permute(&[0, 1, 2, 5, 3, 6, 4, 7]);
            return permuted.reshape([n, c, target_h, target_w, target_d]);
        }

        // Currently only stride=1 is fully supported
        assert!(
            sh == 1 && sw == 1 && sd == 1,
            "fold3d: only stride=1 or stride=kernel_size currently supported"
        );

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

                    // Pad to align contribution
                    let padded = contrib.pad(
                        &[
                            (0, 0),
                            (0, 0),
                            (ki, h - out_h - ki),
                            (kj, w - out_w - kj),
                            (kk, d - out_d - kk),
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

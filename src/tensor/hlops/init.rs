//! High-level initialization operations
//!
//! - randn: Normal distribution using Box-Muller transform
//! - xavier_uniform/normal: Xavier/Glorot initialization
//! - kaiming_uniform/normal: Kaiming/He initialization

use std::f64::consts::PI;

use crate::tensor::{Dim, Dimension, Sqrt, Tensor};

// ============================================================================
// Macro for implementing randn for specific float types
// ============================================================================

macro_rules! impl_randn {
    ($float_type:ty) => {
        impl<const N: usize> Tensor<$float_type, Dim<N>>
        where
            Dim<N>: Dimension,
        {
            /// Create a tensor with standard normal distribution (mean=0, std=1)
            ///
            /// Uses Box-Muller transform to convert uniform random to normal distribution.
            ///
            /// # Example
            ///
            /// ```ignore
            /// let t = Tensor::<f32, Dim2>::randn([3, 4]);
            /// ```
            pub fn randn(shape: [usize; N]) -> Self {
                Self::randn_with(shape, 0.0, 1.0)
            }

            /// Create a tensor with normal distribution (specified mean and std)
            ///
            /// Uses Box-Muller transform: Z = sqrt(-2*ln(U1)) * cos(2*pi*U2)
            ///
            /// # Arguments
            ///
            /// * `shape` - Shape of the output tensor
            /// * `mean` - Mean of the distribution
            /// * `std` - Standard deviation of the distribution
            ///
            /// # Example
            ///
            /// ```ignore
            /// // Xavier initialization style
            /// let t = Tensor::<f32, Dim2>::randn_with([784, 256], 0.0, 0.05);
            /// ```
            pub fn randn_with(shape: [usize; N], mean: $float_type, std: $float_type) -> Self {
                // Generate two independent uniform random tensors
                let u1 = Self::rand(shape);
                let u2 = Self::rand(shape);

                // Box-Muller transform: Z = sqrt(-2*ln(U1)) * cos(2*pi*U2)
                // Add small epsilon to avoid ln(0)
                let epsilon = 1e-7 as $float_type;
                let u1_safe = &u1 + epsilon;

                // sqrt(-2 * ln(u1))
                let ln_u1 = u1_safe.ln();
                let scaled_ln = &ln_u1 * (-2.0 as $float_type);
                let r = scaled_ln.sqrt();

                // cos(2 * pi * u2)
                let two_pi = (2.0 * PI) as $float_type;
                let theta = &u2 * two_pi;
                let cos_theta = theta.cos();

                // Z = r * cos(theta) gives standard normal
                let z = &r * &cos_theta;

                // Scale and shift: X = mean + std * Z
                if mean == 0.0 && std == 1.0 {
                    z
                } else {
                    let scaled = &z * std;
                    &scaled + mean
                }
            }

            /// Xavier/Glorot uniform initialization
            ///
            /// Draws samples from U[-a, a] where a = sqrt(6 / (fan_in + fan_out))
            ///
            /// # Arguments
            ///
            /// * `shape` - Shape of the output tensor
            /// * `fan_in` - Number of input units
            /// * `fan_out` - Number of output units
            pub fn xavier_uniform(shape: [usize; N], fan_in: usize, fan_out: usize) -> Self {
                let a = (6.0 / (fan_in + fan_out) as f64).sqrt() as $float_type;

                // U[-a, a] = 2a * U[0,1) - a
                let u = Self::rand(shape);
                let scaled = &u * (2.0 as $float_type * a);
                &scaled - a
            }

            /// Xavier/Glorot normal initialization
            ///
            /// Draws samples from N(0, std) where std = sqrt(2 / (fan_in + fan_out))
            ///
            /// # Arguments
            ///
            /// * `shape` - Shape of the output tensor
            /// * `fan_in` - Number of input units
            /// * `fan_out` - Number of output units
            pub fn xavier_normal(shape: [usize; N], fan_in: usize, fan_out: usize) -> Self {
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt() as $float_type;
                Self::randn_with(shape, 0.0, std)
            }

            /// Kaiming/He uniform initialization (for ReLU)
            ///
            /// Draws samples from U[-a, a] where a = sqrt(6 / fan_in)
            ///
            /// # Arguments
            ///
            /// * `shape` - Shape of the output tensor
            /// * `fan_in` - Number of input units
            pub fn kaiming_uniform(shape: [usize; N], fan_in: usize) -> Self {
                let a = (6.0 / fan_in as f64).sqrt() as $float_type;

                // U[-a, a] = 2a * U[0,1) - a
                let u = Self::rand(shape);
                let scaled = &u * (2.0 as $float_type * a);
                &scaled - a
            }

            /// Kaiming/He normal initialization (for ReLU)
            ///
            /// Draws samples from N(0, std) where std = sqrt(2 / fan_in)
            ///
            /// # Arguments
            ///
            /// * `shape` - Shape of the output tensor
            /// * `fan_in` - Number of input units
            pub fn kaiming_normal(shape: [usize; N], fan_in: usize) -> Self {
                let std = (2.0 / fan_in as f64).sqrt() as $float_type;
                Self::randn_with(shape, 0.0, std)
            }
        }
    };
}

impl_randn!(f32);
impl_randn!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_randn_shape() {
        let t = Tensor::<f32, Dim2>::randn([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_randn_with_shape() {
        let t = Tensor::<f32, Dim2>::randn_with([5, 6], 0.0, 1.0);
        assert_eq!(t.shape(), &[5, 6]);
    }

    #[test]
    fn test_randn_f64() {
        let t = Tensor::<f64, Dim2>::randn([3, 4]);
        assert_eq!(t.shape(), &[3, 4]);
    }

    #[test]
    fn test_xavier_uniform() {
        let t = Tensor::<f32, Dim2>::xavier_uniform([256, 128], 256, 128);
        assert_eq!(t.shape(), &[256, 128]);
    }

    #[test]
    fn test_xavier_normal() {
        let t = Tensor::<f32, Dim2>::xavier_normal([256, 128], 256, 128);
        assert_eq!(t.shape(), &[256, 128]);
    }

    #[test]
    fn test_kaiming_uniform() {
        let t = Tensor::<f32, Dim2>::kaiming_uniform([256, 128], 256);
        assert_eq!(t.shape(), &[256, 128]);
    }

    #[test]
    fn test_kaiming_normal() {
        let t = Tensor::<f32, Dim2>::kaiming_normal([256, 128], 256);
        assert_eq!(t.shape(), &[256, 128]);
    }

    #[test]
    fn test_randn_f64_with_params() {
        let t = Tensor::<f64, Dim2>::randn_with([10, 10], 5.0, 2.0);
        assert_eq!(t.shape(), &[10, 10]);
    }

    #[test]
    fn test_xavier_uniform_f64() {
        let t = Tensor::<f64, Dim2>::xavier_uniform([100, 50], 100, 50);
        assert_eq!(t.shape(), &[100, 50]);
    }

    #[test]
    fn test_kaiming_normal_f64() {
        let t = Tensor::<f64, Dim2>::kaiming_normal([100, 50], 100);
        assert_eq!(t.shape(), &[100, 50]);
    }
}

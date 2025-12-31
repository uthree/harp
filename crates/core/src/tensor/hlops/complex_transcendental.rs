//! Complex transcendental functions
//!
//! This module provides transcendental functions for complex tensors.
//! All functions are implemented using decomposition to real operations.
//!
//! ## Implemented Functions
//!
//! - `exp(z)` - Complex exponential
//! - `ln(z)` - Complex natural logarithm
//! - `sqrt(z)` - Complex square root
//! - `sin(z)`, `cos(z)` - Complex sine and cosine
//!
//! ## Mathematical Background
//!
//! For z = a + bi:
//!
//! - `exp(a+bi) = exp(a) * (cos(b) + i*sin(b))`
//! - `ln(z) = ln|z| + i*arg(z)`
//! - `sqrt(z) = sqrt(|z|) * exp(i*arg(z)/2)`
//! - `sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)`
//! - `cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)`

use crate::tensor::primops::complex::{ImagPart, RealPart};
use crate::tensor::{Complex32, Complex64, Dimension, FloatDType, Sin, Sqrt, Tensor};

// ============================================================================
// Trait for complex transcendental operations
// ============================================================================

/// Complex exponential
pub trait ComplexExp {
    type Output;
    /// Compute complex exponential: exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
    fn exp(&self) -> Self::Output;
}

/// Complex natural logarithm
pub trait ComplexLn {
    type Output;
    /// Compute complex natural logarithm: ln(z) = ln|z| + i*arg(z)
    fn ln(&self) -> Self::Output;
}

/// Complex square root
pub trait ComplexSqrt {
    type Output;
    /// Compute complex square root: sqrt(z) = sqrt|z| * exp(i*arg(z)/2)
    fn sqrt(&self) -> Self::Output;
}

/// Complex sine
pub trait ComplexSin {
    type Output;
    /// Compute complex sine: sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
    fn sin(&self) -> Self::Output;
}

/// Complex cosine
pub trait ComplexCos {
    type Output;
    /// Compute complex cosine: cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
    fn cos(&self) -> Self::Output;
}

// ============================================================================
// Macro to implement complex transcendental operations
// ============================================================================

macro_rules! impl_complex_exp {
    ($complex_type:ty, $real_type:ty) => {
        impl<D: Dimension> ComplexExp for Tensor<$complex_type, D>
        where
            Tensor<$real_type, D>: Clone,
        {
            type Output = Tensor<$complex_type, D>;

            fn exp(&self) -> Self::Output {
                // exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
                let a = self.real();
                let b = self.imag();

                // exp(a) using the exp method from hlops/transcendental
                let exp_a = a.exp();

                // cos(b) and sin(b) using methods from hlops/transcendental
                let cos_b = b.cos();
                let sin_b = b.sin();

                // exp(a) * cos(b) and exp(a) * sin(b)
                let re = &exp_a * &cos_b;
                let im = &exp_a * &sin_b;

                // Construct complex result
                crate::tensor::primops::complex::complex(&re, &im)
            }
        }
    };
}

macro_rules! impl_complex_ln {
    ($complex_type:ty, $real_type:ty) => {
        impl<D: Dimension> ComplexLn for Tensor<$complex_type, D>
        where
            Tensor<$real_type, D>: Clone,
        {
            type Output = Tensor<$complex_type, D>;

            fn ln(&self) -> Self::Output {
                // ln(z) = ln|z| + i*arg(z)
                let a = self.real();
                let b = self.imag();

                // |z| = sqrt(a^2 + b^2)
                let abs_z = ((&a * &a) + (&b * &b)).sqrt();

                // ln|z| using the ln method from hlops/transcendental
                let ln_abs = abs_z.ln();

                // arg(z) = atan2(b, a)
                // Using simplified atan approximation
                let arg_z = atan2_approx(&b, &a);

                // Construct complex result
                crate::tensor::primops::complex::complex(&ln_abs, &arg_z)
            }
        }
    };
}

macro_rules! impl_complex_sqrt {
    ($complex_type:ty, $real_type:ty) => {
        impl<D: Dimension> ComplexSqrt for Tensor<$complex_type, D>
        where
            Tensor<$real_type, D>: Clone,
        {
            type Output = Tensor<$complex_type, D>;

            fn sqrt(&self) -> Self::Output {
                // sqrt(z) = sqrt(|z|) * exp(i*arg(z)/2)
                //         = sqrt(|z|) * (cos(arg(z)/2) + i*sin(arg(z)/2))
                let a = self.real();
                let b = self.imag();

                // |z| = sqrt(a^2 + b^2)
                let abs_z = ((&a * &a) + (&b * &b)).sqrt();
                let sqrt_abs = abs_z.sqrt();

                // arg(z)/2 using scalar multiplication
                let arg_z = atan2_approx(&b, &a);
                let half_arg = &arg_z * (0.5 as $real_type);

                // cos(arg/2) and sin(arg/2) using methods
                let cos_half = half_arg.cos();
                let sin_half = half_arg.sin();

                // sqrt(|z|) * cos(arg/2) and sqrt(|z|) * sin(arg/2)
                let re = &sqrt_abs * &cos_half;
                let im = &sqrt_abs * &sin_half;

                // Construct complex result
                crate::tensor::primops::complex::complex(&re, &im)
            }
        }
    };
}

macro_rules! impl_complex_sin {
    ($complex_type:ty, $real_type:ty) => {
        impl<D: Dimension> ComplexSin for Tensor<$complex_type, D>
        where
            Tensor<$real_type, D>: Clone,
        {
            type Output = Tensor<$complex_type, D>;

            fn sin(&self) -> Self::Output {
                // sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
                let a = self.real();
                let b = self.imag();

                // sin(a), cos(a) using methods (clone since sin/cos take ownership)
                let sin_a = a.clone().sin();
                let cos_a = a.cos();

                // cosh(b) = (exp(b) + exp(-b)) / 2
                // sinh(b) = (exp(b) - exp(-b)) / 2
                let (cosh_b, sinh_b) = cosh_sinh_impl::<$real_type, D>(&b);

                let re = &sin_a * &cosh_b;
                let im = &cos_a * &sinh_b;

                crate::tensor::primops::complex::complex(&re, &im)
            }
        }
    };
}

macro_rules! impl_complex_cos {
    ($complex_type:ty, $real_type:ty) => {
        impl<D: Dimension> ComplexCos for Tensor<$complex_type, D>
        where
            Tensor<$real_type, D>: Clone,
        {
            type Output = Tensor<$complex_type, D>;

            fn cos(&self) -> Self::Output {
                // cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
                let a = self.real();
                let b = self.imag();

                // sin(a), cos(a) using methods (clone since sin/cos take ownership)
                let sin_a = a.clone().sin();
                let cos_a = a.cos();

                // cosh(b), sinh(b)
                let (cosh_b, sinh_b) = cosh_sinh_impl::<$real_type, D>(&b);

                let re = &cos_a * &cosh_b;
                let im = -(&sin_a * &sinh_b);

                crate::tensor::primops::complex::complex(&re, &im)
            }
        }
    };
}

// ============================================================================
// Helper functions for transcendental operations
// ============================================================================

/// Compute atan2(y, x) using simplified approximation
/// This is a rough implementation; a proper atan2 primop should be added
fn atan2_approx<T: FloatDType + Copy, D: Dimension>(
    y: &Tensor<T, D>,
    x: &Tensor<T, D>,
) -> Tensor<T, D>
where
    Tensor<T, D>: Clone,
{
    // atan2(y, x) ~ atan(y/x) for x > 0
    // This is a simplified version that only works correctly for x > 0
    // A proper implementation would handle all quadrants
    let ratio = y / x;
    atan_approx(&ratio)
}

/// Approximate atan using Taylor series
/// atan(x) ~ x - x^3/3 + x^5/5 - ... (for |x| <= 1)
fn atan_approx<T: FloatDType + Copy, D: Dimension>(x: &Tensor<T, D>) -> Tensor<T, D>
where
    Tensor<T, D>: Clone,
{
    // Simple polynomial approximation: atan(x) ~ x - x^3/3
    // This is accurate for |x| << 1
    let x2 = x * x;
    let x3 = &x2 * x;
    let third = T::from_f64(1.0 / 3.0);
    x - &(&x3 * third)
}

/// Compute cosh(x) and sinh(x)
/// cosh(x) = (exp(x) + exp(-x)) / 2
/// sinh(x) = (exp(x) - exp(-x)) / 2
fn cosh_sinh_impl<T: FloatDType + Copy, D: Dimension>(
    x: &Tensor<T, D>,
) -> (Tensor<T, D>, Tensor<T, D>)
where
    Tensor<T, D>: Clone,
{
    // exp(x) and exp(-x) using the exp method
    let exp_x = x.exp();
    let neg_x = -x;
    let exp_neg_x = neg_x.exp();

    // Use scalar multiplication for 0.5
    let half = T::from_f64(0.5);
    let cosh = (&exp_x + &exp_neg_x) * half;
    let sinh = (&exp_x - &exp_neg_x) * half;

    (cosh, sinh)
}

// ============================================================================
// Apply macros
// ============================================================================

impl_complex_exp!(Complex32, f32);
impl_complex_exp!(Complex64, f64);

impl_complex_ln!(Complex32, f32);
impl_complex_ln!(Complex64, f64);

impl_complex_sqrt!(Complex32, f32);
impl_complex_sqrt!(Complex64, f64);

impl_complex_sin!(Complex32, f32);
impl_complex_sin!(Complex64, f64);

impl_complex_cos!(Complex32, f32);
impl_complex_cos!(Complex64, f64);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Tests will be added when tensor execution is implemented for complex types
}

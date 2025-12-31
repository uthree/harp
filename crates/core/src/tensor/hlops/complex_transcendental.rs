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

use std::sync::Arc;

use crate::tensor::primops::complex::{Conjugate, ImagPart, RealPart};
use crate::tensor::{
    Complex32, Complex64, ComplexGradFn, Dimension, FloatDType, Sin, Sqrt, Tensor,
};

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
// Backward functions for complex transcendental operations
// ============================================================================

/// Macro to define backward structs for transcendental functions
/// These store input and output for gradient computation
macro_rules! impl_transcendental_backward {
    ($name:ident, $complex_type:ty, $real_type:ty, $backward_impl:expr) => {
        pub struct $name<D: Dimension> {
            input: Tensor<$complex_type, D>,
            #[allow(dead_code)]
            output: Tensor<$complex_type, D>,
        }

        impl<D: Dimension> $name<D> {
            pub fn new(input: Tensor<$complex_type, D>, output: Tensor<$complex_type, D>) -> Self {
                Self { input, output }
            }
        }

        impl<D: Dimension> ComplexGradFn<$real_type, D> for $name<D> {
            fn backward(&self, grad_output: &Tensor<$complex_type, D>) {
                $backward_impl(self, grad_output);
            }

            fn name(&self) -> &'static str {
                stringify!($name)
            }
        }
    };
}

// Exp backward: ∂/∂z exp(z) = exp(z)
// grad_input = grad_output * conj(exp(z))
impl_transcendental_backward!(
    Complex32ExpBackward,
    Complex32,
    f32,
    |this: &Complex32ExpBackward<D>, grad_output: &Tensor<Complex32, D>| {
        if this.input.requires_grad() {
            let grad = grad_output * &this.output.conj();
            this.input.backward_with(grad);
        }
    }
);

impl_transcendental_backward!(
    Complex64ExpBackward,
    Complex64,
    f64,
    |this: &Complex64ExpBackward<D>, grad_output: &Tensor<Complex64, D>| {
        if this.input.requires_grad() {
            let grad = grad_output * &this.output.conj();
            this.input.backward_with(grad);
        }
    }
);

// Ln backward: ∂/∂z ln(z) = 1/z
// grad_input = grad_output / conj(z)
impl_transcendental_backward!(
    Complex32LnBackward,
    Complex32,
    f32,
    |this: &Complex32LnBackward<D>, grad_output: &Tensor<Complex32, D>| {
        use crate::tensor::primops::Recip;
        if this.input.requires_grad() {
            let input_conj_recip = this.input.conj().recip();
            let grad = grad_output * &input_conj_recip;
            this.input.backward_with(grad);
        }
    }
);

impl_transcendental_backward!(
    Complex64LnBackward,
    Complex64,
    f64,
    |this: &Complex64LnBackward<D>, grad_output: &Tensor<Complex64, D>| {
        use crate::tensor::primops::Recip;
        if this.input.requires_grad() {
            let input_conj_recip = this.input.conj().recip();
            let grad = grad_output * &input_conj_recip;
            this.input.backward_with(grad);
        }
    }
);

// Sqrt backward: ∂/∂z sqrt(z) = 1/(2*sqrt(z))
// grad_input = grad_output / (2 * conj(sqrt(z)))
impl_transcendental_backward!(
    Complex32SqrtBackward,
    Complex32,
    f32,
    |this: &Complex32SqrtBackward<D>, grad_output: &Tensor<Complex32, D>| {
        use crate::tensor::primops::Recip;
        if this.input.requires_grad() {
            // grad = grad_out / (2 * conj(output))
            // 2 * output = output + output (no scalar multiplication for complex)
            let two_output = &this.output + &this.output;
            let grad = grad_output * &two_output.conj().recip();
            this.input.backward_with(grad);
        }
    }
);

impl_transcendental_backward!(
    Complex64SqrtBackward,
    Complex64,
    f64,
    |this: &Complex64SqrtBackward<D>, grad_output: &Tensor<Complex64, D>| {
        use crate::tensor::primops::Recip;
        if this.input.requires_grad() {
            // grad = grad_out / (2 * conj(output))
            // 2 * output = output + output (no scalar multiplication for complex)
            let two_output = &this.output + &this.output;
            let grad = grad_output * &two_output.conj().recip();
            this.input.backward_with(grad);
        }
    }
);

// Sin backward: ∂/∂z sin(z) = cos(z)
// grad_input = grad_output * conj(cos(z))
impl_transcendental_backward!(
    Complex32SinBackward,
    Complex32,
    f32,
    |this: &Complex32SinBackward<D>, grad_output: &Tensor<Complex32, D>| {
        if this.input.requires_grad() {
            let cos_z = this.input.cos();
            let grad = grad_output * &cos_z.conj();
            this.input.backward_with(grad);
        }
    }
);

impl_transcendental_backward!(
    Complex64SinBackward,
    Complex64,
    f64,
    |this: &Complex64SinBackward<D>, grad_output: &Tensor<Complex64, D>| {
        if this.input.requires_grad() {
            let cos_z = this.input.cos();
            let grad = grad_output * &cos_z.conj();
            this.input.backward_with(grad);
        }
    }
);

// Cos backward: ∂/∂z cos(z) = -sin(z)
// grad_input = -grad_output * conj(sin(z))
impl_transcendental_backward!(
    Complex32CosBackward,
    Complex32,
    f32,
    |this: &Complex32CosBackward<D>, grad_output: &Tensor<Complex32, D>| {
        if this.input.requires_grad() {
            let sin_z = this.input.sin();
            let grad = -(grad_output * &sin_z.conj());
            this.input.backward_with(grad);
        }
    }
);

impl_transcendental_backward!(
    Complex64CosBackward,
    Complex64,
    f64,
    |this: &Complex64CosBackward<D>, grad_output: &Tensor<Complex64, D>| {
        if this.input.requires_grad() {
            let sin_z = this.input.sin();
            let grad = -(grad_output * &sin_z.conj());
            this.input.backward_with(grad);
        }
    }
);

// ============================================================================
// Macro to implement complex transcendental operations
// ============================================================================

macro_rules! impl_complex_exp {
    ($complex_type:ty, $real_type:ty, $backward_type:ident) => {
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
                let result = crate::tensor::primops::complex::complex(&re, &im);

                if self.requires_grad() {
                    let grad_fn: Arc<dyn ComplexGradFn<$real_type, D>> =
                        Arc::new($backward_type::new(self.clone(), result.clone()));
                    result.with_complex_grad_fn(grad_fn)
                } else {
                    result
                }
            }
        }
    };
}

macro_rules! impl_complex_ln {
    ($complex_type:ty, $real_type:ty, $backward_type:ident) => {
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
                let result = crate::tensor::primops::complex::complex(&ln_abs, &arg_z);

                if self.requires_grad() {
                    let grad_fn: Arc<dyn ComplexGradFn<$real_type, D>> =
                        Arc::new($backward_type::new(self.clone(), result.clone()));
                    result.with_complex_grad_fn(grad_fn)
                } else {
                    result
                }
            }
        }
    };
}

macro_rules! impl_complex_sqrt {
    ($complex_type:ty, $real_type:ty, $backward_type:ident) => {
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
                let result = crate::tensor::primops::complex::complex(&re, &im);

                if self.requires_grad() {
                    let grad_fn: Arc<dyn ComplexGradFn<$real_type, D>> =
                        Arc::new($backward_type::new(self.clone(), result.clone()));
                    result.with_complex_grad_fn(grad_fn)
                } else {
                    result
                }
            }
        }
    };
}

macro_rules! impl_complex_sin {
    ($complex_type:ty, $real_type:ty, $backward_type:ident) => {
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

                let result = crate::tensor::primops::complex::complex(&re, &im);

                if self.requires_grad() {
                    let grad_fn: Arc<dyn ComplexGradFn<$real_type, D>> =
                        Arc::new($backward_type::new(self.clone(), result.clone()));
                    result.with_complex_grad_fn(grad_fn)
                } else {
                    result
                }
            }
        }
    };
}

macro_rules! impl_complex_cos {
    ($complex_type:ty, $real_type:ty, $backward_type:ident) => {
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

                let result = crate::tensor::primops::complex::complex(&re, &im);

                if self.requires_grad() {
                    let grad_fn: Arc<dyn ComplexGradFn<$real_type, D>> =
                        Arc::new($backward_type::new(self.clone(), result.clone()));
                    result.with_complex_grad_fn(grad_fn)
                } else {
                    result
                }
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

impl_complex_exp!(Complex32, f32, Complex32ExpBackward);
impl_complex_exp!(Complex64, f64, Complex64ExpBackward);

impl_complex_ln!(Complex32, f32, Complex32LnBackward);
impl_complex_ln!(Complex64, f64, Complex64LnBackward);

impl_complex_sqrt!(Complex32, f32, Complex32SqrtBackward);
impl_complex_sqrt!(Complex64, f64, Complex64SqrtBackward);

impl_complex_sin!(Complex32, f32, Complex32SinBackward);
impl_complex_sin!(Complex64, f64, Complex64SinBackward);

impl_complex_cos!(Complex32, f32, Complex32CosBackward);
impl_complex_cos!(Complex64, f64, Complex64CosBackward);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    // Tests will be added when tensor execution is implemented for complex types
}

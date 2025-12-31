//! Complex number type
//!
//! This module provides a custom `Complex<T>` type for complex number tensors.
//! Using a custom implementation avoids orphan rules issues when implementing
//! tensor operations like `Add<Tensor<Complex<T>>>`.
//!
//! # Memory Layout
//!
//! The `Complex<T>` type uses `#[repr(C)]` to ensure interleaved memory layout
//! (real, imag, real, imag, ...) which is compatible with BLAS libraries and
//! provides good cache efficiency.
//!
//! # Examples
//!
//! ```ignore
//! use harp_core::tensor::Complex;
//!
//! let z = Complex::new(3.0f32, 4.0f32);
//! assert_eq!(z.abs(), 5.0);  // |3 + 4i| = 5
//! assert_eq!(z.conj(), Complex::new(3.0, -4.0));
//! ```

use std::ops::{Add, Div, Mul, Neg, Sub};

use super::FloatDType;

/// Complex number type
///
/// A generic complex number with real and imaginary parts.
/// Uses `#[repr(C)]` for interleaved memory layout compatible with C/BLAS.
///
/// # Type Parameters
///
/// * `T` - The underlying real type (typically f32 or f64)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Complex<T> {
    /// Real part
    pub re: T,
    /// Imaginary part
    pub im: T,
}

impl<T> Complex<T> {
    /// Create a new complex number from real and imaginary parts
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

impl<T: Clone> Complex<T> {
    /// Get the real part
    #[inline]
    pub fn re(&self) -> T {
        self.re.clone()
    }

    /// Get the imaginary part
    #[inline]
    pub fn im(&self) -> T {
        self.im.clone()
    }
}

impl<T: FloatDType> Complex<T> {
    /// Imaginary unit i
    #[inline]
    pub fn i() -> Self {
        Self::new(T::ZERO, T::ONE)
    }

    /// Complex conjugate: conj(a + bi) = a - bi
    #[inline]
    pub fn conj(&self) -> Self {
        Self::new(self.re.clone(), -self.im.clone())
    }

    /// Squared norm (modulus squared): |z|² = re² + im²
    #[inline]
    pub fn norm_sqr(&self) -> T {
        self.re.clone() * self.re.clone() + self.im.clone() * self.im.clone()
    }

    /// Absolute value (modulus): |z| = sqrt(re² + im²)
    #[inline]
    pub fn abs(&self) -> T {
        self.norm_sqr().sqrt()
    }

    /// Argument (phase angle): arg(z) = atan2(im, re)
    #[inline]
    pub fn arg(&self) -> T {
        self.im.clone().atan2(self.re.clone())
    }

    /// Create a complex number from polar form: r * exp(i * theta)
    #[inline]
    pub fn from_polar(r: T, theta: T) -> Self {
        Self::new(r.clone() * theta.clone().cos(), r * theta.sin())
    }

    /// Convert to polar form: (r, theta) where z = r * exp(i * theta)
    #[inline]
    pub fn to_polar(&self) -> (T, T) {
        (self.abs(), self.arg())
    }

    /// Reciprocal: 1/z = conj(z) / |z|²
    #[inline]
    pub fn recip(&self) -> Self {
        let norm_sq = self.norm_sqr();
        Self::new(
            self.re.clone() / norm_sq.clone(),
            -self.im.clone() / norm_sq,
        )
    }

    /// Complex exponential: exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
    #[inline]
    pub fn exp(&self) -> Self {
        let exp_re = self.re.clone().exp();
        Self::new(
            exp_re.clone() * self.im.clone().cos(),
            exp_re * self.im.clone().sin(),
        )
    }

    /// Complex natural logarithm: ln(z) = ln|z| + i*arg(z)
    #[inline]
    pub fn ln(&self) -> Self {
        Self::new(self.abs().ln(), self.arg())
    }

    /// Complex square root: sqrt(z) = sqrt(|z|) * exp(i * arg(z) / 2)
    #[inline]
    pub fn sqrt(&self) -> Self {
        let (r, theta) = self.to_polar();
        let sqrt_r = r.sqrt();
        let half_theta = theta / T::TWO;
        Self::new(
            sqrt_r.clone() * half_theta.clone().cos(),
            sqrt_r * half_theta.sin(),
        )
    }

    /// Complex sine: sin(a + bi) = sin(a)cosh(b) + i*cos(a)sinh(b)
    #[inline]
    pub fn sin(&self) -> Self {
        Self::new(
            self.re.clone().sin() * self.im.clone().cosh(),
            self.re.clone().cos() * self.im.clone().sinh(),
        )
    }

    /// Complex cosine: cos(a + bi) = cos(a)cosh(b) - i*sin(a)sinh(b)
    #[inline]
    pub fn cos(&self) -> Self {
        Self::new(
            self.re.clone().cos() * self.im.clone().cosh(),
            -self.re.clone().sin() * self.im.clone().sinh(),
        )
    }

    /// Complex power: z^w (general complex exponentiation)
    #[inline]
    pub fn pow(&self, w: &Self) -> Self {
        // z^w = exp(w * ln(z))
        (w.clone() * self.ln()).exp()
    }

    /// Power with real exponent: z^n
    #[inline]
    pub fn powf(&self, n: T) -> Self {
        let (r, theta) = self.to_polar();
        let new_r = r.powf(n.clone());
        let new_theta = theta * n;
        Self::from_polar(new_r, new_theta)
    }
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

// Add: (a + bi) + (c + di) = (a + c) + (b + d)i
impl<T: FloatDType> Add for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: FloatDType> Add for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(
            self.re.clone() + rhs.re.clone(),
            self.im.clone() + rhs.im.clone(),
        )
    }
}

// Sub: (a + bi) - (c + di) = (a - c) + (b - d)i
impl<T: FloatDType> Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: FloatDType> Sub for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(
            self.re.clone() - rhs.re.clone(),
            self.im.clone() - rhs.im.clone(),
        )
    }
}

// Mul: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
impl<T: FloatDType> Mul for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone(),
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl<T: FloatDType> Mul for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Complex::new(
            self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone(),
            self.re.clone() * rhs.im.clone() + self.im.clone() * rhs.re.clone(),
        )
    }
}

// Scalar multiplication: (a + bi) * c = ac + bci
impl<T: FloatDType> Mul<T> for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.re * rhs.clone(), self.im * rhs)
    }
}

impl<T: FloatDType> Mul<T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Complex::new(self.re.clone() * rhs.clone(), self.im.clone() * rhs)
    }
}

// Div: (a + bi) / (c + di) = (a + bi) * conj(c + di) / |c + di|²
impl<T: FloatDType> Div for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let norm_sq = rhs.norm_sqr();
        Self::new(
            (self.re.clone() * rhs.re.clone() + self.im.clone() * rhs.im.clone()) / norm_sq.clone(),
            (self.im * rhs.re - self.re * rhs.im) / norm_sq,
        )
    }
}

impl<T: FloatDType> Div for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let norm_sq = rhs.norm_sqr();
        Complex::new(
            (self.re.clone() * rhs.re.clone() + self.im.clone() * rhs.im.clone()) / norm_sq.clone(),
            (self.im.clone() * rhs.re.clone() - self.re.clone() * rhs.im.clone()) / norm_sq,
        )
    }
}

// Scalar division: (a + bi) / c = a/c + (b/c)i
impl<T: FloatDType> Div<T> for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.re / rhs.clone(), self.im / rhs)
    }
}

impl<T: FloatDType> Div<T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Complex::new(self.re.clone() / rhs.clone(), self.im.clone() / rhs)
    }
}

// Neg: -(a + bi) = -a - bi
impl<T: FloatDType> Neg for Complex<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl<T: FloatDType> Neg for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Complex::new(-self.re.clone(), -self.im.clone())
    }
}

// ============================================================================
// Default
// ============================================================================

impl<T: FloatDType> Default for Complex<T> {
    fn default() -> Self {
        Self::new(T::ZERO, T::ZERO)
    }
}

// ============================================================================
// Display
// ============================================================================

impl<T: std::fmt::Display> std::fmt::Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} + {}i)", self.re, self.im)
    }
}

// ============================================================================
// Type aliases
// ============================================================================

/// 32-bit complex number (Complex<f32>)
pub type Complex32 = Complex<f32>;

/// 64-bit complex number (Complex<f64>)
pub type Complex64 = Complex<f64>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_new() {
        let z = Complex::new(3.0f32, 4.0f32);
        assert_eq!(z.re, 3.0);
        assert_eq!(z.im, 4.0);
    }

    #[test]
    fn test_i() {
        let i = Complex32::i();
        assert_eq!(i.re, 0.0);
        assert_eq!(i.im, 1.0);
    }

    #[test]
    fn test_conj() {
        let z = Complex::new(3.0f32, 4.0f32);
        let conj = z.conj();
        assert_eq!(conj.re, 3.0);
        assert_eq!(conj.im, -4.0);
    }

    #[test]
    fn test_norm_sqr() {
        let z = Complex::new(3.0f32, 4.0f32);
        assert!(approx_eq(z.norm_sqr(), 25.0));
    }

    #[test]
    fn test_abs() {
        let z = Complex::new(3.0f32, 4.0f32);
        assert!(approx_eq(z.abs(), 5.0));
    }

    #[test]
    fn test_arg() {
        let z = Complex::new(1.0f32, 1.0f32);
        assert!(approx_eq(z.arg(), std::f32::consts::FRAC_PI_4));
    }

    #[test]
    fn test_add() {
        let a = Complex::new(1.0f32, 2.0f32);
        let b = Complex::new(3.0f32, 4.0f32);
        let c = a + b;
        assert_eq!(c.re, 4.0);
        assert_eq!(c.im, 6.0);
    }

    #[test]
    fn test_sub() {
        let a = Complex::new(5.0f32, 7.0f32);
        let b = Complex::new(3.0f32, 4.0f32);
        let c = a - b;
        assert_eq!(c.re, 2.0);
        assert_eq!(c.im, 3.0);
    }

    #[test]
    fn test_mul() {
        // (1 + 2i) * (3 + 4i) = 3 + 4i + 6i + 8i² = 3 + 10i - 8 = -5 + 10i
        let a = Complex::new(1.0f32, 2.0f32);
        let b = Complex::new(3.0f32, 4.0f32);
        let c = a * b;
        assert!(approx_eq(c.re, -5.0));
        assert!(approx_eq(c.im, 10.0));
    }

    #[test]
    fn test_mul_scalar() {
        let z = Complex::new(2.0f32, 3.0f32);
        let c = z * 2.0f32;
        assert_eq!(c.re, 4.0);
        assert_eq!(c.im, 6.0);
    }

    #[test]
    fn test_div() {
        // (4 + 2i) / (1 + i) = (4 + 2i)(1 - i) / 2 = (4 - 4i + 2i - 2i²) / 2 = (6 - 2i) / 2 = 3 - i
        let a = Complex::new(4.0f32, 2.0f32);
        let b = Complex::new(1.0f32, 1.0f32);
        let c = a / b;
        assert!(approx_eq(c.re, 3.0));
        assert!(approx_eq(c.im, -1.0));
    }

    #[test]
    fn test_div_scalar() {
        let z = Complex::new(4.0f32, 6.0f32);
        let c = z / 2.0f32;
        assert_eq!(c.re, 2.0);
        assert_eq!(c.im, 3.0);
    }

    #[test]
    fn test_neg() {
        let z = Complex::new(3.0f32, -4.0f32);
        let neg_z = -z;
        assert_eq!(neg_z.re, -3.0);
        assert_eq!(neg_z.im, 4.0);
    }

    #[test]
    fn test_recip() {
        // 1 / (3 + 4i) = (3 - 4i) / 25
        let z = Complex::new(3.0f32, 4.0f32);
        let r = z.recip();
        assert!(approx_eq(r.re, 3.0 / 25.0));
        assert!(approx_eq(r.im, -4.0 / 25.0));
    }

    #[test]
    fn test_exp() {
        // exp(0) = 1
        let z = Complex::new(0.0f32, 0.0f32);
        let e = z.exp();
        assert!(approx_eq(e.re, 1.0));
        assert!(approx_eq(e.im, 0.0));

        // exp(i*pi) = -1 (Euler's identity)
        let z = Complex::new(0.0f32, std::f32::consts::PI);
        let e = z.exp();
        assert!(approx_eq(e.re, -1.0));
        assert!(e.im.abs() < EPSILON);
    }

    #[test]
    fn test_ln() {
        // ln(1) = 0
        let z = Complex::new(1.0f32, 0.0f32);
        let l = z.ln();
        assert!(approx_eq(l.re, 0.0));
        assert!(approx_eq(l.im, 0.0));

        // ln(e) = 1
        let z = Complex::new(std::f32::consts::E, 0.0f32);
        let l = z.ln();
        assert!(approx_eq(l.re, 1.0));
        assert!(approx_eq(l.im, 0.0));
    }

    #[test]
    fn test_sqrt() {
        // sqrt(4) = 2
        let z = Complex::new(4.0f32, 0.0f32);
        let s = z.sqrt();
        assert!(approx_eq(s.re, 2.0));
        assert!(approx_eq(s.im, 0.0));

        // sqrt(-1) = i
        let z = Complex::new(-1.0f32, 0.0f32);
        let s = z.sqrt();
        assert!(s.re.abs() < EPSILON);
        assert!(approx_eq(s.im, 1.0));
    }

    #[test]
    fn test_sin_cos() {
        // sin(0) = 0, cos(0) = 1
        let z = Complex::new(0.0f32, 0.0f32);
        let s = z.sin();
        let c = z.cos();
        assert!(approx_eq(s.re, 0.0));
        assert!(approx_eq(s.im, 0.0));
        assert!(approx_eq(c.re, 1.0));
        assert!(approx_eq(c.im, 0.0));
    }

    #[test]
    fn test_from_polar() {
        // r=1, theta=pi/2 should give i
        let z = Complex32::from_polar(1.0, std::f32::consts::FRAC_PI_2);
        assert!(z.re.abs() < EPSILON);
        assert!(approx_eq(z.im, 1.0));
    }

    #[test]
    fn test_to_polar() {
        let z = Complex::new(0.0f32, 1.0f32); // i
        let (r, theta) = z.to_polar();
        assert!(approx_eq(r, 1.0));
        assert!(approx_eq(theta, std::f32::consts::FRAC_PI_2));
    }

    #[test]
    fn test_default() {
        let z = Complex32::default();
        assert_eq!(z.re, 0.0);
        assert_eq!(z.im, 0.0);
    }

    #[test]
    fn test_display() {
        let z = Complex::new(3.0f32, 4.0f32);
        assert_eq!(format!("{}", z), "(3 + 4i)");
    }

    #[test]
    fn test_type_aliases() {
        let _z32: Complex32 = Complex::new(1.0f32, 2.0f32);
        let _z64: Complex64 = Complex::new(1.0f64, 2.0f64);
    }

    #[test]
    fn test_reference_operations() {
        let a = Complex::new(1.0f32, 2.0f32);
        let b = Complex::new(3.0f32, 4.0f32);

        // Test reference add
        let c = &a + &b;
        assert_eq!(c.re, 4.0);
        assert_eq!(c.im, 6.0);

        // Test reference mul
        let d = &a * &b;
        assert!(approx_eq(d.re, -5.0));
        assert!(approx_eq(d.im, 10.0));
    }
}

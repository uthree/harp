//! Type-level dimension system for compile-time shape checking
//!
//! This module provides a type-level representation of tensor dimensions,
//! allowing dimension mismatches to be caught at compile time.
//!
//! # Examples
//!
//! ```ignore
//! use eclat::tensor::{Tensor, D2, D1};
//!
//! let a: Tensor<D2> = Tensor::input([32, 64], DType::F32);
//! let b: Tensor<D1> = a.sum(1);  // D2 -> D1 via sum
//! ```

use std::fmt::Debug;

// ============================================================================
// Core Dimension Trait
// ============================================================================

/// A type-level representation of tensor dimensionality.
///
/// Types implementing this trait represent the number of dimensions a tensor has.
/// This enables compile-time checking of dimension compatibility in operations.
pub trait Dimension: Clone + Debug + 'static {
    /// The number of dimensions (compile-time constant for fixed dims).
    const NDIM: usize;

    /// Check if a given shape is valid for this dimension.
    fn check_shape(shape: &[usize]) -> bool {
        shape.len() == Self::NDIM
    }
}

// ============================================================================
// Fixed Dimension Types
// ============================================================================

/// 0-dimensional tensor (scalar).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D0;

impl Dimension for D0 {
    const NDIM: usize = 0;
}

/// 1-dimensional tensor (vector).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D1;

impl Dimension for D1 {
    const NDIM: usize = 1;
}

/// 2-dimensional tensor (matrix).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D2;

impl Dimension for D2 {
    const NDIM: usize = 2;
}

/// 3-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D3;

impl Dimension for D3 {
    const NDIM: usize = 3;
}

/// 4-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D4;

impl Dimension for D4 {
    const NDIM: usize = 4;
}

/// 5-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D5;

impl Dimension for D5 {
    const NDIM: usize = 5;
}

/// 6-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D6;

impl Dimension for D6 {
    const NDIM: usize = 6;
}

/// 7-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D7;

impl Dimension for D7 {
    const NDIM: usize = 7;
}

/// 8-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D8;

impl Dimension for D8 {
    const NDIM: usize = 8;
}

/// Dynamic dimension (runtime-checked).
///
/// Use this when the number of dimensions is not known at compile time
/// or exceeds D6.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dyn(pub usize);

impl Dimension for Dyn {
    const NDIM: usize = usize::MAX; // Sentinel value

    fn check_shape(_shape: &[usize]) -> bool {
        // Dynamic dimension accepts any shape
        true
    }
}

// ============================================================================
// Dimension Transformation Traits
// ============================================================================

/// Add one dimension (for unsqueeze operation).
///
/// # Example
/// ```ignore
/// // D2.unsqueeze(0) -> D3
/// let a: Tensor<D2> = input([32, 64], F32);
/// let b: Tensor<D3> = a.unsqueeze(0);  // [1, 32, 64]
/// ```
pub trait DimAdd1: Dimension {
    /// The resulting dimension type after adding one dimension.
    type Output: Dimension;
}

impl DimAdd1 for D0 {
    type Output = D1;
}
impl DimAdd1 for D1 {
    type Output = D2;
}
impl DimAdd1 for D2 {
    type Output = D3;
}
impl DimAdd1 for D3 {
    type Output = D4;
}
impl DimAdd1 for D4 {
    type Output = D5;
}
impl DimAdd1 for D5 {
    type Output = D6;
}
impl DimAdd1 for D6 {
    type Output = Dyn;
}
impl DimAdd1 for Dyn {
    type Output = Dyn;
}

/// Remove one dimension (for squeeze/reduce operations).
///
/// # Example
/// ```ignore
/// // D2.sum(axis) -> D1
/// let a: Tensor<D2> = input([32, 64], F32);
/// let b: Tensor<D1> = a.sum(1);  // [32]
/// ```
pub trait DimSub1: Dimension {
    /// The resulting dimension type after removing one dimension.
    type Output: Dimension;
}

impl DimSub1 for D1 {
    type Output = D0;
}
impl DimSub1 for D2 {
    type Output = D1;
}
impl DimSub1 for D3 {
    type Output = D2;
}
impl DimSub1 for D4 {
    type Output = D3;
}
impl DimSub1 for D5 {
    type Output = D4;
}
impl DimSub1 for D6 {
    type Output = D5;
}
impl DimSub1 for Dyn {
    type Output = Dyn;
}

/// Dimension equality constraint (for binary operations, broadcasting).
///
/// This trait marks dimension types that are compatible in binary operations.
/// Same dimensions are always compatible, and Dyn is compatible with any dimension.
pub trait DimEq<Rhs: Dimension>: Dimension {}

// Same dimensions are always equal
impl<D: Dimension> DimEq<D> for D {}

// Dyn is compatible with any dimension (both directions)
impl DimEq<D0> for Dyn {}
impl DimEq<D1> for Dyn {}
impl DimEq<D2> for Dyn {}
impl DimEq<D3> for Dyn {}
impl DimEq<D4> for Dyn {}
impl DimEq<D5> for Dyn {}
impl DimEq<D6> for Dyn {}

impl DimEq<Dyn> for D0 {}
impl DimEq<Dyn> for D1 {}
impl DimEq<Dyn> for D2 {}
impl DimEq<Dyn> for D3 {}
impl DimEq<Dyn> for D4 {}
impl DimEq<Dyn> for D5 {}
impl DimEq<Dyn> for D6 {}

// ============================================================================
// Utility Traits
// ============================================================================

/// Convert dimension type to runtime value.
pub trait IntoUsize {
    /// Get the number of dimensions as a runtime value.
    fn ndim(&self) -> usize;
}

impl IntoUsize for D0 {
    fn ndim(&self) -> usize {
        0
    }
}
impl IntoUsize for D1 {
    fn ndim(&self) -> usize {
        1
    }
}
impl IntoUsize for D2 {
    fn ndim(&self) -> usize {
        2
    }
}
impl IntoUsize for D3 {
    fn ndim(&self) -> usize {
        3
    }
}
impl IntoUsize for D4 {
    fn ndim(&self) -> usize {
        4
    }
}
impl IntoUsize for D5 {
    fn ndim(&self) -> usize {
        5
    }
}
impl IntoUsize for D6 {
    fn ndim(&self) -> usize {
        6
    }
}
impl IntoUsize for Dyn {
    fn ndim(&self) -> usize {
        self.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_ndim() {
        assert_eq!(D0::NDIM, 0);
        assert_eq!(D1::NDIM, 1);
        assert_eq!(D2::NDIM, 2);
        assert_eq!(D3::NDIM, 3);
        assert_eq!(D4::NDIM, 4);
    }

    #[test]
    fn test_check_shape() {
        assert!(D0::check_shape(&[]));
        assert!(!D0::check_shape(&[1]));

        assert!(D1::check_shape(&[10]));
        assert!(!D1::check_shape(&[]));
        assert!(!D1::check_shape(&[10, 20]));

        assert!(D2::check_shape(&[10, 20]));
        assert!(!D2::check_shape(&[10]));

        // Dyn accepts any shape
        assert!(Dyn::check_shape(&[]));
        assert!(Dyn::check_shape(&[1, 2, 3, 4, 5]));
    }

    #[test]
    fn test_dim_add1() {
        fn check_add<D: DimAdd1>() -> usize {
            D::Output::NDIM
        }

        assert_eq!(check_add::<D0>(), 1);
        assert_eq!(check_add::<D1>(), 2);
        assert_eq!(check_add::<D2>(), 3);
        assert_eq!(check_add::<D3>(), 4);
    }

    #[test]
    fn test_dim_sub1() {
        fn check_sub<D: DimSub1>() -> usize {
            D::Output::NDIM
        }

        assert_eq!(check_sub::<D1>(), 0);
        assert_eq!(check_sub::<D2>(), 1);
        assert_eq!(check_sub::<D3>(), 2);
        assert_eq!(check_sub::<D4>(), 3);
    }

    #[test]
    fn test_dim_eq() {
        // This test verifies that DimEq is implemented correctly
        // by checking that certain operations compile
        fn require_eq<A: DimEq<B>, B: Dimension>() {}

        // Same dimensions
        require_eq::<D2, D2>();
        require_eq::<D3, D3>();

        // Dyn with others
        require_eq::<Dyn, D2>();
        require_eq::<D2, Dyn>();
    }
}

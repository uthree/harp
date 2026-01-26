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

pub trait Dimension: Clone + Debug + 'static {
    /// The number of dimensions (compile-time constant for fixed dims).
    const NDIM: usize;

    /// Next larger dimension (e.g., D2::Larger = D3).
    type Larger: Dimension;

    /// Next smaller dimension (e.g., D2::Smaller = D1).
    type Smaller: Dimension;

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
    type Larger = D1;
    type Smaller = D0; // D0 is the smallest
}

/// 1-dimensional tensor (vector).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D1;

impl Dimension for D1 {
    const NDIM: usize = 1;
    type Larger = D2;
    type Smaller = D0;
}

/// 2-dimensional tensor (matrix).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D2;

impl Dimension for D2 {
    const NDIM: usize = 2;
    type Larger = D3;
    type Smaller = D1;
}

/// 3-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D3;

impl Dimension for D3 {
    const NDIM: usize = 3;
    type Larger = D4;
    type Smaller = D2;
}

/// 4-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D4;

impl Dimension for D4 {
    const NDIM: usize = 4;
    type Larger = D5;
    type Smaller = D3;
}

/// 5-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D5;

impl Dimension for D5 {
    const NDIM: usize = 5;
    type Larger = D6;
    type Smaller = D4;
}

/// 6-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D6;

impl Dimension for D6 {
    const NDIM: usize = 6;
    type Larger = D7;
    type Smaller = D5;
}

/// 7-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D7;

impl Dimension for D7 {
    const NDIM: usize = 7;
    type Larger = D8;
    type Smaller = D6;
}

/// 8-dimensional tensor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct D8;

impl Dimension for D8 {
    const NDIM: usize = 8;
    type Larger = Dyn; // Beyond D8, use dynamic
    type Smaller = D7;
}

/// Dynamic dimension (runtime-checked).
///
/// Use this when the number of dimensions is not known at compile time
/// or exceeds D6.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dyn(pub usize);

impl Dimension for Dyn {
    const NDIM: usize = usize::MAX; // Sentinel value
    type Larger = Dyn;
    type Smaller = Dyn;

    fn check_shape(_shape: &[usize]) -> bool {
        // Dynamic dimension accepts any shape
        true
    }
}

// ============================================================================
// Dimension Compatibility Traits
// ============================================================================

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
impl DimEq<D7> for Dyn {}
impl DimEq<D8> for Dyn {}

impl DimEq<Dyn> for D0 {}
impl DimEq<Dyn> for D1 {}
impl DimEq<Dyn> for D2 {}
impl DimEq<Dyn> for D3 {}
impl DimEq<Dyn> for D4 {}
impl DimEq<Dyn> for D5 {}
impl DimEq<Dyn> for D6 {}
impl DimEq<Dyn> for D7 {}
impl DimEq<Dyn> for D8 {}

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
    fn test_dim_larger() {
        fn check_larger<D: Dimension>() -> usize {
            D::Larger::NDIM
        }

        assert_eq!(check_larger::<D0>(), 1);
        assert_eq!(check_larger::<D1>(), 2);
        assert_eq!(check_larger::<D2>(), 3);
        assert_eq!(check_larger::<D3>(), 4);
        assert_eq!(check_larger::<D7>(), 8);
        assert_eq!(check_larger::<D8>(), usize::MAX); // -> Dyn
    }

    #[test]
    fn test_dim_smaller() {
        fn check_smaller<D: Dimension>() -> usize {
            D::Smaller::NDIM
        }

        assert_eq!(check_smaller::<D0>(), 0); // D0 -> D0
        assert_eq!(check_smaller::<D1>(), 0);
        assert_eq!(check_smaller::<D2>(), 1);
        assert_eq!(check_smaller::<D3>(), 2);
        assert_eq!(check_smaller::<D4>(), 3);
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

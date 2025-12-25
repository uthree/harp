//! Dimension trait and implementations for static/dynamic dimension management
//!
//! This module provides compile-time dimension checking for tensors using const generics.

use std::fmt::Debug;

/// Trait for tensor dimensions
///
/// This trait is implemented by types that represent the dimensionality of a tensor.
/// It supports both statically-known dimensions (compile-time) and dynamic dimensions
/// (runtime).
///
/// # Examples
///
/// ```ignore
/// use harp::tensor::{Dim, DimDyn, Dimension};
///
/// // Static 2D dimension
/// let dim2 = Dim::<2>;
/// assert_eq!(dim2.ndim(), 2);
///
/// // Dynamic dimension
/// let dim_dyn = DimDyn::new(3);
/// assert_eq!(dim_dyn.ndim(), 3);
/// ```
pub trait Dimension: Clone + Debug + Send + Sync + 'static {
    /// The number of dimensions, if known at compile time
    const NDIM: Option<usize>;

    /// The dimension type with one fewer dimension (for squeeze/reduce operations)
    /// Returns DimDyn for edge cases
    type Smaller: Dimension;

    /// The dimension type with one more dimension (for unsqueeze/expand operations)
    type Larger: Dimension;

    /// Get the number of dimensions at runtime
    fn ndim(&self) -> usize;

    /// Check if this is a statically-known dimension
    fn is_static(&self) -> bool {
        Self::NDIM.is_some()
    }
}

/// Static dimension marker type using const generics
///
/// `Dim<N>` represents a tensor with exactly N dimensions known at compile time.
/// This enables compile-time dimension checking for operations.
///
/// # Examples
///
/// ```ignore
/// use harp::tensor::Dim;
///
/// // Create tensors with specific dimensions
/// let tensor_1d: Tensor<Dim<1>> = Tensor::zeros([10]);
/// let tensor_2d: Tensor<Dim<2>> = Tensor::zeros([3, 4]);
/// let tensor_3d: Tensor<Dim<3>> = Tensor::zeros([2, 3, 4]);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim<const N: usize>;

// Blanket implementation for all Dim<N>
// Smaller and Larger use DimDyn as fallback since we can't do N-1/N+1 in stable Rust
impl<const N: usize> Dimension for Dim<N> {
    const NDIM: Option<usize> = Some(N);
    type Smaller = DimDyn; // Can't compute N-1 at type level in stable Rust
    type Larger = DimDyn; // Can't compute N+1 at type level in stable Rust

    fn ndim(&self) -> usize {
        N
    }
}

/// Trait for dimensions that have a statically known smaller dimension
///
/// This enables type-safe dimension reduction operations.
pub trait HasSmaller: Dimension {
    type Smaller: Dimension;
}

/// Trait for dimensions that have a statically known larger dimension
///
/// This enables type-safe dimension expansion operations.
pub trait HasLarger: Dimension {
    type Larger: Dimension;
}

// Implement HasSmaller for Dim<1> through Dim<6>
impl HasSmaller for Dim<1> {
    type Smaller = Dim<0>;
}
impl HasSmaller for Dim<2> {
    type Smaller = Dim<1>;
}
impl HasSmaller for Dim<3> {
    type Smaller = Dim<2>;
}
impl HasSmaller for Dim<4> {
    type Smaller = Dim<3>;
}
impl HasSmaller for Dim<5> {
    type Smaller = Dim<4>;
}
impl HasSmaller for Dim<6> {
    type Smaller = Dim<5>;
}

// Implement HasLarger for Dim<0> through Dim<5>
impl HasLarger for Dim<0> {
    type Larger = Dim<1>;
}
impl HasLarger for Dim<1> {
    type Larger = Dim<2>;
}
impl HasLarger for Dim<2> {
    type Larger = Dim<3>;
}
impl HasLarger for Dim<3> {
    type Larger = Dim<4>;
}
impl HasLarger for Dim<4> {
    type Larger = Dim<5>;
}
impl HasLarger for Dim<5> {
    type Larger = Dim<6>;
}

/// Dynamic dimension type for runtime-determined dimensionality
///
/// Use this when the number of dimensions isn't known at compile time.
/// This is less type-safe than `Dim<N>` but more flexible.
///
/// # Examples
///
/// ```ignore
/// use harp::tensor::DimDyn;
///
/// let dim = DimDyn::new(3);
/// assert_eq!(dim.ndim(), 3);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct DimDyn(usize);

impl DimDyn {
    /// Create a new dynamic dimension with the specified number of dimensions
    pub fn new(ndim: usize) -> Self {
        Self(ndim)
    }

    /// Convert from a static dimension
    pub fn from_static<const N: usize>(_dim: Dim<N>) -> Self {
        Self(N)
    }
}

impl Dimension for DimDyn {
    const NDIM: Option<usize> = None;
    type Smaller = DimDyn; // Dynamic stays dynamic
    type Larger = DimDyn;

    fn ndim(&self) -> usize {
        self.0
    }
}

// Convenient type aliases for common dimensions
/// Scalar (0-dimensional tensor)
pub type Dim0 = Dim<0>;
/// 1-dimensional tensor (vector)
pub type Dim1 = Dim<1>;
/// 2-dimensional tensor (matrix)
pub type Dim2 = Dim<2>;
/// 3-dimensional tensor
pub type Dim3 = Dim<3>;
/// 4-dimensional tensor (common for batched images: NCHW)
pub type Dim4 = Dim<4>;
/// 5-dimensional tensor
pub type Dim5 = Dim<5>;
/// 6-dimensional tensor
pub type Dim6 = Dim<6>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_dimension() {
        let dim2 = Dim::<2>;
        assert_eq!(dim2.ndim(), 2);
        assert!(dim2.is_static());
        assert_eq!(Dim::<2>::NDIM, Some(2));
    }

    #[test]
    fn test_dynamic_dimension() {
        let dim = DimDyn::new(3);
        assert_eq!(dim.ndim(), 3);
        assert!(!dim.is_static());
        assert_eq!(DimDyn::NDIM, None);
    }

    #[test]
    fn test_dim_aliases() {
        let dim0: Dim0 = Dim::<0>;
        let dim1: Dim1 = Dim::<1>;
        let dim2: Dim2 = Dim::<2>;
        let dim3: Dim3 = Dim::<3>;
        let dim4: Dim4 = Dim::<4>;

        assert_eq!(dim0.ndim(), 0);
        assert_eq!(dim1.ndim(), 1);
        assert_eq!(dim2.ndim(), 2);
        assert_eq!(dim3.ndim(), 3);
        assert_eq!(dim4.ndim(), 4);
    }

    #[test]
    fn test_from_static() {
        let static_dim = Dim::<3>;
        let dyn_dim = DimDyn::from_static(static_dim);
        assert_eq!(dyn_dim.ndim(), 3);
    }
}

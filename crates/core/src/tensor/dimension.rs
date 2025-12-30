//! Dimension trait and implementations for static/dynamic dimension management
//!
//! This module provides compile-time dimension checking for tensors using const generics.

use ndarray::{Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
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
/// use harp_core::tensor::{Dim, DimDyn, Dimension};
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

    /// The corresponding ndarray dimension type
    ///
    /// This enables type-safe conversion between Tensor and ndarray:
    /// - Dim0 ↔ Ix0
    /// - Dim1 ↔ Ix1
    /// - Dim2 ↔ Ix2
    /// - etc.
    /// - DimDyn ↔ IxDyn
    type NdArrayDim: ndarray::Dimension;

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
/// use harp_core::tensor::Dim;
///
/// // Create tensors with specific dimensions
/// let tensor_1d: Tensor<f32, Dim<1>> = Tensor::zeros([10]);
/// let tensor_2d: Tensor<Dim<2>> = Tensor::zeros([3, 4]);
/// let tensor_3d: Tensor<Dim<3>> = Tensor::zeros([2, 3, 4]);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim<const N: usize>;

// Specific implementations for Dim<0> through Dim<6> with correct Smaller/Larger types
impl Dimension for Dim<0> {
    const NDIM: Option<usize> = Some(0);
    type Smaller = DimDyn; // No smaller than scalar
    type Larger = Dim<1>;
    type NdArrayDim = Ix0;

    fn ndim(&self) -> usize {
        0
    }
}

impl Dimension for Dim<1> {
    const NDIM: Option<usize> = Some(1);
    type Smaller = Dim<0>;
    type Larger = Dim<2>;
    type NdArrayDim = Ix1;

    fn ndim(&self) -> usize {
        1
    }
}

impl Dimension for Dim<2> {
    const NDIM: Option<usize> = Some(2);
    type Smaller = Dim<1>;
    type Larger = Dim<3>;
    type NdArrayDim = Ix2;

    fn ndim(&self) -> usize {
        2
    }
}

impl Dimension for Dim<3> {
    const NDIM: Option<usize> = Some(3);
    type Smaller = Dim<2>;
    type Larger = Dim<4>;
    type NdArrayDim = Ix3;

    fn ndim(&self) -> usize {
        3
    }
}

impl Dimension for Dim<4> {
    const NDIM: Option<usize> = Some(4);
    type Smaller = Dim<3>;
    type Larger = Dim<5>;
    type NdArrayDim = Ix4;

    fn ndim(&self) -> usize {
        4
    }
}

impl Dimension for Dim<5> {
    const NDIM: Option<usize> = Some(5);
    type Smaller = Dim<4>;
    type Larger = Dim<6>;
    type NdArrayDim = Ix5;

    fn ndim(&self) -> usize {
        5
    }
}

impl Dimension for Dim<6> {
    const NDIM: Option<usize> = Some(6);
    type Smaller = Dim<5>;
    type Larger = Dim<7>;
    type NdArrayDim = Ix6;

    fn ndim(&self) -> usize {
        6
    }
}

impl Dimension for Dim<7> {
    const NDIM: Option<usize> = Some(7);
    type Smaller = Dim<6>;
    type Larger = Dim<8>;
    type NdArrayDim = IxDyn; // No Ix7 in ndarray

    fn ndim(&self) -> usize {
        7
    }
}

impl Dimension for Dim<8> {
    const NDIM: Option<usize> = Some(8);
    type Smaller = Dim<7>;
    type Larger = DimDyn; // No larger defined
    type NdArrayDim = IxDyn; // No Ix8 in ndarray

    fn ndim(&self) -> usize {
        8
    }
}

/// Dynamic dimension type for runtime-determined dimensionality
///
/// Use this when the number of dimensions isn't known at compile time.
/// This is less type-safe than `Dim<N>` but more flexible.
///
/// # Examples
///
/// ```ignore
/// use harp_core::tensor::DimDyn;
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
    type NdArrayDim = IxDyn;

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
/// 7-dimensional tensor
pub type Dim7 = Dim<7>;
/// 8-dimensional tensor (used for 3D unfold: [N, C, out_H, out_W, out_D, kH, kW, kD])
pub type Dim8 = Dim<8>;

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

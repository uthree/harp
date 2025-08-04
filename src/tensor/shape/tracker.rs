//! Provides `ShapeTracker`, a utility for tracking the relationship between
//! a tensor's logical shape and its physical memory layout.
//!
//! This is crucial for handling views, permutations, and other non-contiguous
//! tensor operations without copying data. The tracker holds the symbolic shape
//! and the strides needed to compute the memory offset for any given index.

use crate::tensor::shape::expr::Expr;

/// Tracks the shape and strides of a tensor view.
///
/// A `ShapeTracker` instance represents a specific "view" into a tensor's data.
/// It allows calculating the flat memory offset for a given multidimensional index.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeTracker {
    shape: Vec<Expr>,   // logical shape for each axis.
    strides: Vec<Expr>, // actual memory offsets for each axis.
}

impl ShapeTracker {
    /// Creates a new `ShapeTracker` for a contiguous tensor.
    ///
    /// For a contiguous tensor, the strides are calculated based on the shape,
    /// assuming a standard row-major memory layout.
    ///
    /// # Arguments
    ///
    /// * `shape` - The logical shape of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::tensor::shape::tracker::ShapeTracker;
    /// use harp::tensor::shape::expr::Expr;
    ///
    /// // For a tensor of shape [10, 20]
    /// let tracker = ShapeTracker::new(vec![10.into(), 20.into()]);
    /// // Stride for axis 0 is 20, stride for axis 1 is 1.
    /// assert_eq!(tracker.strides(), &[20.into(), 1.into()]);
    /// ```
    pub fn new(shape: Vec<Expr>) -> Self {
        let mut strides = vec![Expr::from(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1].clone() * shape[i + 1].clone();
        }
        Self {
            shape: shape.iter().map(|sh| sh.clone().simplify()).collect(),
            strides: strides.iter().map(|s| s.clone().simplify()).collect(),
        }
    }

    /// Returns the number of dimensions of the tensor.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the logical shape of the tensor.
    pub fn shape(&self) -> &[Expr] {
        &self.shape
    }

    /// Returns the strides for each axis of the tensor.
    pub fn strides(&self) -> &[Expr] {
        &self.strides
    }

    /// Permutes the axes of the tensor view.
    ///
    /// This operation does not change the underlying data but modifies the
    /// shape and strides to reflect the new axis order.
    ///
    /// # Arguments
    ///
    /// * `axes` - A vector specifying the new order of the axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::tensor::shape::tracker::ShapeTracker;
    ///
    /// let tracker = ShapeTracker::new(vec![10.into(), 20.into()]);
    /// // Permute axes from (0, 1) to (1, 0)
    /// let permuted_tracker = tracker.permute(vec![1, 0]);
    /// assert_eq!(permuted_tracker.shape(), &[20.into(), 10.into()]);
    /// assert_eq!(permuted_tracker.strides(), &[1.into(), 20.into()]);
    /// ```
    pub fn permute(self, axes: Vec<usize>) -> Self {
        assert!(self.ndim() == axes.len());
        let mut new_shape = vec![];
        let mut new_strides = vec![];
        for axis in axes.iter() {
            new_shape.push(self.shape[*axis].clone().simplify());
            new_strides.push(self.strides[*axis].clone().simplify());
        }
        ShapeTracker {
            shape: new_shape,
            strides: new_strides,
        }
    }
}

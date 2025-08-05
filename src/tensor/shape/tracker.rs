//! Provides `ShapeTracker`, a utility for tracking the relationship between
//! a tensor's logical shape and its physical memory layout.
//!
//! This is crucial for handling views, permutations, and other non-contiguous
//! tensor operations without copying data. The tracker holds the symbolic shape
//! and the strides needed to compute the memory offset for any given index.

use crate::ast::AstNode;
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

    /// Checks if the tensor view is contiguous in memory.
    ///
    /// A view is contiguous if its strides match the standard row-major layout
    /// for its shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::tensor::shape::tracker::ShapeTracker;
    ///
    /// let contiguous_tracker = ShapeTracker::new(vec![10.into(), 20.into()]);
    /// assert!(contiguous_tracker.is_contiguous());
    ///
    /// let permuted_tracker = contiguous_tracker.permute(vec![1, 0]);
    /// assert!(!permuted_tracker.is_contiguous());
    /// ```
    pub fn is_contiguous(&self) -> bool {
        let contiguous_strides = ShapeTracker::new(self.shape.clone()).strides;
        self.strides == contiguous_strides
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

    /// Adds a new dimension of size 1 at a specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The position where the new axis is inserted.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::tensor::shape::tracker::ShapeTracker;
    ///
    /// let tracker = ShapeTracker::new(vec![10.into(), 20.into()]);
    /// // Unsqueeze at axis 1 to get shape [10, 1, 20]
    /// let unsqueezed = tracker.unsqueeze(1);
    /// assert_eq!(unsqueezed.shape(), &[10.into(), 1.into(), 20.into()]);
    /// assert_eq!(unsqueezed.strides(), &[20.into(), 0.into(), 1.into()]);
    /// ```
    pub fn unsqueeze(mut self, axis: usize) -> Self {
        assert!(axis <= self.ndim());
        self.shape.insert(axis, 1.into());
        self.strides.insert(axis, 0.into());
        self
    }

    /// Removes a dimension of size 1 at a specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove. Must be a dimension of size 1.
    ///
    /// # Panics
    ///
    /// Panics if the dimension at the specified axis is not equal to 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::tensor::shape::tracker::ShapeTracker;
    ///
    /// let tracker = ShapeTracker::new(vec![10.into(), 1.into(), 20.into()]);
    /// // Squeeze axis 1 to get shape [10, 20]
    /// let squeezed = tracker.squeeze(1);
    /// assert_eq!(squeezed.shape(), &[10.into(), 20.into()]);
    /// assert_eq!(squeezed.strides(), &[20.into(), 1.into()]);
    /// ```
    pub fn squeeze(mut self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        assert_eq!(
            self.shape[axis],
            1.into(),
            "can only squeeze an axis of size 1"
        );
        self.shape.remove(axis);
        self.strides.remove(axis);
        self
    }

    /// Expands the tensor to a new shape without copying data.
    ///
    /// This is done by setting the stride of the expanded dimension to 0.
    /// The expanded dimensions must have been of size 1.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The target shape.
    ///
    /// # Panics
    ///
    /// Panics if the expansion is invalid (e.g., trying to expand a dimension
    /// that is not of size 1).
    pub fn expand(self, new_shape: Vec<Expr>) -> Self {
        assert_eq!(
            self.ndim(),
            new_shape.len(),
            "expand must not change the number of dimensions"
        );

        let mut new_strides = self.strides.clone();
        for (i, (old_dim, new_dim)) in self.shape.iter().zip(&new_shape).enumerate() {
            if old_dim == new_dim {
                continue;
            }
            assert_eq!(*old_dim, 1.into(), "can only expand a dimension of size 1");
            new_strides[i] = 0.into();
        }

        ShapeTracker {
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Calculates the memory offset expression for a given set of loop variables.
    ///
    /// This is the core function for translating logical indices into a flat
    /// memory offset. It computes `sum(loop_vars[i] * strides[i])`.
    pub fn offset_expr(&self, loop_vars: &[String]) -> Expr {
        assert_eq!(self.ndim(), loop_vars.len());
        self.strides
            .iter()
            .zip(loop_vars.iter())
            .fold(Expr::from(0), |acc, (stride, var)| {
                acc + stride.clone() * Expr::from(AstNode::var(var))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_expand_simple() {
        let tracker = ShapeTracker::new(vec![1.into(), 10.into()]);
        let expanded = tracker.expand(vec![5.into(), 10.into()]);
        assert_eq!(expanded.shape(), &[5.into(), 10.into()]);
        assert_eq!(expanded.strides(), &[0.into(), 1.into()]);
    }

    #[test]
    fn test_expand_multiple_dims() {
        let tracker = ShapeTracker::new(vec![1.into(), 20.into(), 1.into()]);
        let expanded = tracker.expand(vec![10.into(), 20.into(), 30.into()]);
        assert_eq!(expanded.shape(), &[10.into(), 20.into(), 30.into()]);
        assert_eq!(expanded.strides(), &[0.into(), 1.into(), 0.into()]);
    }

    #[test]
    #[should_panic]
    fn test_expand_invalid_dim() {
        let tracker = ShapeTracker::new(vec![2.into(), 10.into()]);
        // This should panic because the first dimension is not 1.
        tracker.expand(vec![5.into(), 10.into()]);
    }

    #[test]
    #[should_panic]
    fn test_expand_wrong_ndim() {
        let tracker = ShapeTracker::new(vec![1.into(), 10.into()]);
        // This should panic because the number of dimensions is different.
        tracker.expand(vec![5.into(), 10.into(), 1.into()]);
    }
}

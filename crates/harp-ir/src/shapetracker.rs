//! Defines the ShapeTracker for managing tensor views.

use crate::{Dim, Shape};

/// Tracks the mapping from a logical tensor shape to its physical memory layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeTracker {
    pub shape: Shape,
    pub strides: Vec<usize>, // For now, strides are fixed.
    pub offset: usize,
}

impl ShapeTracker {
    /// Creates a new `ShapeTracker` for a contiguous tensor.
    pub fn new(shape: Shape) -> Self {
        let mut strides = vec![1; shape.dims.len()];
        for i in (0..shape.dims.len() - 1).rev() {
            let dim = match shape.dims[i + 1] {
                Dim::Fixed(d) => d,
                // Symbolic dimensions make contiguous strides complex.
                // For now, we panic. A real implementation would need symbolic math.
                Dim::Symbolic(_) => panic!("Symbolic dimensions are not yet supported for stride calculation."),
            };
            strides[i] = strides[i + 1] * dim;
        }
        Self {
            shape,
            strides,
            offset: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_strides() {
        // Shape (3, 4, 5)
        let shape = Shape::new(vec![Dim::Fixed(3), Dim::Fixed(4), Dim::Fixed(5)]);
        let tracker = ShapeTracker::new(shape);

        // Strides should be (4*5, 5, 1) = (20, 5, 1)
        assert_eq!(tracker.strides, vec![20, 5, 1]);
        assert_eq!(tracker.offset, 0);
    }

    #[test]
    #[should_panic]
    fn test_symbolic_shape_panic() {
        let shape = Shape::new(vec![Dim::Fixed(3), Dim::Symbolic("N".to_string())]);
        // This should panic because we can't calculate strides with symbolic dims yet.
        ShapeTracker::new(shape);
    }
}

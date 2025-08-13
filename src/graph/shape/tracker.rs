use std::collections::HashSet;

use super::expr::Expr;

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeTracker {
    shape: Vec<Expr>,
    strides: Vec<Expr>,
    offset: Expr,
}

impl ShapeTracker {
    pub fn new(shape: Vec<impl Into<Expr> + Clone>) -> Self {
        let shape: Vec<Expr> = shape.iter().map(|d| d.clone().into().simplify()).collect();
        let mut strides = vec![Expr::from(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = (strides[i + 1].clone() * shape[i + 1].clone()).simplify();
        }
        Self {
            shape,
            strides,
            offset: 0.into(),
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

    pub fn permute(self, axes: Vec<usize>) -> Self {
        assert!(self.ndim() == axes.len());
        let axes_set: HashSet<_> = axes.iter().collect();
        assert!(axes_set.len() == axes.len(), "duplicate axis in permute");
        let mut new_shape = vec![];
        let mut new_strides = vec![];
        for axis in axes.iter() {
            new_shape.push(self.shape[*axis].clone().simplify());
            new_strides.push(self.strides[*axis].clone().simplify());
        }
        ShapeTracker {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        }
    }

    pub fn unsqueeze(mut self, axis: usize) -> Self {
        assert!(axis <= self.ndim());
        self.shape.insert(axis, 1.into());
        self.strides.insert(axis, 0.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::shape::expr::Expr;
    use rstest::rstest;

    #[rstest]
    #[case(vec![1, 2, 3], vec![6, 3, 1])]
    #[case(vec![4, 5], vec![5, 1])]
    #[case(vec![10], vec![1])]
    fn test_new_shape_tracker(
        #[case] shape: Vec<usize>,
        #[case] expected_strides_values: Vec<usize>,
    ) {
        let tracker = ShapeTracker::new(shape.clone());
        let expected_shape: Vec<Expr> = shape.iter().map(|&d| Expr::from(d)).collect();
        let expected_strides: Vec<Expr> = expected_strides_values
            .iter()
            .map(|&s| Expr::from(s))
            .collect();

        assert_eq!(tracker.shape(), &expected_shape);
        assert_eq!(tracker.ndim(), shape.len());
        assert_eq!(tracker.strides(), &expected_strides);
    }

    #[test]
    fn test_permute() {
        let tracker = ShapeTracker::new(vec![1, 2, 3]);
        let permuted = tracker.permute(vec![2, 0, 1]);
        assert_eq!(
            permuted.shape(),
            &[Expr::from(3), Expr::from(1), Expr::from(2)]
        );
        assert_eq!(
            permuted.strides(),
            &[Expr::from(1), Expr::from(6), Expr::from(3)]
        );
    }

    #[test]
    #[should_panic]
    fn test_permute_invalid_axes_len() {
        let tracker = ShapeTracker::new(vec![1, 2, 3]);
        tracker.permute(vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "duplicate axis in permute")]
    fn test_permute_duplicate_axes() {
        let tracker = ShapeTracker::new(vec![1, 2, 3]);
        tracker.permute(vec![0, 1, 1]);
    }

    #[rstest]
    #[case(0, vec![1, 1, 2, 3], vec![0, 6, 3, 1])]
    #[case(1, vec![1, 1, 2, 3], vec![6, 0, 3, 1])]
    #[case(3, vec![1, 2, 3, 1], vec![6, 3, 1, 0])]
    fn test_unsqueeze(
        #[case] axis: usize,
        #[case] expected_shape: Vec<usize>,
        #[case] expected_strides: Vec<usize>,
    ) {
        let tracker = ShapeTracker::new(vec![1, 2, 3]);
        let unsqueezed = tracker.unsqueeze(axis);
        let expected_shape_expr: Vec<Expr> =
            expected_shape.iter().map(|&d| Expr::from(d)).collect();
        let expected_strides_expr: Vec<Expr> =
            expected_strides.iter().map(|&s| Expr::from(s)).collect();
        assert_eq!(unsqueezed.shape(), &expected_shape_expr);
        assert_eq!(unsqueezed.strides(), &expected_strides_expr);
    }

    #[test]
    #[should_panic]
    fn test_unsqueeze_invalid_axis() {
        let tracker = ShapeTracker::new(vec![1, 2, 3]);
        tracker.unsqueeze(4);
    }
}

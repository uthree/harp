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
        let mut strides = vec![Expr::from(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1].clone() * shape[i + 1].clone();
        }
        Self {
            shape: shape.iter().map(|d| d.clone().into().simplify()).collect(),
            strides: strides.iter().map(|s| s.clone().simplify()).collect(),
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

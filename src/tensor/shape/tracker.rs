use crate::tensor::shape::expr::Expr;

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeTracker {
    shape: Vec<Expr>,   // logical shape for each axis.
    strides: Vec<Expr>, // actual memory offsets for each axis.
}

impl ShapeTracker {
    // initialize contiguous view from shape
    pub fn new(shape: Vec<Expr>) -> Self {
        let mut strides = vec![Expr::from(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1].clone() * shape[i + 1].clone();
        }
        Self {
            shape: shape
                .iter()
                .map(|sh| sh.clone().simplify())
                .collect(),
            strides: strides.iter().map(|s| s.clone().simplify()).collect(),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[Expr] {
        &self.shape
    }

    pub fn strides(&self) -> &[Expr] {
        &self.strides
    }

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

use crate::tensor::shape::expr::Expr;

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeTracker {
    shape: Vec<Expr>,   // logical shape for each axis.
    strides: Vec<Expr>, // actual memory offsets for each axis.
}

impl ShapeTracker {
    // initialize contiguous view from shape
    pub fn new(shape: Vec<Expr>) -> Self {
        let mut s = vec![];
        for d in shape.iter().rev() {
            s.push(d.clone().simplify());
        }
        ShapeTracker {
            shape: shape
                .clone()
                .iter()
                .map(|sh| sh.clone().simplify())
                .collect(),
            strides: s,
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
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

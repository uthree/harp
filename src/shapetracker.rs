//! ShapeTracker is a data structure that tracks the shape and memory layout of a tensor.
//! It is inspired by tinygrad's ShapeTracker.
//! It allows for efficient operations like reshape, permute, and slice without copying data.

use crate::uop::{UOp, DType};

#[derive(Clone, Debug, PartialEq)]
pub struct View {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub mask: Option<Vec<(usize, usize)>>,
    pub contiguous: bool,
}

impl View {
    pub fn new(
        shape: Vec<usize>,
        strides: Option<Vec<usize>>,
        offset: Option<usize>,
        mask: Option<Vec<(usize, usize)>>,
    ) -> Self {
        let strides = strides.unwrap_or_else(|| Self::default_strides(&shape));
        let contiguous = offset.is_none() && mask.is_none() && strides == Self::default_strides(&shape);
        Self {
            shape,
            strides,
            offset: offset.unwrap_or(0),
            mask,
            contiguous,
        }
    }

    fn default_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Generates the UOp expression for a given index.
    fn expr_indices(&self, indices: &[UOp]) -> UOp {
        assert_eq!(indices.len(), self.shape.len());
        let mut acc: Option<UOp> = None;
        for (i, st) in self.strides.iter().enumerate() {
            if self.shape[i] != 1 && *st != 0 {
                let term = if *st == 1 {
                    indices[i].clone()
                } else {
                    &indices[i] * &UOp::from(*st as u64)
                };
                if let Some(current_acc) = acc {
                    acc = Some(current_acc + term);
                } else {
                    acc = Some(term);
                }
            }
        }
        let mut result = acc.unwrap_or_else(|| UOp::from(0u64));
        if self.offset != 0 {
            result += UOp::from(self.offset as u64);
        }
        result
    }

    fn expr_node(&self, idx: &UOp) -> UOp {
        let mut acc: UOp = (self.offset as i32).into();
        let mut current_idx = idx.clone();

        for (shape, stride) in self.shape.iter().rev().zip(self.strides.iter().rev()) {
            let term = (current_idx.clone() % UOp::from(*shape as i32)) * UOp::from(*stride as i32);
            acc += term;
            current_idx /= UOp::from(*shape as i32);
        }
        acc
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShapeTracker {
    pub views: Vec<View>,
}

impl ShapeTracker {
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            views: vec![View::new(shape, None, None, None)],
        }
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.views.last().unwrap().shape
    }

    pub fn expr_indices(&self, mut idx: UOp) -> UOp {
        for view in self.views.iter().rev() {
            idx = view.expr_node(&idx);
        }
        idx
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let old_shape = self.shape();
        assert_eq!(
            old_shape.iter().product::<usize>(),
            new_shape.iter().product::<usize>(),
            "Reshape validation failed: element count must be the same"
        );
        // TODO: This is a simplification. A real reshape needs to intelligently
        // modify the view stack. For now, we just create a new contiguous tracker.
        // This will fail for chained view ops, e.g., reshape -> permute.
        ShapeTracker::new(new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uop::{DType, UOp};

    #[test]
    fn test_simple_tracker_expr() {
        let st = ShapeTracker::new(vec![10, 20]);
        let idx = UOp::var("i", DType::I32);
        let expr = st.expr_indices(idx);
        // Expected: i % 20 + (i / 20) * 20
        // This is a simplified check
        assert!(format!("{:?}", expr).contains("Mul"));
        assert!(format!("{:?}", expr).contains("Add"));
        assert!(format!("{:?}", expr).contains("Rem"));
    }
}

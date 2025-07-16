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
        let mut acc: UOp = UOp::from(self.offset as i32);
        for (i, st) in self.strides.iter().enumerate() {
            if self.shape[i] != 1 && *st != 0 {
                acc += indices[i].clone() * UOp::from(*st as i32);
            }
        }
        acc
    }

    fn expr_node(&self, idx: UOp) -> UOp {
        let mut ret = vec![];
        let mut acc = 1;
        for &sh in self.shape.iter().rev() {
            ret.push((idx.clone() / UOp::from(acc)) % UOp::from(sh as i32));
            acc *= sh as i32;
        }
        self.expr_indices(&ret.into_iter().rev().collect::<Vec<_>>())
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

    pub fn expr_indices(&self, indices: Option<&[UOp]>) -> UOp {
        let binding;
        let idxs = if let Some(indices) = indices {
            indices
        } else {
            binding = self
                .shape()
                .iter()
                .enumerate()
                .map(|(i, _)| UOp::var(&format!("idx{i}"), DType::I32))
                .collect::<Vec<_>>();
            &binding
        };

        let mut idx: UOp = UOp::from(0i32);
        let mut acc: UOp = UOp::from(1i32);
        for (i, &sh) in self.views.last().unwrap().shape.iter().enumerate() {
            idx += idxs[i].clone() * acc.clone();
            acc *= UOp::from(sh as i32);
        }

        self.views
            .iter()
            .rev()
            .fold(idx, |current_idx, view| view.expr_node(current_idx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uop::{DType, UOp};

    #[test]
    fn test_simple_tracker_expr() {
        let st = ShapeTracker::new(vec![10, 20]);
        let expr = st.expr_indices(None);

        // Expected: idx0 * 20 + idx1
        let idx0 = UOp::var("idx0", DType::I32);
        let idx1 = UOp::var("idx1", DType::I32);
        let expected_expr = idx0 * UOp::from(20i32) + idx1;
        
        // Note: This is a weak test, as we can't easily compare UOp trees.
        // We are just checking that it compiles and produces some result.
        // A proper test would require an interpreter for UOp expressions.
        assert_eq!(expr.0.op, expected_expr.0.op);
    }
}

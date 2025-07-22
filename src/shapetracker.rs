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
        let mut ret = vec![];
        let mut acc: u64 = 1;
        for &sh in self.shape.iter().rev() {
            ret.push((idx / &UOp::from(acc)) % &UOp::from(sh as u64));
            acc *= sh as u64;
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
                .map(|(i, _)| UOp::var(&format!("idx{i}"), DType::U64))
                .collect::<Vec<_>>();
            &binding
        };
        self.views.last().unwrap().expr_indices(idxs)
    }

    pub fn expr_node(&self, idx: &UOp) -> UOp {
        self.views.last().unwrap().expr_node(idx)
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
        let idxs = vec![UOp::var("idx0", DType::U64), UOp::var("idx1", DType::U64)];
        let expr = st.expr_indices(Some(&idxs));

        // Expected: idx0 * 20 + idx1
        let expected_expr = &idxs[0] * &UOp::from(20u64) + &idxs[1];
        
        // This is a weak test. A proper test would require an interpreter.
        assert_eq!(format!("{:?}", expr), format!("{:?}", expected_expr));
    }
}

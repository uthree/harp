//! ShapeTracker is a data structure that tracks the shape and memory layout of a tensor.
//! It is inspired by tinygrad's ShapeTracker.
//! It allows for efficient operations like reshape, permute, and slice without copying data.

use crate::uop::{DType, UOp};

/// Represents a single view into a tensor's data.
///
/// A `View` defines a logical interpretation of a contiguous block of memory. It includes
/// the shape, strides, and an offset, allowing for operations like slicing and
/// permuting without copying the underlying data.
#[derive(Clone, Debug, PartialEq)]
pub struct View {
    /// The logical shape of the tensor view.
    pub shape: Vec<usize>,
    /// The number of elements to skip in memory to move one unit along each dimension.
    pub strides: Vec<usize>,
    /// The offset from the beginning of the underlying buffer.
    pub offset: usize,
    /// An optional mask to limit the valid range of indices for each dimension.
    pub mask: Option<Vec<(usize, usize)>>,
    /// A flag indicating if the view is contiguous in memory.
    pub contiguous: bool,
}

impl View {
    /// Creates a new `View`.
    pub fn new(
        shape: Vec<usize>,
        strides: Option<Vec<usize>>,
        offset: Option<usize>,
        mask: Option<Vec<(usize, usize)>>,
    ) -> Self {
        let strides = strides.unwrap_or_else(|| Self::default_strides(&shape));
        let contiguous =
            offset.is_none() && mask.is_none() && strides == Self::default_strides(&shape);
        Self {
            shape,
            strides,
            offset: offset.unwrap_or(0),
            mask,
            contiguous,
        }
    }

    /// Calculates the default strides for a given shape, assuming contiguous memory.
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

    /// Generates the `UOp` expression for calculating the memory index from logical indices.
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

    /// Generates the `UOp` expression for calculating the memory index from a single flat index.
    ///
    /// This method includes a fast path for contiguous tensors. If the view is
    /// contiguous, it directly returns the index `idx`, avoiding complex calculations.
    /// Otherwise, it decomposes the flat index into multi-dimensional indices and
    /// calculates the physical address using strides.
    fn expr_node(&self, idx: &UOp) -> UOp {
        // Fast path for contiguous tensors.
        if self.contiguous {
            return idx.clone();
        }

        // General path for non-contiguous tensors.
        let mut ret = vec![];
        let mut acc: u64 = 1;
        for &sh in self.shape.iter().rev() {
            ret.push((idx / &UOp::from(acc)) % &UOp::from(sh as u64));
            acc *= sh as u64;
        }
        self.expr_indices(&ret.into_iter().rev().collect::<Vec<_>>())
    }

    /// Attempts to reshape the view without creating a new contiguous view.
    /// This is possible if the reshape operation only involves merging and splitting
    /// dimensions that are contiguous relative to each other.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Option<View> {
        if self.shape.iter().product::<usize>() != new_shape.iter().product() {
            return None;
        }
        if self.shape == new_shape {
            return Some(self.clone());
        }

        let mut new_strides = vec![0; new_shape.len()];
        let mut old_idx = 0;
        let mut new_idx = 0;

        while old_idx < self.shape.len() && new_idx < new_shape.len() {
            let old_dims_start = old_idx;
            let new_dims_start = new_idx;
            let mut prod_old = 1;
            let mut prod_new = 1;

            // Find a block of dimensions that have the same number of elements
            loop {
                if prod_old == prod_new && prod_old > 1 {
                    break;
                }
                if old_idx >= self.shape.len() || new_idx >= new_shape.len() {
                    break;
                }
                if prod_old < prod_new {
                    prod_old *= self.shape[old_idx];
                    old_idx += 1;
                } else {
                    prod_new *= new_shape[new_idx];
                    new_idx += 1;
                }
            }

            if prod_old != prod_new {
                return None;
            }

            let old_slice_shape = &self.shape[old_dims_start..old_idx];
            let old_slice_strides = &self.strides[old_dims_start..old_idx];
            let new_slice_shape = &new_shape[new_dims_start..new_idx];

            if old_slice_shape.len() == 1 && new_slice_shape.len() > 1 {
                // EXPAND: e.g., [6] -> [2, 3]
                // The new dimensions are treated as a contiguous block, with strides
                // calculated based on the original dimension's stride.
                new_strides[new_dims_start + new_slice_shape.len() - 1] = old_slice_strides[0];
                for i in (0..new_slice_shape.len() - 1).rev() {
                    new_strides[new_dims_start + i] =
                        new_strides[new_dims_start + i + 1] * new_slice_shape[i + 1];
                }
            } else if old_slice_shape.len() > 1 && new_slice_shape.len() == 1 {
                // MERGE: e.g., [2, 3] -> [6]
                // Check for contiguity within the slice being merged.
                for i in 0..old_slice_shape.len() - 1 {
                    if old_slice_strides[i] != old_slice_strides[i + 1] * old_slice_shape[i + 1] {
                        return None; // Not contiguous, cannot merge.
                    }
                }
                new_strides[new_dims_start] = old_slice_strides[old_slice_shape.len() - 1];
            } else if old_slice_shape.len() == 1 && new_slice_shape.len() == 1 {
                // ONE-TO-ONE
                new_strides[new_dims_start] = old_slice_strides[0];
            } else {
                // Many-to-many reshape (e.g., [2, 3] -> [3, 2]) is a permute.
                // This logic doesn't handle it. Fail.
                return None;
            }
        }

        if old_idx != self.shape.len() || new_idx != new_shape.len() {
            return None;
        }

        Some(View::new(
            new_shape,
            Some(new_strides),
            Some(self.offset),
            self.mask.clone(),
        ))
    }
}

/// Tracks the shape and memory layout of a tensor through a stack of `View`s.
///
/// This allows for lazy evaluation of shape operations. Only the final view in the
/// stack represents the current logical shape of the tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct ShapeTracker {
    pub views: Vec<View>,
}

impl ShapeTracker {
    /// Creates a new `ShapeTracker` for a tensor of a given shape.
    pub fn new(shape: Vec<usize>) -> Self {
        Self {
            views: vec![View::new(shape, None, None, None)],
        }
    }

    /// Returns the current logical shape of the tensor.
    pub fn shape(&self) -> &Vec<usize> {
        &self.views.last().unwrap().shape
    }

    /// Generates the `UOp` expression for a given set of logical indices.
    pub fn expr_indices(&self, indices: Option<&[UOp]>) -> UOp {
        let binding;
        let idxs = if let Some(indices) = indices {
            indices
        } else {
            // If no indices are provided, create symbolic ones.
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

    /// Generates the `UOp` expression for a single flat index.
    pub fn expr_node(&self, idx: &UOp) -> UOp {
        self.views.last().unwrap().expr_node(idx)
    }

    /// Creates a new `ShapeTracker` representing a reshaped tensor.
    ///
    /// This method attempts to perform a "smart" reshape that avoids creating
    /// a new contiguous view if possible. If the reshape is complex (e.g.,
    /// requires a permute), it falls back to creating a new, contiguous view.
    ///
    /// # Panics
    /// Panics if the total number of elements in `new_shape` is not equal to the
    /// total number of elements in the current shape.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let old_shape = self.shape();
        assert_eq!(
            old_shape.iter().product::<usize>(),
            new_shape.iter().product::<usize>(),
            "Reshape validation failed: element count must be the same"
        );

        if old_shape == &new_shape {
            return self.clone();
        }

        let view = self.views.last().unwrap();

        // Attempt to perform a "smart" reshape on the current view.
        if let Some(new_view) = view.reshape(new_shape.clone()) {
            // If successful, create a new ShapeTracker with just the new view.
            // A more advanced implementation might stack views.
            return Self {
                views: vec![new_view],
            };
        }

        // Fallback for complex cases: create a new contiguous view.
        // This implies that the data would need to be made contiguous before this
        // new shape is applied, which is what the old implementation did.
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

        // This is a weak test. A proper test would require an interpreter for UOps.
        // For now, we compare the debug string representation.
        assert_eq!(format!("{:?}", expr), format!("{:?}", expected_expr));
    }
}

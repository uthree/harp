//! Shape and view management for tensors.

use std::fmt;

/// Represents the shape (dimensions) of a tensor.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Creates a new shape from dimensions.
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Shape(dims.into())
    }

    /// Creates a scalar shape (empty dimensions).
    pub fn scalar() -> Self {
        Shape(vec![])
    }

    /// Returns the dimensions as a slice.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Returns the number of dimensions (rank).
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Returns true if this is a scalar (rank 0).
    pub fn is_scalar(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.0.iter().product::<usize>().max(1)
    }

    /// Returns the size of the i-th dimension.
    pub fn dim(&self, i: usize) -> usize {
        self.0[i]
    }

    /// Computes the broadcast shape between self and other.
    /// Returns None if shapes are incompatible.
    pub fn broadcast(&self, other: &Shape) -> Option<Shape> {
        let max_rank = self.rank().max(other.rank());
        let mut result = Vec::with_capacity(max_rank);

        for i in 0..max_rank {
            let d1 = if i < max_rank - self.rank() {
                1
            } else {
                self.0[i - (max_rank - self.rank())]
            };
            let d2 = if i < max_rank - other.rank() {
                1
            } else {
                other.0[i - (max_rank - other.rank())]
            };

            if d1 == d2 {
                result.push(d1);
            } else if d1 == 1 {
                result.push(d2);
            } else if d2 == 1 {
                result.push(d1);
            } else {
                return None;
            }
        }

        Some(Shape(result))
    }

    /// Returns strides for a contiguous layout.
    pub fn strides(&self) -> Vec<usize> {
        if self.0.is_empty() {
            return vec![];
        }
        let mut strides = vec![1; self.rank()];
        for i in (0..self.rank() - 1).rev() {
            strides[i] = strides[i + 1] * self.0[i + 1];
        }
        strides
    }

    /// Computes the flat index from a multi-dimensional index.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        let strides = self.strides();
        indices
            .iter()
            .zip(strides.iter())
            .map(|(i, s)| i * s)
            .sum()
    }

    /// Computes the multi-dimensional index from a flat index.
    pub fn multi_index(&self, mut flat: usize) -> Vec<usize> {
        let strides = self.strides();
        let mut indices = Vec::with_capacity(self.rank());
        for &stride in &strides {
            indices.push(flat / stride);
            flat %= stride;
        }
        indices
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.0)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, d) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", d)?;
        }
        if self.0.len() == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")
    }
}

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Shape(v)
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Shape(v.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Self {
        Shape(v.to_vec())
    }
}

impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Shape::scalar()
    }
}

impl From<usize> for Shape {
    fn from(v: usize) -> Self {
        Shape(vec![v])
    }
}

impl From<(usize,)> for Shape {
    fn from(v: (usize,)) -> Self {
        Shape(vec![v.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(v: (usize, usize)) -> Self {
        Shape(vec![v.0, v.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(v: (usize, usize, usize)) -> Self {
        Shape(vec![v.0, v.1, v.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(v: (usize, usize, usize, usize)) -> Self {
        Shape(vec![v.0, v.1, v.2, v.3])
    }
}

/// Represents a view into a tensor with stride and offset information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct View {
    pub shape: Shape,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub mask: Option<Vec<(usize, usize)>>,
}

impl View {
    /// Creates a contiguous view for the given shape.
    pub fn contiguous(shape: Shape) -> Self {
        let strides = shape.strides().into_iter().map(|s| s as isize).collect();
        View {
            shape,
            strides,
            offset: 0,
            mask: None,
        }
    }

    /// Creates a view for a scalar.
    pub fn scalar() -> Self {
        View {
            shape: Shape::scalar(),
            strides: vec![],
            offset: 0,
            mask: None,
        }
    }

    /// Returns true if this view is contiguous.
    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 || self.mask.is_some() {
            return false;
        }
        let expected = self.shape.strides();
        self.strides
            .iter()
            .zip(expected.iter())
            .all(|(&a, &b)| a == b as isize)
    }

    /// Returns the number of elements in this view.
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Computes the physical index from a logical index.
    pub fn physical_index(&self, indices: &[usize]) -> Option<usize> {
        if let Some(ref mask) = self.mask {
            for (i, &(start, end)) in mask.iter().enumerate() {
                if indices[i] < start || indices[i] >= end {
                    return None;
                }
            }
        }

        let mut idx = self.offset as isize;
        for (i, &stride) in self.strides.iter().enumerate() {
            idx += indices[i] as isize * stride;
        }
        Some(idx as usize)
    }

    /// Creates a reshaped view.
    pub fn reshape(&self, new_shape: Shape) -> Option<View> {
        if self.shape.numel() != new_shape.numel() {
            return None;
        }
        if !self.is_contiguous() {
            return None;
        }
        Some(View::contiguous(new_shape))
    }

    /// Creates a permuted view.
    pub fn permute(&self, axes: &[usize]) -> View {
        let new_dims: Vec<_> = axes.iter().map(|&i| self.shape.dim(i)).collect();
        let new_strides: Vec<_> = axes.iter().map(|&i| self.strides[i]).collect();
        View {
            shape: Shape::new(new_dims),
            strides: new_strides,
            offset: self.offset,
            mask: None, // TODO: permute mask
        }
    }

    /// Creates an expanded view (broadcast).
    pub fn expand(&self, new_shape: Shape) -> Option<View> {
        if new_shape.rank() < self.shape.rank() {
            return None;
        }

        let rank_diff = new_shape.rank() - self.shape.rank();
        let mut new_strides = vec![0isize; new_shape.rank()];

        for (i, stride) in new_strides.iter_mut().enumerate() {
            if i < rank_diff {
                *stride = 0;
            } else {
                let old_i = i - rank_diff;
                let old_dim = self.shape.dim(old_i);
                let new_dim = new_shape.dim(i);
                if old_dim == new_dim {
                    *stride = self.strides[old_i];
                } else if old_dim == 1 {
                    *stride = 0;
                } else {
                    return None;
                }
            }
        }

        Some(View {
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            mask: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_basic() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.rank(), 3);
        assert_eq!(s.numel(), 24);
        assert_eq!(s.dim(1), 3);
    }

    #[test]
    fn test_shape_broadcast() {
        let a = Shape::new([3, 1]);
        let b = Shape::new([1, 4]);
        assert_eq!(a.broadcast(&b), Some(Shape::new([3, 4])));

        let a = Shape::new([2, 3]);
        let b = Shape::new([3]);
        assert_eq!(a.broadcast(&b), Some(Shape::new([2, 3])));

        let a = Shape::new([2, 3]);
        let b = Shape::new([4, 3]);
        assert_eq!(a.broadcast(&b), None);
    }

    #[test]
    fn test_shape_strides() {
        let s = Shape::new([2, 3, 4]);
        assert_eq!(s.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_shape_index() {
        let s = Shape::new([2, 3]);
        assert_eq!(s.flat_index(&[1, 2]), 5);
        assert_eq!(s.multi_index(5), vec![1, 2]);
    }

    #[test]
    fn test_view_contiguous() {
        let v = View::contiguous(Shape::new([2, 3]));
        assert!(v.is_contiguous());
        assert_eq!(v.strides, vec![3, 1]);
    }

    #[test]
    fn test_view_permute() {
        let v = View::contiguous(Shape::new([2, 3, 4]));
        let p = v.permute(&[2, 0, 1]);
        assert_eq!(p.shape.dims(), &[4, 2, 3]);
        assert_eq!(p.strides, vec![1, 12, 4]);
    }
}

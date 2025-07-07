use crate::shape::symbolic::Expr;
use std::fmt;

/// Tracks the shape and memory layout of a tensor using symbolic expressions.
///
/// `ShapeTracker` stores two main components:
/// - `map`: A vector of `Expr` representing how each dimension maps to a linear memory address.
/// - `max`: A vector of `Expr` representing the maximum extent of each dimension.
#[derive(Debug, PartialEq, Clone)]
pub struct ShapeTracker {
    /// Symbolic expressions defining the mapping from logical indices to linear memory offsets.
    pub map: Vec<Expr>,
    /// Symbolic expressions defining the maximum size of each dimension.
    pub max: Vec<Expr>,
}

impl ShapeTracker {
    /// Creates a new `ShapeTracker` for a contiguous tensor with the given dimensions.
    ///
    /// This function calculates the `map` (strides) and `max` (dimensions) for a tensor
    /// assuming a row-major, contiguous memory layout.
    ///
    /// # Arguments
    ///
    /// * `dims` - A `Vec<Expr>` representing the dimensions of the tensor.
    ///
    /// # Returns
    ///
    /// A new `ShapeTracker` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::shape::tracker::ShapeTracker;
    /// use harp::shape::symbolic::Expr;
    /// use harp::s;
    ///
    /// let dims = s![2, 3, 4];
    /// let tracker = ShapeTracker::full(dims);
    ///
    /// assert_eq!(tracker.max, s![2, 3, 4]);
    /// ```
    pub fn full(dims: Vec<Expr>) -> Self {
        // calculate maps and strides
        let mut alu: Expr = 1.into();
        let mut maps = vec![];
        let mut maxs = vec![];
        for d in dims.iter().rev() {
            maps.push((Expr::Index * alu.clone()).simplify());
            maxs.push(d.clone().simplify());
            alu = alu * d.clone();
        }
        let maps = maps.iter().rev().map(|m| m.to_owned()).collect::<Vec<_>>();
        let maxs = maxs.iter().rev().map(|m| m.to_owned()).collect::<Vec<_>>();
        ShapeTracker {
            max: maxs,
            map: maps,
        }
    }
}

impl From<Vec<usize>> for ShapeTracker {
    /// Creates a new `ShapeTracker` from a vector of integer dimensions.
    /// This is a convenience method that converts integers to `Expr::Int`.
    ///
    /// # Arguments
    ///
    /// * `dims` - A `Vec<usize>` representing the dimensions of the tensor.
    ///
    /// # Returns
    ///
    /// A new `ShapeTracker` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::shape::tracker::ShapeTracker;
    ///
    /// let tracker: ShapeTracker = vec![2, 3, 4].into();
    /// assert_eq!(tracker.max.len(), 3);
    /// assert_eq!(tracker.map.len(), 3);
    /// ```
    fn from(dims: Vec<usize>) -> Self {
        Self::full(dims.into_iter().map(|d| d.into()).collect())
    }
}

impl fmt::Display for ShapeTracker {
    /// Formats the `ShapeTracker` for display.
    ///
    /// This provides a human-readable representation of the shape and map.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "shape=[{}], map={}",
            self.max
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            self.map
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join(" + "),
        )
    }
}

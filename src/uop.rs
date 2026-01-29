//! UOp: Unified operation representation for the computation graph.

use std::sync::Arc;

use crate::dtype::{DType, ScalarValue};
use crate::ops::Ops;
use crate::shape::Shape;

/// Argument types for UOps.
#[derive(Debug, Clone)]
pub enum UOpArg {
    /// Scalar constant value.
    Scalar(ScalarValue),
    /// Shape for reshape/expand operations.
    Shape(Shape),
    /// Axes for permute/reduce operations.
    Axes(Vec<usize>),
    /// Keep dimensions flag for reduce operations.
    KeepDims(bool),
    /// Buffer identifier.
    BufferId(usize),
    /// Target dtype for cast.
    DType(DType),
    /// Padding configuration: Vec<(before, after)>.
    Padding(Vec<(usize, usize)>),
}

/// Inner data for a UOp node.
struct UOpInner {
    op: Ops,
    dtype: DType,
    shape: Shape,
    src: Vec<UOp>,
    arg: Option<UOpArg>,
}

/// A node in the computation graph (unified operation).
///
/// UOp represents a single operation in the lazy evaluation graph.
/// It is reference-counted and immutable once created.
#[derive(Clone)]
pub struct UOp(Arc<UOpInner>);

impl UOp {
    /// Creates a new UOp node.
    pub fn new(
        op: Ops,
        dtype: DType,
        shape: Shape,
        src: Vec<UOp>,
        arg: Option<UOpArg>,
    ) -> Self {
        UOp(Arc::new(UOpInner {
            op,
            dtype,
            shape,
            src,
            arg,
        }))
    }

    /// Creates a constant UOp.
    pub fn constant(value: ScalarValue, shape: Shape) -> Self {
        UOp::new(
            Ops::Const,
            value.dtype(),
            shape,
            vec![],
            Some(UOpArg::Scalar(value)),
        )
    }

    /// Creates a load UOp from a buffer.
    pub fn load(buffer_id: usize, dtype: DType, shape: Shape) -> Self {
        UOp::new(
            Ops::Load,
            dtype,
            shape,
            vec![],
            Some(UOpArg::BufferId(buffer_id)),
        )
    }

    // Accessors

    /// Returns the operation type.
    pub fn op(&self) -> Ops {
        self.0.op
    }

    /// Returns the data type.
    pub fn dtype(&self) -> DType {
        self.0.dtype
    }

    /// Returns the shape.
    pub fn shape(&self) -> &Shape {
        &self.0.shape
    }

    /// Returns the source operands.
    pub fn src(&self) -> &[UOp] {
        &self.0.src
    }

    /// Returns the argument if present.
    pub fn arg(&self) -> Option<&UOpArg> {
        self.0.arg.as_ref()
    }

    /// Returns the number of elements.
    pub fn numel(&self) -> usize {
        self.0.shape.numel()
    }

    // Unary operations

    /// Negation.
    pub fn neg(&self) -> UOp {
        UOp::new(Ops::Neg, self.dtype(), self.shape().clone(), vec![self.clone()], None)
    }

    /// Exponential.
    pub fn exp(&self) -> UOp {
        UOp::new(Ops::Exp, self.dtype(), self.shape().clone(), vec![self.clone()], None)
    }

    /// Natural logarithm.
    pub fn log(&self) -> UOp {
        UOp::new(Ops::Log, self.dtype(), self.shape().clone(), vec![self.clone()], None)
    }

    /// Square root.
    pub fn sqrt(&self) -> UOp {
        UOp::new(Ops::Sqrt, self.dtype(), self.shape().clone(), vec![self.clone()], None)
    }

    /// Reciprocal (1/x).
    pub fn recip(&self) -> UOp {
        UOp::new(Ops::Recip, self.dtype(), self.shape().clone(), vec![self.clone()], None)
    }

    /// Sine.
    pub fn sin(&self) -> UOp {
        UOp::new(Ops::Sin, self.dtype(), self.shape().clone(), vec![self.clone()], None)
    }

    /// Cosine.
    pub fn cos(&self) -> UOp {
        UOp::new(Ops::Cos, self.dtype(), self.shape().clone(), vec![self.clone()], None)
    }

    // Binary operations

    fn binary_op(&self, other: &UOp, op: Ops) -> UOp {
        let shape = self.shape().broadcast(other.shape())
            .expect("Shapes must be broadcastable");
        // Use promoted dtype (for simplicity, use self's dtype if both are same category)
        let dtype = if self.dtype().is_float() || other.dtype().is_float() {
            if self.dtype() == DType::Float64 || other.dtype() == DType::Float64 {
                DType::Float64
            } else {
                DType::Float32
            }
        } else {
            self.dtype()
        };
        UOp::new(op, dtype, shape, vec![self.clone(), other.clone()], None)
    }

    /// Addition.
    pub fn add(&self, other: &UOp) -> UOp {
        self.binary_op(other, Ops::Add)
    }

    /// Subtraction.
    pub fn sub(&self, other: &UOp) -> UOp {
        self.binary_op(other, Ops::Sub)
    }

    /// Multiplication.
    pub fn mul(&self, other: &UOp) -> UOp {
        self.binary_op(other, Ops::Mul)
    }

    /// Division.
    pub fn div(&self, other: &UOp) -> UOp {
        self.binary_op(other, Ops::Div)
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &UOp) -> UOp {
        self.binary_op(other, Ops::Max)
    }

    /// Less than comparison.
    pub fn lt(&self, other: &UOp) -> UOp {
        let shape = self.shape().broadcast(other.shape())
            .expect("Shapes must be broadcastable");
        UOp::new(Ops::CmpLt, DType::Bool, shape, vec![self.clone(), other.clone()], None)
    }

    /// Equality comparison.
    pub fn eq(&self, other: &UOp) -> UOp {
        let shape = self.shape().broadcast(other.shape())
            .expect("Shapes must be broadcastable");
        UOp::new(Ops::CmpEq, DType::Bool, shape, vec![self.clone(), other.clone()], None)
    }

    // Ternary operations

    /// Where operation: select from self or other based on condition.
    pub fn where_op(cond: &UOp, x: &UOp, y: &UOp) -> UOp {
        let shape = cond.shape().broadcast(x.shape())
            .and_then(|s| s.broadcast(y.shape()))
            .expect("Shapes must be broadcastable");
        let dtype = x.dtype();
        UOp::new(Ops::Where, dtype, shape, vec![cond.clone(), x.clone(), y.clone()], None)
    }

    // Reduce operations

    /// Sum reduction along specified axes.
    pub fn sum(&self, axes: Option<Vec<usize>>, keepdims: bool) -> UOp {
        let axes = axes.unwrap_or_else(|| (0..self.shape().rank()).collect());
        let new_shape = self.reduce_shape(&axes, keepdims);
        UOp::new(
            Ops::Sum,
            self.dtype(),
            new_shape,
            vec![self.clone()],
            Some(UOpArg::Axes(axes)),
        )
    }

    /// Max reduction along specified axes.
    pub fn reduce_max(&self, axes: Option<Vec<usize>>, keepdims: bool) -> UOp {
        let axes = axes.unwrap_or_else(|| (0..self.shape().rank()).collect());
        let new_shape = self.reduce_shape(&axes, keepdims);
        UOp::new(
            Ops::ReduceMax,
            self.dtype(),
            new_shape,
            vec![self.clone()],
            Some(UOpArg::Axes(axes)),
        )
    }

    fn reduce_shape(&self, axes: &[usize], keepdims: bool) -> Shape {
        let mut new_dims = Vec::new();
        for (i, &d) in self.shape().dims().iter().enumerate() {
            if axes.contains(&i) {
                if keepdims {
                    new_dims.push(1);
                }
            } else {
                new_dims.push(d);
            }
        }
        if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        }
    }

    // Movement operations

    /// Reshape to a new shape.
    pub fn reshape(&self, new_shape: Shape) -> UOp {
        assert_eq!(
            self.shape().numel(),
            new_shape.numel(),
            "Reshape must preserve total elements"
        );
        UOp::new(
            Ops::Reshape,
            self.dtype(),
            new_shape.clone(),
            vec![self.clone()],
            Some(UOpArg::Shape(new_shape)),
        )
    }

    /// Expand (broadcast) to a new shape.
    pub fn expand(&self, new_shape: Shape) -> UOp {
        UOp::new(
            Ops::Expand,
            self.dtype(),
            new_shape.clone(),
            vec![self.clone()],
            Some(UOpArg::Shape(new_shape)),
        )
    }

    /// Permute dimensions.
    pub fn permute(&self, axes: Vec<usize>) -> UOp {
        let new_dims: Vec<_> = axes.iter().map(|&i| self.shape().dim(i)).collect();
        let new_shape = Shape::new(new_dims);
        UOp::new(
            Ops::Permute,
            self.dtype(),
            new_shape,
            vec![self.clone()],
            Some(UOpArg::Axes(axes)),
        )
    }

    /// Cast to a different dtype.
    pub fn cast(&self, dtype: DType) -> UOp {
        if self.dtype() == dtype {
            return self.clone();
        }
        UOp::new(
            Ops::Cast,
            dtype,
            self.shape().clone(),
            vec![self.clone()],
            Some(UOpArg::DType(dtype)),
        )
    }

    /// Returns true if this UOp points to the same node as other.
    pub fn ptr_eq(&self, other: &UOp) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    /// Returns a unique ID for this UOp node (based on Arc pointer).
    pub fn ptr_id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
}

impl std::fmt::Debug for UOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "UOp({:?}, {:?}, {:?}, srcs={})",
            self.op(),
            self.dtype(),
            self.shape(),
            self.src().len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uop_constant() {
        let c = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3]));
        assert_eq!(c.op(), Ops::Const);
        assert_eq!(c.dtype(), DType::Float32);
        assert_eq!(c.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_uop_binary() {
        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3]));
        let b = UOp::constant(ScalarValue::Float32(2.0), Shape::new([2, 3]));
        let c = a.add(&b);
        assert_eq!(c.op(), Ops::Add);
        assert_eq!(c.src().len(), 2);
    }

    #[test]
    fn test_uop_reduce() {
        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3, 4]));
        let s = a.sum(Some(vec![1]), false);
        assert_eq!(s.shape().dims(), &[2, 4]);

        let s2 = a.sum(Some(vec![1]), true);
        assert_eq!(s2.shape().dims(), &[2, 1, 4]);

        let s3 = a.sum(None, false);
        assert!(s3.shape().is_scalar());
    }

    #[test]
    fn test_uop_reshape() {
        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3]));
        let b = a.reshape(Shape::new([6]));
        assert_eq!(b.shape().dims(), &[6]);
    }

    #[test]
    fn test_uop_permute() {
        let a = UOp::constant(ScalarValue::Float32(1.0), Shape::new([2, 3, 4]));
        let b = a.permute(vec![2, 0, 1]);
        assert_eq!(b.shape().dims(), &[4, 2, 3]);
    }
}

pub mod lowerer;
pub mod shapetracker;
use crate::ast::{AstNode, DType, Op as AstOp};
use std::cell::Cell;
use std::ops::Deref;
use std::rc::Rc;

thread_local! {
    static NEXT_ID: Cell<usize> = const { Cell::new(0) };
}

fn next_id() -> usize {
    NEXT_ID.with(|cell| {
        let id = cell.get();
        cell.set(id + 1);
        id
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    op: TensorOp,
    src: Vec<Tensor>,
    dtype: DType,
    id: usize,
}

#[derive(Debug, Clone)]
pub struct Tensor(Rc<TensorData>);

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op && self.src == other.src
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    Elementwise(AstNode), // Capture(n)がn番目のsrcが入ることを表すプレースホルダとする。これにより、マージされたElementwise演算子の表現が可能になる。
    Reduce(AstOp, Vec<usize>),
    Contiguous,
    Leaf,
}

impl Deref for Tensor {
    type Target = TensorData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Tensor {
    pub fn new(op: TensorOp, src: Vec<Tensor>, dtype: DType) -> Tensor {
        // Graph construction-time type checks
        match &op {
            TensorOp::Elementwise(ast) => {
                debug_assert_eq!(
                    ast.dtype, dtype,
                    "Tensor dtype must match AST dtype for Elementwise op"
                );
            }
            TensorOp::Reduce(..) => {
                debug_assert!(!src.is_empty());
                debug_assert_eq!(
                    src[0].dtype, dtype,
                    "Tensor dtype must match source dtype for Reduce op"
                );
            }
            TensorOp::Contiguous => {
                debug_assert!(!src.is_empty());
                debug_assert_eq!(
                    src[0].dtype, dtype,
                    "Tensor dtype must match source dtype for Contiguous op"
                );
            }
            TensorOp::Leaf => {
                debug_assert!(src.is_empty());
            }
        }
        Tensor(Rc::new(TensorData {
            op,
            src,
            dtype,
            id: next_id(),
        }))
    }
}

macro_rules! impl_tensor_binary_op {
    ($trait:ident, $fname:ident, $op:expr) => {
        impl<T> std::ops::$trait<T> for Tensor
        where
            T: Into<Tensor>,
        {
            type Output = Tensor;
            fn $fname(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                let op_fn = $op;
                let ast_node = op_fn(
                    AstNode::capture(0, self.dtype.clone()),
                    AstNode::capture(1, rhs.dtype.clone()),
                );
                Tensor::new(
                    TensorOp::Elementwise(ast_node.clone()),
                    vec![self.clone(), rhs],
                    ast_node.dtype,
                )
            }
        }
    };
}

impl_tensor_binary_op!(Add, add, |a: AstNode, b: AstNode| a + b);
impl_tensor_binary_op!(Sub, sub, |a: AstNode, b: AstNode| a - b);
impl_tensor_binary_op!(Mul, mul, |a: AstNode, b: AstNode| a * b);
impl_tensor_binary_op!(Div, div, |a: AstNode, b: AstNode| a / b);

macro_rules! impl_tensor_unary_op {
    ($trait:ident, $fname:ident, $op:expr) => {
        impl std::ops::$trait for Tensor {
            type Output = Tensor;
            fn $fname(self) -> Self::Output {
                let op_fn = $op;
                let ast_node = op_fn(AstNode::capture(0, self.dtype.clone()));
                Tensor::new(
                    TensorOp::Elementwise(ast_node.clone()),
                    vec![self.clone()],
                    ast_node.dtype,
                )
            }
        }
    };
}

impl_tensor_unary_op!(Neg, neg, |a: AstNode| -a);

macro_rules! impl_tensor_assign_op {
    ($trait:ident, $fname:ident, $op_trait:ident, $op_fname:ident) => {
        impl<T> std::ops::$trait<T> for Tensor
        where
            T: Into<Tensor>,
        {
            fn $fname(&mut self, rhs: T) {
                // Since Tensor is immutable due to Rc, we can't modify it in-place.
                // Instead, we perform the operation (which creates a new Tensor)
                // and replace `self` with the result.
                // We must clone `self` because the standard op methods consume it.
                *self = std::ops::$op_trait::$op_fname(self.clone(), rhs.into());
            }
        }
    };
}

impl_tensor_assign_op!(AddAssign, add_assign, Add, add);
impl_tensor_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_tensor_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_tensor_assign_op!(DivAssign, div_assign, Div, div);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, Op};

    fn new_tensor(dtype: DType) -> Tensor {
        Tensor::new(TensorOp::Leaf, vec![], dtype)
    }

    #[test]
    fn test_tensor_add() {
        let a = new_tensor(DType::F32);
        let b = new_tensor(DType::F32);
        let c = a + b;

        assert_eq!(c.src.len(), 2);
        assert_eq!(c.dtype, DType::F32);
        if let TensorOp::Elementwise(ast) = &c.op {
            assert_eq!(ast.op, Op::Add);
        } else {
            panic!("Expected Elementwise op");
        }
    }

    #[test]
    fn test_tensor_sub() {
        let a = new_tensor(DType::F32);
        let b = new_tensor(DType::F32);
        let c = a - b;

        assert_eq!(c.src.len(), 2);
        assert_eq!(c.dtype, DType::F32);
        if let TensorOp::Elementwise(ast) = &c.op {
            // Sub is implemented as Add(a, Neg(b))
            assert_eq!(ast.op, Op::Add);
            assert_eq!(ast.src[1].op, Op::Neg);
        } else {
            panic!("Expected Elementwise op");
        }
    }

    #[test]
    fn test_tensor_mul() {
        let a = new_tensor(DType::F32);
        let b = new_tensor(DType::F32);
        let c = a * b;

        assert_eq!(c.src.len(), 2);
        assert_eq!(c.dtype, DType::F32);
        if let TensorOp::Elementwise(ast) = &c.op {
            assert_eq!(ast.op, Op::Mul);
        } else {
            panic!("Expected Elementwise op");
        }
    }

    #[test]
    fn test_tensor_div() {
        let a = new_tensor(DType::F32);
        let b = new_tensor(DType::F32);
        let c = a / b;

        assert_eq!(c.src.len(), 2);
        assert_eq!(c.dtype, DType::F32);
        if let TensorOp::Elementwise(ast) = &c.op {
            // Div is implemented as Mul(a, Recip(b))
            assert_eq!(ast.op, Op::Mul);
            assert_eq!(ast.src[1].op, Op::Recip);
        } else {
            panic!("Expected Elementwise op");
        }
    }

    #[test]
    fn test_tensor_neg() {
        let a = new_tensor(DType::F32);
        let b = -a;

        assert_eq!(b.src.len(), 1);
        assert_eq!(b.dtype, DType::F32);
        if let TensorOp::Elementwise(ast) = &b.op {
            assert_eq!(ast.op, Op::Neg);
        } else {
            panic!("Expected Elementwise op");
        }
    }

    #[test]
    fn test_tensor_implicit_cast() {
        let a = new_tensor(DType::I32);
        let b = new_tensor(DType::F32);
        let c = a + b;

        assert_eq!(c.src.len(), 2);
        assert_eq!(c.dtype, DType::F32);
        if let TensorOp::Elementwise(ast) = &c.op {
            assert_eq!(ast.op, Op::Add);
            assert_eq!(ast.src[0].op, Op::Cast(DType::F32));
        } else {
            panic!("Expected Elementwise op");
        }
    }

    #[test]
    fn test_tensor_add_assign() {
        let mut a = new_tensor(DType::F32);
        let b = new_tensor(DType::F32);
        let c = a.clone() + b.clone();
        a += b;
        assert_eq!(a, c);
    }

    #[test]
    fn test_tensor_sub_assign() {
        let mut a = new_tensor(DType::F32);
        let b = new_tensor(DType::F32);
        let c = a.clone() - b.clone();
        a -= b;
        assert_eq!(a, c);
    }
}

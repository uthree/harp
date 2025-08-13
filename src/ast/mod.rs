use std::env::Args;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    F32,   // float
    Usize, // size_t

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array
}

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    F32(f32),
    Usize(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstOp {
    Const(Const),
    Add,
    Mul,
    Max,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AstNode {
    op: AstOp,
    args: Vec<AstNode>,
    dtype: DType,
}

impl AstNode {
    fn _new(op: AstOp, args: Vec<AstNode>, dtype: DType) -> Self {
        Self { op, args, dtype }
    }
}

macro_rules! impl_from_num_for_astnode {
    ($(($t:ty, $v: ident)),*) => {
        $(
            impl From<$t> for AstNode {
                fn from(n: $t) -> Self {
                    AstNode::_new(AstOp::Const(Const::$v(n)), vec![], DType::$v)
                }
            }
        )*
    };
}

impl_from_num_for_astnode!((usize, Usize));

macro_rules! impl_astnode_binary_op {
    ($trait:ident, $fname:ident, $variant:ident) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: T) -> Self::Output {
                let dtype = (&self.dtype).clone();
                AstNode::_new(AstOp::$variant, vec![self, rhs.into()], dtype)
            }
        }
    };
}

impl_astnode_binary_op!(Add, add, Add);
impl_astnode_binary_op!(Mul, mul, Mul);

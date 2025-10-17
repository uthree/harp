mod helper;
mod op;
pub mod pat;
pub use helper::*;
pub use pat::{capture, AstRewriteRule, AstRewriter};

#[derive(Debug, Clone)]
pub struct AstNode {
    pub op: AstOp,
    pub dtype: DType,
}

impl AstNode {
    /// 子ノードへの参照をVecで返す
    pub fn children(&self) -> Vec<&AstNode> {
        match &self.op {
            AstOp::Add(lhs, rhs)
            | AstOp::Mul(lhs, rhs)
            | AstOp::Max(lhs, rhs)
            | AstOp::Idiv(lhs, rhs)
            | AstOp::Rem(lhs, rhs) => vec![lhs.as_ref(), rhs.as_ref()],
            AstOp::Neg(operand)
            | AstOp::Recip(operand)
            | AstOp::Sqrt(operand)
            | AstOp::Sin(operand)
            | AstOp::Log2(operand)
            | AstOp::Exp2(operand)
            | AstOp::Cast(operand, _) => vec![operand.as_ref()],
            AstOp::Load {
                target,
                offset,
                size,
            } => vec![target.as_ref(), offset.as_ref(), size.as_ref()],
            AstOp::Store {
                target,
                offset,
                value,
            } => vec![target.as_ref(), offset.as_ref(), value.as_ref()],
            AstOp::Const(_) | AstOp::Var(_) | AstOp::Capture(_) => vec![],
        }
    }

    /// 子ノードを置き換えた新しいAstNodeを作成
    pub fn with_children(&self, new_children: Vec<AstNode>) -> AstNode {
        let new_op = match &self.op {
            AstOp::Add(_, _) => {
                assert_eq!(new_children.len(), 2);
                AstOp::Add(
                    Box::new(new_children[0].clone()),
                    Box::new(new_children[1].clone()),
                )
            }
            AstOp::Mul(_, _) => {
                assert_eq!(new_children.len(), 2);
                AstOp::Mul(
                    Box::new(new_children[0].clone()),
                    Box::new(new_children[1].clone()),
                )
            }
            AstOp::Max(_, _) => {
                assert_eq!(new_children.len(), 2);
                AstOp::Max(
                    Box::new(new_children[0].clone()),
                    Box::new(new_children[1].clone()),
                )
            }
            AstOp::Idiv(_, _) => {
                assert_eq!(new_children.len(), 2);
                AstOp::Idiv(
                    Box::new(new_children[0].clone()),
                    Box::new(new_children[1].clone()),
                )
            }
            AstOp::Rem(_, _) => {
                assert_eq!(new_children.len(), 2);
                AstOp::Rem(
                    Box::new(new_children[0].clone()),
                    Box::new(new_children[1].clone()),
                )
            }
            AstOp::Neg(_) => {
                assert_eq!(new_children.len(), 1);
                AstOp::Neg(Box::new(new_children[0].clone()))
            }
            AstOp::Recip(_) => {
                assert_eq!(new_children.len(), 1);
                AstOp::Recip(Box::new(new_children[0].clone()))
            }
            AstOp::Sqrt(_) => {
                assert_eq!(new_children.len(), 1);
                AstOp::Sqrt(Box::new(new_children[0].clone()))
            }
            AstOp::Sin(_) => {
                assert_eq!(new_children.len(), 1);
                AstOp::Sin(Box::new(new_children[0].clone()))
            }
            AstOp::Log2(_) => {
                assert_eq!(new_children.len(), 1);
                AstOp::Log2(Box::new(new_children[0].clone()))
            }
            AstOp::Exp2(_) => {
                assert_eq!(new_children.len(), 1);
                AstOp::Exp2(Box::new(new_children[0].clone()))
            }
            AstOp::Cast(_, dtype) => {
                assert_eq!(new_children.len(), 1);
                AstOp::Cast(Box::new(new_children[0].clone()), dtype.clone())
            }
            AstOp::Load { .. } => {
                assert_eq!(new_children.len(), 3);
                AstOp::Load {
                    target: Box::new(new_children[0].clone()),
                    offset: Box::new(new_children[1].clone()),
                    size: Box::new(new_children[2].clone()),
                }
            }
            AstOp::Store { .. } => {
                assert_eq!(new_children.len(), 3);
                AstOp::Store {
                    target: Box::new(new_children[0].clone()),
                    offset: Box::new(new_children[1].clone()),
                    value: Box::new(new_children[2].clone()),
                }
            }
            op @ (AstOp::Const(_) | AstOp::Var(_) | AstOp::Capture(_)) => {
                assert_eq!(new_children.len(), 0);
                op.clone()
            }
        };

        AstNode {
            op: new_op,
            dtype: self.dtype.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AstOp {
    Const(ConstValue),
    Var(String),
    Add(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Max(Box<AstNode>, Box<AstNode>),
    Idiv(Box<AstNode>, Box<AstNode>), // integer division
    Rem(Box<AstNode>, Box<AstNode>),  // remainder
    Neg(Box<AstNode>),
    Recip(Box<AstNode>),
    Sqrt(Box<AstNode>),
    Sin(Box<AstNode>),
    Log2(Box<AstNode>),
    Exp2(Box<AstNode>),
    Cast(Box<AstNode>, DType),

    Load {
        // load values to stack from buffer
        target: Box<AstNode>,
        offset: Box<AstNode>,
        size: Box<AstNode>,
    },
    Store {
        // write values to buffer
        target: Box<AstNode>,
        offset: Box<AstNode>,
        value: Box<AstNode>,
    },

    Capture(isize), // for pattern matching
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Unknown, // unknown type, placeholder
    None,    // void
    Isize,   // signed integer
    Usize,   // unsigned integer
    F32,     // 32-bit float
    Bool,    // bool,

    Ptr(Box<DType>),        // pointer of some type
    Vec(Box<DType>, usize), // fixed-size vector for simd operation
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Isize(isize),
    Usize(usize),
    F32(f32),
    Bool(bool),
}

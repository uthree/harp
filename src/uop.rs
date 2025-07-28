use std::{ops::Deref, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq)]
// kind of operation
enum Op {
    Add,
    Neg,
    Mul,
    Recip,
    Rem,
    Max,
    Sin,
    Log2,
    Exp2,
    Sqrt,
}

// UOp data
#[derive(Debug, Clone, PartialEq)]
pub struct UOpData {
    op: Op,
    src: Vec<UOp>,
}

// UOp reference wrapper
#[derive(Debug, Clone, PartialEq)]
pub struct UOp(Rc<UOpData>);

impl Deref for UOp {
    type Target = UOpData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

macro_rules! impl_unary_uop_constructor {
    ($op: ident, $func: ident) => {
        impl UOp {
            pub fn $func(a: UOp) -> UOp {
                UOp(Rc::new(UOpData {
                    op: Op::$op,
                    src: vec![a],
                }))
            }
        }
    };
}

impl_unary_uop_constructor!(Neg, neg);
impl_unary_uop_constructor!(Recip, recip);
impl_unary_uop_constructor!(Sin, sin);
impl_unary_uop_constructor!(Sqrt, sqrt);
impl_unary_uop_constructor!(Log2, log2);
impl_unary_uop_constructor!(Exp2, exp2);

macro_rules! impl_binary_uop_constructor {
    ($op: ident, $func: ident) => {
        impl UOp {
            pub fn $func(a: UOp, b: UOp) -> UOp {
                UOp(Rc::new(UOpData {
                    op: Op::$op,
                    src: vec![a, b],
                }))
            }
        }
    };
}

impl_binary_uop_constructor!(Add, add);
impl_binary_uop_constructor!(Mul, mul);
impl_binary_uop_constructor!(Max, max);
impl_binary_uop_constructor!(Rem, rem);

use std::rc::Rc;

// datatypes
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum DType {}

// operator types
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Ops {
    Add,
    Mul,
}

// internal data of UOp
#[derive(Clone, PartialEq, Debug)]
struct UOp_ {
    op: DType,
    src: Vec<UOp>,
}

// micro operator
#[derive(Clone, PartialEq, Debug)]
pub struct UOp(Rc<UOp_>);

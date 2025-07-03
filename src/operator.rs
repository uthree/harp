use std::fmt::Debug;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Operator {
    Input,
    Add,
    Mul,
    Recip,
}

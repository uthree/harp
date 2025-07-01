use crate::shape::symbolic::Expr;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeTracker {
    dims: Vec<Expr>,
    indexes: Vec<Expr>,
}

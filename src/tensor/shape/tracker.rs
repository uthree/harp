use crate::tensor::shape::expr::Expr;

#[derive(Debug, Clone)]
pub struct ShapeTracker {
    pub shape: Vec<Expr>,   // logical shape for each axis.
    pub strides: Vec<Expr>, // actual memory offsets for each axis.
}

use crate::graph::shape::expr::Expr;

pub enum View {
    Linear { shape: Vec<Expr> },
}

use std::fmt::Debug;

#[derive(Clone, Debug)]
pub enum Expr {
    // literal
    Var(String),
    Const(i32),
}

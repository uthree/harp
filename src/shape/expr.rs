use std::fmt::Debug;

#[derive(Clone, Debug)]
pub enum Expr {
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Var(String),
    Const(i32),
}

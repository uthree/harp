#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Const(usize),
    Var(String), // variable
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    Neg(Box<Self>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Var(String),
    Idx(usize),
    Int(isize),
    Neg(Box<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Max(Box<Expr>, Box<Expr>),
    Rem(Box<Expr>, Box<Expr>),
}

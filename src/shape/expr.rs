use std::fmt::Debug;

#[derive(Clone, Debug)]
pub struct Expr(Vec<Term>);

#[derive(Clone, Debug)]
pub struct Term {
    prod: Vec<Atom>,
    div: Vec<Atom>,
}

#[derive(Clone, Debug)]
pub enum Atom {
    Var(String),
    Const(isize),
}

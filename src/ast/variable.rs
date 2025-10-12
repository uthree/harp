use super::{AstNode, DType};

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub dtype: DType,
    pub constant: bool,
    pub size_expr: Option<Box<AstNode>>, // For dynamic arrays, the size expression
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scope {
    pub declarations: Vec<VariableDecl>,
}

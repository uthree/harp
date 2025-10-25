#[derive(Debug, Clone)]
pub enum AstNode {
    // arithmetics
    Add(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Max(Box<AstNode>, Box<AstNode>),
    Rem(Box<AstNode>, Box<AstNode>),
    Ma(Box<AstNode>, Box<AstNode>),
}

#[derive(Debug, Clone)]
pub struct AstNode {}

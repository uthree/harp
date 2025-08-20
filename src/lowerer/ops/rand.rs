use crate::{ast::AstNode, lowerer::Lowerer};

pub fn lower_rand(_lowerer: &mut Lowerer) -> AstNode {
    AstNode::rand()
}

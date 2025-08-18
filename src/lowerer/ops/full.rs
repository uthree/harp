use crate::{ast::AstNode, graph::GraphOp, lowerer::Lowerer};

pub fn lower_full(_lowerer: &mut Lowerer, op: &GraphOp) -> AstNode {
    if let GraphOp::Full(c) = op {
        AstNode::from(c.clone())
    } else {
        panic!("Expected Full operation");
    }
}

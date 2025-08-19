use crate::{
    ast::{AstNode, AstOp, DType},
    lowerer::Lowerer,
};

pub fn lower_rand(_lowerer: &mut Lowerer) -> AstNode {
    AstNode::_new(AstOp::Rand, vec![], DType::F32)
}

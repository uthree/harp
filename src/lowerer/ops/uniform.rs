use crate::{
    ast::{AstNode, AstOp, DType},
    lowerer::Lowerer,
};

pub fn lower_uniform(_lowerer: &mut Lowerer) -> AstNode {
    AstNode::_new(AstOp::Uniform, vec![], DType::F32)
}

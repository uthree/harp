use crate::{ast::AstNode, tensor::Graph};

#[derive(Debug, Clone)]
pub struct Lowerer {
    var_counter: usize,
} // Converts Tensor to AstNode

impl Lowerer {
    fn new() -> Self {
        Lowerer { var_counter: 0 }
    }

    //fn lower(&mut self, graph: &Graph) -> AstNode {}
}

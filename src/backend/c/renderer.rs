use crate::{
    ast::{AstNode, DType, Function},
    backend::Renderer,
};
use std::fmt::Write;

#[derive(Debug, Default)]
pub struct CRenderer {
    indent_level: usize,
}

impl Renderer for CRenderer {
    type CodeRepr = String;
    type Option = ();

    fn with_option(&mut self, option: Self::Option) {}

    fn new() -> Self {
        CRenderer::default()
    }

    fn render(&mut self, program: crate::ast::Program) -> Self::CodeRepr {
        todo!()
    }
}

impl CRenderer {
    fn render_function(&mut self, function: Function) {}

    fn render_node(&mut self, node: AstNode) -> String {
        let mut buffer = String::new();
        match node {
            AstNode::Add(lhs, rhs) => match *rhs {
                AstNode::Neg(negv) => write!(
                    buffer,
                    "( {} - {} )",
                    self.render_node(*lhs),
                    self.render_node(*negv)
                )
                .unwrap(),
                _ => write!(
                    buffer,
                    "({} + {})",
                    self.render_node(*lhs),
                    self.render_node(*rhs)
                )
                .unwrap(),
            },
            AstNode::Mul(lhs, rhs) => match *rhs {
                AstNode::Recip(negv) => write!(
                    buffer,
                    "( {} / {} )",
                    self.render_node(*lhs),
                    self.render_node(*negv)
                )
                .unwrap(),
                _ => write!(
                    buffer,
                    "({} * {})",
                    self.render_node(*lhs),
                    self.render_node(*rhs)
                )
                .unwrap(),
            },
            AstNode::Neg(v) => write!(buffer, "-{}", self.render_node(*v)).unwrap(),
            _ => todo!(),
        }
        buffer
    }

    fn render_dtype(&mut self, dtype: DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "ssize_t".to_string(),
            DType::Usize => "size_t".to_string(),
            DType::Void => "void".to_string(),
            DType::Ptr(t) => format!("{}*", self.render_dtype(*t)).to_string(),
            DType::Vec(t, size) => format!("{}[{}]", self.render_dtype(*t), size),
        }
    }
}

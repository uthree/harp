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
            AstNode::Const(c) => write!(buffer, "{}", self.render_const(c)).unwrap(),
            AstNode::Var(s) => write!(buffer, "{}", s).unwrap(),
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
                AstNode::Recip(recipv) => write!(
                    buffer,
                    "( {} / {} )",
                    self.render_node(*lhs),
                    self.render_node(*recipv)
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
            AstNode::Rem(lhs, rhs) => write!(
                buffer,
                "({} % {})",
                self.render_node(*lhs),
                self.render_node(*rhs)
            )
            .unwrap(),
            AstNode::Neg(v) => write!(buffer, "-{}", self.render_node(*v)).unwrap(),
            AstNode::Recip(v) => write!(buffer, "(1 / {})", self.render_node(*v)).unwrap(),
            AstNode::Block(insts) => {
                self.indent_level += 1;
                write!(buffer, "{{").unwrap();
                for inst in insts.iter() {
                    buffer.push('\n');
                    self.render_indent(&mut buffer);
                    self.render_node(inst.clone());
                    buffer.push(';');
                }
                buffer.push('\n');
                self.render_indent(&mut buffer);
                write!(buffer, "}}").unwrap();
                self.indent_level -= 1;
            }
            _ => todo!(),
        }
        buffer
    }

    fn render_indent(&self, buffer: &mut String) {
        for _ in 0..self.indent_level {
            buffer.push('\t');
        }
    }

    fn render_const(&self, c: crate::ast::ConstLiteral) -> String {
        use crate::ast::ConstLiteral::*;
        match c {
            F32(v) => format!("{}", v),
            Isize(v) => format!("{}", v),
            Usize(v) => format!("{}", v),
        }
    }

    fn render_dtype(&mut self, dtype: DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "ssize_t".to_string(),
            DType::Usize => "size_t".to_string(),
            DType::Void => "void".to_string(),
            DType::Ptr(t) => format!("{}*", self.render_dtype(*t)),
            DType::Vec(t, size) => format!("{}[{}]", self.render_dtype(*t), size),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;
    use rstest::rstest;

    fn var(name: &str) -> AstNode {
        AstNode::Var(name.to_string())
    }

    #[rstest]
    // Add
    #[case(var("a") + var("b"), "(a + b)")]
    #[case(var("a") + (-var("b")), "( a - b )")]
    // Mul
    #[case(var("a") * var("b"), "(a * b)")]
    #[case(var("a") * var("b").recip(), "( a / b )")]
    // Neg
    #[case(-var("a"), "-a")]
    // Complex
    #[case(-(var("a") + var("b")) * var("c"), "(-(a + b) * c)")]
    fn test_render_node(#[case] input: AstNode, #[case] expected: &str) {
        let mut renderer = CRenderer::new();
        assert_eq!(renderer.render_node(input), expected);
    }
}

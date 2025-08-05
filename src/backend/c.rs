//! C language backend for rendering the AST.

use crate::ast::{AstNode, Const, DType, Op};
use crate::backend::Renderer;
use std::fmt::Write;

/// A renderer that converts an `AstNode` into a C source code string.
#[derive(Default)]
pub struct CRenderer {
    buffer: String,
    indent_level: usize,
}

impl CRenderer {
    /// Renders a single `AstNode` recursively.
    fn render_node(&mut self, ast: &AstNode) {
        match &ast.op {
            Op::Block => {
                self.writeln("{");
                self.indent_level += 1;
                for node in &ast.src {
                    self.render_node(node);
                }
                self.indent_level -= 1;
                self.writeln("}");
            }
            Op::Range {
                loop_var,
                max,
                block,
            } => {
                self.write_indent();
                write!(self.buffer, "for (int {loop_var} = 0; {loop_var} < ").unwrap();
                self.render_node(max);
                self.writeln(") {");
                self.indent_level += 1;
                self.render_node(block);
                self.indent_level -= 1;
                self.writeln("}");
            }
            Op::FuncDef { name, args, body } => {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|(name, dtype)| format!("{} {}", self.dtype_to_c(dtype), name))
                    .collect();
                // Assuming void return for now, can be improved later
                self.writeln(&format!("void {}({}) {{", name, args_str.join(", ")));
                self.indent_level += 1;
                self.render_node(body);
                self.indent_level -= 1;
                self.writeln("}");
            }
            Op::Call(name) => {
                self.write_indent();
                write!(self.buffer, "{name}(").unwrap();
                for (i, arg) in ast.src.iter().enumerate() {
                    if i > 0 {
                        write!(self.buffer, ", ").unwrap();
                    }
                    self.render_node(arg);
                }
                self.writeln(");");
            }
            Op::Assign { dst, src } => {
                self.write_indent();
                self.render_node(dst);
                write!(self.buffer, " = ").unwrap();
                self.render_node(src);
                self.writeln(";");
            }
            Op::Store { dst, src } => {
                self.write_indent();
                write!(self.buffer, "*(").unwrap();
                self.render_node(dst);
                write!(self.buffer, ") = ").unwrap();
                self.render_node(src);
                self.writeln(";");
            }
            Op::Load(addr) => {
                write!(self.buffer, "*(").unwrap();
                self.render_node(addr);
                write!(self.buffer, ")").unwrap();
            }
            Op::BufferIndex { buffer, index } => {
                self.render_node(buffer);
                write!(self.buffer, "[").unwrap();
                self.render_node(index);
                write!(self.buffer, "]").unwrap();
            }
            Op::Var(name) => {
                write!(self.buffer, "{name}").unwrap();
            }
            Op::Const(c) => {
                self.render_const(c);
            }
            Op::Cast(dtype) => {
                let type_str = self.dtype_to_c(dtype);
                write!(self.buffer, "({type_str})").unwrap();
                self.render_node(&ast.src[0]);
            }
            Op::Add => self.render_binary_op("+", ast),
            Op::Mul => self.render_binary_op("*", ast),
            Op::Max => self.render_binary_op_func("fmax", ast), // C standard library
            // Add other ops here...
            _ => unimplemented!("Rendering for {:?} is not implemented.", ast.op),
        }
    }

    fn render_binary_op(&mut self, op_str: &str, ast: &AstNode) {
        write!(self.buffer, "(").unwrap();
        self.render_node(&ast.src[0]);
        write!(self.buffer, " {op_str} ").unwrap();
        self.render_node(&ast.src[1]);
        write!(self.buffer, ")").unwrap();
    }

    fn render_binary_op_func(&mut self, func_name: &str, ast: &AstNode) {
        write!(self.buffer, "{func_name}(").unwrap();
        self.render_node(&ast.src[0]);
        write!(self.buffer, ", ").unwrap();
        self.render_node(&ast.src[1]);
        write!(self.buffer, ")").unwrap();
    }

    fn render_const(&mut self, c: &Const) {
        match c {
            Const::F32(v) => write!(self.buffer, "{v}").unwrap(),
            Const::F64(v) => write!(self.buffer, "{v}").unwrap(),
            Const::I32(v) => write!(self.buffer, "{v}").unwrap(),
            Const::I64(v) => write!(self.buffer, "{v}").unwrap(),
            _ => unimplemented!("Const type not supported in C renderer"),
        }
    }

    fn dtype_to_c(&self, dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::F64 => "double".to_string(),
            DType::I32 => "int".to_string(),
            DType::I64 => "long long".to_string(),
            // Assuming pointers are for buffers of a certain type
            DType::Ptr(inner) => format!("{}*", self.dtype_to_c(inner)),
            _ => panic!("DType {dtype:?} not supported in C renderer"),
        }
    }

    fn write_indent(&mut self) {
        write!(self.buffer, "{}", "    ".repeat(self.indent_level)).unwrap();
    }

    fn writeln(&mut self, s: &str) {
        self.write_indent();
        writeln!(self.buffer, "{s}").unwrap();
    }
}

impl Renderer for CRenderer {
    fn new() -> Self {
        CRenderer::default()
    }

    fn render(&mut self, ast: AstNode) -> String {
        self.buffer.clear();
        self.indent_level = 0;
        self.render_node(&ast);
        self.buffer.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    #[test]
    fn test_render_simple_add() {
        let a = AstNode::var("a");
        let b = AstNode::from(1.0f32);
        let add_expr = a + b;
        let mut renderer = CRenderer::new();
        let code = renderer.render(add_expr);
        assert_eq!(code.trim(), "((float)a + 1)");
    }

    #[test]
    fn test_render_for_loop() {
        let loop_var = "i".to_string();
        let max = AstNode::from(10i32);
        let body = AstNode::new(Op::Block, vec![], DType::None);
        let loop_node = AstNode::new(
            Op::Range {
                loop_var,
                max: Box::new(max),
                block: Box::new(body),
            },
            vec![],
            DType::None,
        );

        let mut renderer = CRenderer::new();
        let code = renderer.render(loop_node);
        let expected = "for (int i = 0; i < 10) {\n    {\n    }\n}";
        assert_eq!(
            code.trim().replace(" ", "").replace("\n", ""),
            expected.replace(" ", "").replace("\n", "")
        );
    }

    #[test]
    fn test_render_func_def() {
        let args = vec![
            ("a".to_string(), DType::Ptr(Box::new(DType::F32))),
            ("b".to_string(), DType::I32),
        ];
        let body = AstNode::new(Op::Block, vec![], DType::None);
        let func_def = AstNode::func_def("my_func", args, body);

        let mut renderer = CRenderer::new();
        let code = renderer.render(func_def);
        let expected = "void my_func(float* a, int b) {\n    {\n    }\n}";
        assert_eq!(
            code.trim().replace(" ", "").replace("\n", ""),
            expected.replace(" ", "").replace("\n", "")
        );
    }
}

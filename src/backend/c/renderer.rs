use log::{debug, info, trace};

use crate::ast::{AstNode, AstOp, Const, DType};
use crate::backend::Renderer;
use std::fmt::Write;

#[derive(Default)]
pub struct CRenderer {
    indent_level: usize,
    buffer: String,
}

impl CRenderer {
    fn render_node(&mut self, ast: &AstNode) {
        trace!("Rendering node: {:?}", ast.op);
        match &ast.op {
            AstOp::Program => {
                self.buffer.push_str("#include <math.h>\n");
                self.buffer.push_str("#include <stddef.h>\n");
                self.buffer.push_str("#include <stdint.h>\n");
                for node in &ast.src {
                    self.render_node(node);
                    self.writeln("");
                }
            }
            AstOp::Block => {
                for node in &ast.src {
                    self.render_node(node);
                }
            }
            AstOp::Func { name, args } => {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|(name, dtype)| format!("{} {}", Self::dtype_to_c(dtype), name))
                    .collect();
                self.writeln(&format!("void {}({}) {{ ", name, args_str.join(", ")));
                self.indent_level += 1;
                for node in &ast.src {
                    self.render_node(node);
                }
                self.indent_level -= 1;
                self.writeln("}");
            }
            AstOp::Declare { name, dtype } => {
                self.write_indent();
                write!(self.buffer, "{} {}", Self::dtype_to_c(dtype), name).unwrap();
                if let Some(value) = ast.src.first() {
                    write!(self.buffer, " = ").unwrap();
                    self.render_node(value);
                }
                writeln!(self.buffer, ";").unwrap();
            }
            AstOp::Assign => {
                self.write_indent();
                self.render_node(&ast.src[0]);
                write!(self.buffer, " = ").unwrap();
                self.render_node(&ast.src[1]);
                writeln!(self.buffer, ";").unwrap();
            }
            AstOp::Store => {
                self.write_indent();
                self.render_node(&ast.src[0]);
                write!(self.buffer, " = ").unwrap();
                self.render_node(&ast.src[1]);
                writeln!(self.buffer, ";").unwrap();
            }
            AstOp::Load => {
                write!(self.buffer, "*(").unwrap();
                self.render_node(&ast.src[0]);
                write!(self.buffer, ")").unwrap();
            }
            AstOp::Index => {
                self.render_node(&ast.src[0]);
                write!(self.buffer, "[").unwrap();
                self.render_node(&ast.src[1]);
                write!(self.buffer, "]").unwrap();
            }
            AstOp::Range { counter, step } => {
                self.write_indent();
                write!(self.buffer, "for (size_t {} = 0; {} < ", counter, counter).unwrap();
                self.render_node(&ast.src[0]);
                if *step == 1 {
                    writeln!(self.buffer, "; {}++) {{ ", counter).unwrap();
                } else {
                    writeln!(self.buffer, "; {} += {}) {{ ", counter, step).unwrap();
                }
                self.indent_level += 1;
                for node in ast.src.iter().skip(1) {
                    self.render_node(node);
                }
                self.indent_level -= 1;
                self.writeln("}");
            }
            AstOp::Call(name) => {
                self.write_indent();
                write!(self.buffer, "{}(", name).unwrap();
                for (i, arg) in ast.src.iter().enumerate() {
                    if i > 0 {
                        write!(self.buffer, ", ").unwrap();
                    }
                    self.render_node(arg);
                }
                writeln!(self.buffer, ");").unwrap();
            }
            AstOp::Var(name) => {
                write!(self.buffer, "{}", name).unwrap();
            }
            AstOp::Const(c) => {
                self.render_const(c);
            }
            AstOp::Cast(dtype) => {
                write!(self.buffer, "({})", Self::dtype_to_c(dtype)).unwrap();
                self.render_node(&ast.src[0]);
            }
            AstOp::Add => self.render_binary_op("+", ast),
            AstOp::Sub => self.render_binary_op("-", ast),
            AstOp::Mul => self.render_binary_op("*", ast),
            AstOp::Div => self.render_binary_op("/", ast),
            AstOp::Rem => self.render_binary_op("%", ast),
            AstOp::Max => self.render_binary_op_func("fmax", ast),
            AstOp::Neg => {
                write!(self.buffer, "(-").unwrap();
                self.render_node(&ast.src[0]);
                write!(self.buffer, ")").unwrap();
            }
            AstOp::Sin => self.render_unary_op_func("sin", ast),
            AstOp::Sqrt => self.render_unary_op_func("sqrt", ast),
            AstOp::Log2 => self.render_unary_op_func("log2", ast),
            AstOp::Exp2 => self.render_unary_op_func("exp2", ast),
            _ => unimplemented!("Rendering for `{:?}` is not implemented.", ast.op),
        }
    }

    fn render_unary_op_func(&mut self, func_name: &str, ast: &AstNode) {
        let full_func_name = if ast.dtype == DType::F32 {
            format!("{}f", func_name)
        } else {
            func_name.to_string()
        };
        write!(self.buffer, "{}(", full_func_name).unwrap();
        self.render_node(&ast.src[0]);
        write!(self.buffer, ")").unwrap();
    }

    fn render_binary_op(&mut self, op_str: &str, ast: &AstNode) {
        write!(self.buffer, "(").unwrap();
        self.render_node(&ast.src[0]);
        write!(self.buffer, " {} ", op_str).unwrap();
        self.render_node(&ast.src[1]);
        write!(self.buffer, ")").unwrap();
    }

    fn render_binary_op_func(&mut self, func_name: &str, ast: &AstNode) {
        write!(self.buffer, "{}(", func_name).unwrap();
        self.render_node(&ast.src[0]);
        write!(self.buffer, ", ").unwrap();
        self.render_node(&ast.src[1]);
        write!(self.buffer, ")").unwrap();
    }

    fn render_const(&mut self, c: &Const) {
        match c {
            Const::F32(v) => {
                let v_f32 = f32::from_bits(*v);
                if v_f32.is_infinite() && v_f32.is_sign_negative() {
                    write!(self.buffer, "-INFINITY").unwrap()
                } else {
                    write!(self.buffer, "{:.7}", v_f32).unwrap()
                }
            }
            Const::Isize(v) => write!(self.buffer, "{}", v).unwrap(),
            Const::Usize(v) => write!(self.buffer, "{}", v).unwrap(),
        }
    }

    fn dtype_to_c(dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Isize => "intptr_t".to_string(),
            DType::Usize => "size_t".to_string(),
            DType::Ptr(inner) => format!("{}*", Self::dtype_to_c(inner)),
            DType::Any => "void".to_string(),
            _ => panic!("DType {:?} not supported in C renderer", dtype),
        }
    }

    fn write_indent(&mut self) {
        write!(self.buffer, "{}", "\t".repeat(self.indent_level)).unwrap();
    }

    fn writeln(&mut self, s: &str) {
        self.write_indent();
        writeln!(self.buffer, "{}", s).unwrap();
    }
}

impl Renderer for CRenderer {
    type CodeRepr = String;
    type Option = ();

    fn new() -> Self {
        Self::default()
    }

    fn with_option(&mut self, _option: Self::Option) {}

    fn render(&mut self, ast: crate::ast::AstNode) -> String {
        info!("Start rendering C code.");
        self.render_node(&ast);
        let code = self.buffer.clone();
        debug!("\n--- Rendered C code ---\n{code}\n-----------------------");
        info!("Finished rendering C code.");
        code
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    #[test]
    fn test_simple_add() {
        let _ = env_logger::builder().is_test(true).try_init();
        let ast = AstNode::var("a", DType::Any) + AstNode::var("b", DType::Any);
        let mut renderer = CRenderer::new();
        let code = renderer.render(ast);
        assert_eq!(code, "(a + b)");
    }
}

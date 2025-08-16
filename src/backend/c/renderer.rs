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
        match &ast.op {
            AstOp::Program => {
                // Add standard headers
                self.buffer.push_str("#include <stdlib.h>\n"); // For rand() and RAND_MAX
                self.buffer.push_str("#include <stddef.h>\n");
                self.buffer.push_str("#include <stdint.h>\n");
                self.buffer.push_str("#include <math.h>\n"); // For fmax, etc.

                for node in &ast.src {
                    self.render_node(node);
                    self.writeln(""); // Add a newline between functions
                }
            }
            AstOp::Range { counter, step } => {
                self.write_indent();
                write!(self.buffer, "for (size_t {counter} = 0; {counter} < ").unwrap();
                self.render_node(&ast.src[0]); // max
                if *step == 1 {
                    writeln!(self.buffer, "; {counter}++) {{").unwrap();
                } else {
                    writeln!(self.buffer, "; {counter} += {step}) {{").unwrap();
                }
                self.indent_level += 1;

                for node in ast.src.iter().skip(1) {
                    self.render_node(node);
                }

                self.indent_level -= 1;
                self.writeln("} ");
            }
            AstOp::Func { name, args } => {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|(name, dtype)| format!("{} {}", Self::dtype_to_c(dtype), name))
                    .collect();
                let args_str = args_str.join(", ");
                self.writeln(&format!("void {name}({args_str}) {{"));
                self.indent_level += 1;
                for node in &ast.src {
                    self.render_node(node);
                }
                self.indent_level -= 1;
                self.writeln("}");
            }
            AstOp::Call(name) => {
                if name == "rand" {
                    write!(self.buffer, "rand()").unwrap();
                    return;
                }
                self.write_indent();
                write!(self.buffer, "{name}(").unwrap();
                for (i, arg) in ast.src.iter().enumerate() {
                    if i > 0 {
                        write!(self.buffer, ", ").unwrap();
                    }
                    self.render_node(arg);
                }
                writeln!(self.buffer, ");").unwrap();
            }
            AstOp::Var(name) => {
                write!(self.buffer, "{name}").unwrap();
            }
            AstOp::Const(c) => {
                self.render_const(c);
            }
            AstOp::Cast(dtype) => {
                let rendered_child = Self::dtype_to_c(dtype);
                write!(self.buffer, "({rendered_child})").unwrap();
                self.render_node(&ast.src[0]);
            }
            AstOp::Add => self.render_binary_op("+", ast),
            AstOp::Sub => self.render_binary_op("-", ast),
            AstOp::Mul => self.render_binary_op("*", ast),
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
            _ => unimplemented!("Rendering for `{ast:?}` is not implemented."),
        }
    }

    fn render_unary_op_func(&mut self, func_name: &str, ast: &AstNode) {
        let full_func_name = if ast.dtype == DType::F32 {
            format!("{func_name}f")
        } else {
            func_name.to_string()
        };
        write!(self.buffer, "{full_func_name}(").unwrap();
        self.render_node(&ast.src[0]);
        write!(self.buffer, ")").unwrap();
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
            Const::F32(v) if v.is_infinite() && v.is_sign_negative() => {
                write!(self.buffer, "-INFINITY").unwrap()
            }
            Const::F32(v) => write!(self.buffer, "{v:.7}").unwrap(),
            Const::Isize(v) => write!(self.buffer, "{v}").unwrap(),
            Const::Usize(v) => write!(self.buffer, "{v}").unwrap(),
        }
    }

    fn dtype_to_c(dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::Ptr(inner) => format!("{}*", Self::dtype_to_c(inner)),
            _ => panic!("DType {dtype:?} not supported in C renderer"),
        }
    }

    fn write_indent(&mut self) {
        write!(self.buffer, "{}", "\t".repeat(self.indent_level)).unwrap();
    }

    fn writeln(&mut self, s: &str) {
        self.write_indent();
        writeln!(self.buffer, "{s}").unwrap();
    }
}

impl Renderer for CRenderer {
    fn new() -> Self {
        Self::default()
    }

    fn render(&mut self, ast: crate::ast::AstNode) -> String {
        self.render_node(&ast);
        let code = self.buffer.clone();
        code
    }
}

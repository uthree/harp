use crate::{
    ast::{AstNode, AstOp, Const, DType},
    backend::Renderer,
};
use log::debug;
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
            AstOp::Program => {
                for node in &ast.src {
                    self.render_node(node);
                    self.writeln(""); // Add a newline between functions
                }
            }
            AstOp::Block => {
                self.writeln("{");
                self.indent_level += 1;
                for node in &ast.src {
                    self.render_node(node);
                }
                self.indent_level -= 1;
                self.writeln("}");
            }
            AstOp::Range { loop_var, step } => {
                self.write_indent();
                write!(self.buffer, "for (size_t {loop_var} = 0; {loop_var} < ").unwrap();
                self.render_node(&ast.src[0]); // max
                write!(self.buffer, "; {loop_var} += {step}) {{ ").unwrap();
                self.indent_level += 1;

                // The rest of src is the loop body.
                // If the loop body is a single block, unwrap it to avoid double braces.
                if ast.src.len() == 2 && matches!(ast.src[1].op, AstOp::Block) {
                    for node in &ast.src[1].src {
                        self.render_node(node);
                    }
                } else {
                    for node in ast.src.iter().skip(1) {
                        self.render_node(node);
                    }
                }
                self.indent_level -= 1;
                self.writeln("}");
            }
            AstOp::Func { name, args } => {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|(name, dtype)| format!("{} {}", Self::dtype_to_c(dtype), name))
                    .collect();
                let args_str = args_str.join(", ");
                self.writeln(&format!("void {name}({args_str}) {{ "));
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
            AstOp::Assign => {
                self.write_indent();
                self.render_node(&ast.src[0]); // dst
                write!(self.buffer, " = ").unwrap();
                self.render_node(&ast.src[1]); // src
                writeln!(self.buffer, ";").unwrap();
            }
            AstOp::Declare { name, dtype } => {
                self.write_indent();
                write!(self.buffer, "{} {} = ", Self::dtype_to_c(dtype), name).unwrap();
                self.render_node(&ast.src[0]); // value
                writeln!(self.buffer, ";").unwrap();
            }
            AstOp::Store => {
                self.write_indent();
                self.render_node(&ast.src[0]); // dst
                write!(self.buffer, " = ").unwrap();
                self.render_node(&ast.src[1]); // src
                writeln!(self.buffer, ";").unwrap();
            }
            AstOp::Deref => {
                // The address itself is what we want to render, e.g., buffer[index]
                self.render_node(&ast.src[0]); // addr
            }
            AstOp::BufferIndex => {
                write!(self.buffer, "(").unwrap();
                self.render_node(&ast.src[0]); // buffer
                write!(self.buffer, ")").unwrap();
                write!(self.buffer, "[").unwrap();
                self.render_node(&ast.src[1]); // index
                write!(self.buffer, "]").unwrap();
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
            AstOp::Rem => {
                if ast.dtype == DType::F32 {
                    self.render_binary_op_func("fmodf", ast)
                } else if ast.dtype == DType::F64 {
                    self.render_binary_op_func("fmod", ast)
                } else {
                    self.render_binary_op("%", ast)
                }
            }
            AstOp::Max => self.render_binary_op_func("fmax", ast),
            AstOp::LessThan => self.render_binary_op("<", ast),
            AstOp::Neg => {
                write!(self.buffer, "(-").unwrap();
                self.render_node(&ast.src[0]);
                write!(self.buffer, ")").unwrap();
            }
            AstOp::Recip => {
                write!(self.buffer, "(1.0f / ").unwrap();
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
            Const::F32(v) => write!(self.buffer, "{v:.7}f").unwrap(),
            Const::F64(v) if v.is_infinite() && v.is_sign_negative() => {
                write!(self.buffer, "-INFINITY").unwrap()
            }
            Const::F64(v) => write!(self.buffer, "{v}").unwrap(),
            Const::I8(v) => write!(self.buffer, "(int8_t){v}").unwrap(),
            Const::I16(v) => write!(self.buffer, "(int16_t){v}").unwrap(),
            Const::I32(v) => write!(self.buffer, "{v}").unwrap(),
            Const::I64(v) => write!(self.buffer, "{v}ll").unwrap(),
            Const::U8(v) => write!(self.buffer, "(uint8_t){v}u").unwrap(),
            Const::U16(v) => write!(self.buffer, "(uint16_t){v}u").unwrap(),
            Const::U32(v) => write!(self.buffer, "{v}u").unwrap(),
            Const::U64(v) => write!(self.buffer, "{v}ull").unwrap(),
            Const::USize(v) => write!(self.buffer, "(size_t){v}ull").unwrap(),
        }
    }

    fn dtype_to_c(dtype: &DType) -> String {
        match dtype {
            DType::F32 => "float".to_string(),
            DType::F64 => "double".to_string(),
            DType::I8 => "int8_t".to_string(),
            DType::I16 => "int16_t".to_string(),
            DType::I32 => "int32_t".to_string(),
            DType::I64 => "int64_t".to_string(),
            DType::U8 => "uint8_t".to_string(),
            DType::U16 => "uint16_t".to_string(),
            DType::U32 => "uint32_t".to_string(),
            DType::U64 => "uint64_t".to_string(),
            DType::USize => "size_t".to_string(),
            DType::Void => "void".to_string(),
            DType::Ptr(inner) => format!("{}*", Self::dtype_to_c(inner)),
            DType::Array(inner, ..) => format!("{}*", Self::dtype_to_c(inner)),
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
        CRenderer::default()
    }
    fn render(&mut self, ast: AstNode) -> String {
        self.buffer.clear();
        self.indent_level = 0;
        // Add standard headers
        self.buffer.push_str("#include <stdlib.h>\n"); // For rand() and RAND_MAX
        self.buffer.push_str("#include <stddef.h>\n");
        self.buffer.push_str("#include <stdint.h>\n");
        self.buffer.push_str("#include <math.h>\n\n"); // For fmax, etc.

        self.render_node(&ast);
        let code = self.buffer.clone();
        debug!("\n--- Rendered C code ---\n{code}\n-----------------------");
        code
    }
}

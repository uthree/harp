//! C language backend for rendering the AST.

use crate::ast::{AstNode, Const, DType, Op};
use crate::backend::{Buffer, Compiler, Kernel, Renderer};
use libloading::{Library, Symbol};
use log::debug;
use std::ffi::c_void;
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
                self.writeln("{ ");
                self.indent_level += 1;
                for node in &ast.src {
                    self.render_node(node);
                }
                self.indent_level -= 1;
                self.writeln("} ");
            }
            Op::Range {
                loop_var,
                max,
                block,
            } => {
                self.write_indent();
                write!(self.buffer, "for (int {loop_var} = 0; {loop_var} < ").unwrap();
                self.render_node(max);
                writeln!(self.buffer, "; {loop_var}++) {{").unwrap();
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
                self.writeln(&format!("void {}({}) {{ ", name, args_str.join(", ")));
                self.indent_level += 1;
                self.render_node(body);
                self.indent_level -= 1;
                self.writeln("} ");
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
                self.writeln("); ");
            }
            Op::Assign { dst, src } => {
                self.write_indent();
                self.render_node(dst);
                write!(self.buffer, " = ").unwrap();
                self.render_node(src);
                self.writeln("; ");
            }
            Op::Store { dst, src } => {
                self.write_indent();
                self.render_node(dst);
                write!(self.buffer, " = ").unwrap();
                self.render_node(src);
                self.writeln(";");
            }
            Op::Load(addr) => {
                // The address itself is what we want to render, e.g., buffer[index]
                self.render_node(addr);
            }
            Op::BufferIndex { buffer, index } => {
                write!(self.buffer, "(").unwrap();
                self.render_node(buffer);
                write!(self.buffer, ")").unwrap();
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
                write!(self.buffer, "({})", self.dtype_to_c(dtype)).unwrap();
                self.render_node(&ast.src[0]);
            }
            Op::Add => self.render_binary_op("+", ast),
            Op::Mul => self.render_binary_op("*", ast),
            Op::Max => self.render_binary_op_func("fmax", ast),
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
            DType::U64 => "size_t".to_string(),
            DType::Void => "void".to_string(),
            DType::Ptr(inner) => format!("{}*", self.dtype_to_c(inner)),
            _ => panic!("DType {{dtype:?}} not supported in C renderer"),
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
        // Add standard headers
        self.buffer.push_str("#include <stddef.h>\n");
        self.buffer.push_str("#include <math.h>\n\n"); // For fmax, etc.

        self.render_node(&ast);
        let code = self.buffer.clone();
        debug!("Rendered C code:\n{code}");
        code
    }
}

/// A C compiler that uses shell commands to compile C code into a dynamic library.
#[derive(Default)]
pub struct CCompiler {
    // Options for the C compiler can be added here.
}

impl CCompiler {
    /// Checks if a C compiler is available on the system by running `cc --version`.
    pub fn check_availability(&self) -> bool {
        let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let result = std::process::Command::new(compiler)
            .arg("--version")
            .output();
        match result {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }
}

/// A kernel representing a function loaded from a C dynamic library.
/// It owns the library to ensure it stays loaded for the lifetime of the kernel.
pub struct CKernel {
    library: Library,
    func_name: String,
}

impl<Var: Buffer> Kernel<Var> for CKernel {
    fn call(&self, mut buffers: Vec<Var>, shape_variables: Vec<usize>) -> Vec<Var> {
        type CFunc = unsafe extern "C" fn(*mut *mut c_void, *const usize);

        unsafe {
            let func: Symbol<CFunc> = self
                .library
                .get(self.func_name.as_bytes())
                .expect("Failed to load symbol from library");

            let mut buffer_ptrs: Vec<*mut c_void> =
                buffers.iter_mut().map(|b| b.as_mut_ptr()).collect();

            func(buffer_ptrs.as_mut_ptr(), shape_variables.as_ptr());
        }
        buffers
    }
}

impl<Var: Buffer, CodeRepr, CompilerOption> Compiler<Var, CodeRepr, CompilerOption> for CCompiler
where
    CodeRepr: AsRef<str>,
{
    fn new() -> Self {
        CCompiler::default()
    }

    fn is_available(&self) -> bool {
        self.check_availability()
    }

    fn with_option(&mut self, _option: CompilerOption) {
        unimplemented!();
    }

    fn compile(&mut self, code: CodeRepr) -> impl Kernel<Var> {
        let mut source_file = tempfile::Builder::new()
            .prefix("kernel")
            .suffix(".c")
            .tempfile_in("/tmp")
            .unwrap();
        std::io::Write::write_all(&mut source_file, code.as_ref().as_bytes()).unwrap();

        let out_dir = tempfile::tempdir_in("/tmp").unwrap();

        let (lib_name, compiler) = if cfg!(target_os = "macos") {
            ("kernel.dylib", "clang")
        } else {
            ("kernel.so", "gcc")
        };
        let lib_path = out_dir.path().join(lib_name);

        let command = format!(
            "{} -shared -fPIC -O3 -o {} {}",
            compiler,
            lib_path.to_str().unwrap(),
            source_file.path().to_str().unwrap()
        );

        debug!("Running compile command: {command}");

        let output = std::process::Command::new(compiler)
            .arg("-shared")
            .arg("-fPIC")
            .arg("-O3")
            .arg("-o")
            .arg(&lib_path)
            .arg(source_file.path())
            .output()
            .expect("Failed to execute compiler");

        if !output.status.success() {
            panic!(
                "Compiler failed with status {:?}:\n{}",
                output.status,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        let library = unsafe { Library::new(&lib_path).expect("Failed to load dynamic library") };

        let func_name = "kernel_main".to_string();

        CKernel { library, func_name }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::backend::Buffer;
    use crate::tensor::shape::expr::Expr;
    use std::ffi::c_void;

    // A mock buffer for testing purposes.
    struct MockBuffer {
        data: Vec<f32>,
    }

    impl Buffer for MockBuffer {
        fn as_mut_ptr(&mut self) -> *mut c_void {
            self.data.as_mut_ptr() as *mut c_void
        }
        fn dtype(&self) -> DType {
            DType::F32 // Mock implementation
        }
        fn shape(&self) -> Vec<Expr> {
            vec![] // Mock implementation
        }
    }

    #[test]
    fn test_ccompiler_availability() {
        let compiler = CCompiler::default();
        assert!(compiler.check_availability());
    }
}

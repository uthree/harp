//! C language backend for rendering the AST.

use crate::ast::{AstNode, Const, DType, Op};
use crate::backend::{Buffer, Compiler, Kernel, Renderer};
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::fmt::Write;
use tempfile::NamedTempFile;

use log::debug;

// --- Existing CRenderer implementation ---

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
                self.writeln(") {");
                self.indent_level += 1;
                self.render_node(block);
                self.indent_level -= 1;
                self.writeln("} ");
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
                write!(self.buffer, "*(").unwrap();
                self.render_node(dst);
                write!(self.buffer, ") = ").unwrap();
                self.render_node(src);
                self.writeln("; ");
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
        self.render_node(&ast);
        let code = self.buffer.clone();
        debug!("Rendered C code:\n{}", code);
        code
    }
}

// --- New CCompiler and CKernel implementation ---

/// A C compiler that uses the `cc` crate to compile C code into a dynamic library.
#[derive(Default)]
pub struct CCompiler {
    // Options for the C compiler can be added here.
}

impl CCompiler {
    /// Checks if a C compiler is available on the system.
    pub fn check_availability(&self) -> bool {
        cc::Build::new().try_get_compiler().is_ok()
    }
}

/// A kernel representing a function loaded from a C dynamic library.
/// It owns the library to ensure it stays loaded for the lifetime of the kernel.
pub struct CKernel {
    library: Library,
    func_name: String,
}

impl<Var: Buffer> Kernel<Var> for CKernel {
    fn call(&self, buffers: Vec<Var>, shape_variables: Vec<usize>) -> Vec<Var> {
        type CFunc = unsafe extern "C" fn(*mut *mut c_void, *const usize);

        unsafe {
            let func: Symbol<CFunc> = self
                .library
                .get(self.func_name.as_bytes())
                .expect("Failed to load symbol from library");

            let mut buffer_ptrs: Vec<*mut c_void> = buffers
                .iter()
                .map(|b| std::mem::transmute(b)) // UNSAFE: Placeholder
                .collect();

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
        let mut source_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut source_file, code.as_ref().as_bytes()).unwrap();

        let out_dir = tempfile::tempdir().unwrap();
        let lib_name = "kernel";

        cc::Build::new()
            .file(source_file.path())
            .shared_flag(true)
            .pic(true)
            .opt_level(3)
            .out_dir(out_dir.path())
            .compile(lib_name);

        let lib_path = out_dir.path().join(format!("lib{lib_name}.so"));
        #[cfg(target_os = "macos")]
        let lib_path = out_dir.path().join(format!("lib{lib_name}.dylib"));
        #[cfg(target_os = "windows")]
        let lib_path = out_dir.path().join(format!("{}.dll", lib_name));

        let library = unsafe { Library::new(&lib_path).expect("Failed to load dynamic library") };

        let func_name = "kernel_main".to_string();

        CKernel { library, func_name }
    }
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;
    use crate::backend::{Buffer, Compiler};

    #[derive(Debug)]
    struct MockBuffer(Vec<f32>);
    impl Buffer for MockBuffer {}

    #[test]
    fn test_render_func_def() {
        let args = vec![
            ("a".to_string(), DType::Ptr(Box::new(DType::F32))),
            ("b".to_string(), DType::I32),
        ];
        let body = AstNode::new(Op::Block, vec![], DType::Void);
        let func_def = AstNode::func_def("my_func", args, body);

        let mut renderer = CRenderer::new();
        let code = renderer.render(func_def);
        let expected = "void my_func(float* a, int b)";
        assert!(code.contains(expected));
    }

    #[test]
    #[ignore]
    fn test_compile_and_run_c_kernel() {
        let c_code = r#"
            #include <stddef.h>
            void kernel_main(void** buffers, size_t* shape_vars) {
                float* buf0 = (float*)buffers[0];
                float* buf1 = (float*)buffers[1];
                buf0[0] += buf1[0];
            }
        "#;

        let mut compiler = CCompiler::default();
        assert!(compiler.check_availability());

        let kernel = <CCompiler as Compiler<MockBuffer, _, ()>>::compile(&mut compiler, c_code);

        let buf1 = MockBuffer(vec![1.0, 2.0]);
        let buf2 = MockBuffer(vec![3.0, 4.0]);

        let buffers = vec![buf1, buf2];
        let shape_vars = vec![];

        let result_buffers = kernel.call(buffers, shape_vars);
        assert_eq!(result_buffers.len(), 2);
    }
}

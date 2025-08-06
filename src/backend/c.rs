//! C language backend for rendering the AST.

use crate::ast::{AstNode, AstOp, Const, DType};
use crate::backend::{Buffer, Compiler, Kernel, Renderer};
use libloading::{Library, Symbol};
use log::debug;
use std::ffi::c_void;
use std::fmt::Write;

/// A buffer allocated on the C side.
pub struct CBuffer {
    /// Raw pointer to the allocated memory.
    pub ptr: *mut c_void,
    /// The number of elements in the buffer.
    pub size: usize,
    /// The data type of the elements.
    pub dtype: DType,
}

impl Buffer for CBuffer {
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.ptr as *mut u8,
                self.size * self.dtype.size_in_bytes(),
            )
        }
    }

    fn dtype(&self) -> DType {
        self.dtype.clone()
    }

    fn shape(&self) -> Vec<usize> {
        vec![self.size]
    }
}

impl Drop for CBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                libc::free(self.ptr);
            }
        }
    }
}

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
            AstOp::Block => {
                self.writeln("{ ");
                self.indent_level += 1;
                for node in &ast.src {
                    self.render_node(node);
                }
                self.indent_level -= 1;
                self.writeln("} ");
            }
            AstOp::Range {
                loop_var,
                max,
                block,
            } => {
                self.write_indent();
                write!(self.buffer, "for (int {loop_var} = 0; {loop_var} < ").unwrap();
                self.render_node(max);
                writeln!(self.buffer, "; {loop_var}++) {{ ").unwrap();
                self.indent_level += 1;
                self.render_node(block);
                self.indent_level -= 1;
                self.writeln("} ");
            }
            AstOp::Func { name, args, body } => {
                let args_str: Vec<String> = args
                    .iter()
                    .map(|(name, dtype)| format!("{} {}", self.dtype_to_c(dtype), name))
                    .collect();
                let args_str = args_str.join(", ");
                self.writeln(&format!("void {name}({args_str}) {{"));
                self.indent_level += 1;
                self.render_node(body);
                self.indent_level -= 1;
                self.writeln("} ");
            }
            AstOp::Call(name) => {
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
            AstOp::Assign { dst, src } => {
                self.write_indent();
                self.render_node(dst);
                write!(self.buffer, " = ").unwrap();
                self.render_node(src);
                self.writeln("; ");
            }
            AstOp::Store { dst, src } => {
                self.write_indent();
                self.render_node(dst);
                write!(self.buffer, " = ").unwrap();
                self.render_node(src);
                self.writeln(";");
            }
            AstOp::Deref(addr) => {
                // The address itself is what we want to render, e.g., buffer[index]
                self.render_node(addr);
            }
            AstOp::BufferIndex { buffer, index } => {
                write!(self.buffer, "(").unwrap();
                self.render_node(buffer);
                write!(self.buffer, ")").unwrap();
                write!(self.buffer, "[").unwrap();
                self.render_node(index);
                write!(self.buffer, "]").unwrap();
            }
            AstOp::Var(name) => {
                write!(self.buffer, "{name}").unwrap();
            }
            AstOp::Const(c) => {
                self.render_const(c);
            }
            AstOp::Cast(dtype) => {
                let rendered_child = self.dtype_to_c(dtype);
                write!(self.buffer, "({rendered_child})").unwrap();
                self.render_node(&ast.src[0]);
            }
            AstOp::Add => self.render_binary_op("+", ast),
            AstOp::Mul => self.render_binary_op("*", ast),
            AstOp::Max => self.render_binary_op_func("fmax", ast),
            _ => unimplemented!("Rendering for `{:?}` is not implemented.", ast.op),
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
            DType::FixedArray(inner, ..) => format!("{}*", self.dtype_to_c(inner)),
            _ => panic!("DType {{dtype:?}} not supported in C renderer"),
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
///
/// It owns the `libloading::Library` instance to ensure that the dynamic library
/// remains loaded in memory for the entire lifetime of the kernel. This prevents
/// dangling pointers to the function.
pub struct CKernel {
    /// The loaded dynamic library.
    library: Library,
    /// The name of the function to be called within the library (e.g., "kernel_main").
    func_name: String,
}

impl Kernel<CBuffer> for CKernel {
    /// Executes the loaded C kernel function.
    ///
    /// This method prepares the arguments and calls the FFI function loaded from the
    /// dynamic library.
    ///
    /// # Arguments
    ///
    /// * `buffers`: A vector of `CBuffer`s that will be passed to the kernel.
    ///   The C function receives this as a `void**`.
    /// * `shape_variables`: A vector of `usize` values representing dynamic shape
    ///   parameters. The C function receives this as a `size_t*`.
    ///
    /// # Returns
    ///
    /// The vector of buffers, which may have been modified by the kernel.
    ///
    /// # Panics
    ///
    /// Panics if the function specified by `func_name` cannot be found in the loaded library.
    fn call(&self, mut buffers: Vec<CBuffer>, shape_variables: Vec<usize>) -> Vec<CBuffer> {
        // Define the signature of the C function we expect to call.
        // This matches the `kernel_main(void** buffers, size_t* shape_vars)`
        // function generated by the `CRenderer`.
        type CFunc = unsafe extern "C" fn(*mut *mut c_void, *const usize);

        unsafe {
            // Load the function symbol from the dynamic library.
            let func: Symbol<CFunc> = self
                .library
                .get(self.func_name.as_bytes())
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to load symbol '{}' from library: {}",
                        self.func_name, e
                    )
                });

            // Prepare the `void**` argument by collecting pointers from the CBuffers.
            let mut buffer_ptrs: Vec<*mut c_void> = buffers.iter_mut().map(|b| b.ptr).collect();

            // Call the C function with the prepared pointers.
            func(buffer_ptrs.as_mut_ptr(), shape_variables.as_ptr());
        }

        // Return the buffers, as they are now populated by the C kernel.
        buffers
    }
}

impl Compiler<CBuffer, String, ()> for CCompiler {
    fn new() -> Self {
        CCompiler::default()
    }

    fn is_available(&self) -> bool {
        self.check_availability()
    }

    fn with_option(&mut self, _option: ()) {
        unimplemented!();
    }

    fn compile(&mut self, code: String) -> impl Kernel<CBuffer> {
        let mut source_file = tempfile::Builder::new()
            .prefix("kernel")
            .suffix(".c")
            .tempfile_in("/tmp")
            .unwrap();
        std::io::Write::write_all(&mut source_file, code.as_bytes()).unwrap();

        let out_dir = tempfile::tempdir_in("/tmp").unwrap();

        let (lib_name, compiler) = if cfg!(target_os = "macos") {
            ("kernel.dylib", "clang")
        } else {
            ("kernel.so", "gcc")
        };
        let lib_path = out_dir.path().join(lib_name);

        debug!(
            "Running compile command: {} -shared -fPIC -O3 -o {} {}",
            compiler,
            lib_path.to_str().unwrap(),
            source_file.path().to_str().unwrap()
        );

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

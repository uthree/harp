use super::c::compiler::ClangCompiler;
use super::c::renderer::CStyleRenderer;
use super::{Backend, BackendError, BackendOptions, Buffer, Buffer_, Compiler, Renderer};
use crate::uop::UOp;
use log::debug;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

/// A `Backend` that uses Clang to compile and execute kernels.
///
/// This backend orchestrates the rendering of a `UOp` graph to C code,
/// compiling it with Clang into a shared library, and then dynamically
/// loading and executing the kernel function. It manages memory buffers
/// on the host (CPU).
#[derive(Debug)]
pub struct ClangBackend {
    compiler: ClangCompiler,
    renderer: CStyleRenderer,
    buffer_counter: Cell<usize>,
    buffers: RefCell<HashMap<usize, Vec<u8>>>, 
    log_generated_code: bool,
}

impl Default for ClangBackend {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ClangBackend {
    /// Creates a new `ClangBackend`.
    ///
    /// # Errors
    ///
    /// Returns a `BackendError::CompilerNotFound` if `clang` is not found in the system's PATH.
    pub fn new() -> Result<Self, BackendError> {
        let compiler = ClangCompiler;
        if !compiler.is_available() {
            return Err(BackendError::CompilerNotFound("clang".to_string()));
        }
        Ok(Self {
            compiler,
            renderer: CStyleRenderer,
            buffer_counter: Cell::new(0),
            buffers: RefCell::new(HashMap::new()),
            log_generated_code: false,
        })
    }

    /// Enables or disables logging of the generated C code.
    pub fn with_generated_code_logging(mut self, enable: bool) -> Self {
        self.log_generated_code = enable;
        self
    }
}

impl Backend for ClangBackend {
    fn alloc(&self, size: usize, backend: Rc<dyn Backend>) -> Buffer {
        let id = self.buffer_counter.get();
        self.buffer_counter.set(id + 1);
        self.buffers.borrow_mut().insert(id, vec![0; size]);
        Buffer(Rc::new(Buffer_ { id, size, backend }))
    }

    fn free(&self, id: usize) {
        self.buffers.borrow_mut().remove(&id);
    }

    fn get_buffer_ptr(&self, id: usize) -> *mut u8 {
        // This is unsafe, but required for FFI. The pointer is valid as long as the buffer exists.
        self.buffers.borrow_mut().get_mut(&id).unwrap().as_mut_ptr()
    }

    fn compile_and_exec(
        &self,
        uops: &[UOp],
        args: &[&Buffer],
        shape_args: &[usize],
        options: &BackendOptions,
    ) {
        let clang_options = match options {
            BackendOptions::Clang(opts) => opts,
            // _ => panic!("Mismatched backend options: Expected Clang options for ClangBackend"),
        };

        debug!("Compiling and executing UOp kernel: {uops:?}");
        let code = self.renderer.render(uops);

        if self.log_generated_code {
            debug!("--- Generated C Code ---\n{code}\n------------------------");
        }

        let kernel = self.compiler.compile(&code, clang_options).unwrap();
        debug!("Compilation successful, executing kernel");

        kernel.exec(args, shape_args);
        debug!("Execution finished");
    }
}

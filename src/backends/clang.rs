use super::c::compiler::{ClangCompileOptions, ClangCompiler};
use super::c::renderer::CStyleRenderer;
use super::{Backend, BackendError, Buffer, Buffer_, Compiler, Renderer};
use crate::uop::UOp;
use log::debug;
use std::cell::{Cell, RefCell, RefMut};
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Debug)]
pub struct ClangBackend {
    compiler: ClangCompiler,
    renderer: CStyleRenderer,
    compile_options: RefCell<ClangCompileOptions>,
    buffer_counter: Cell<usize>,
    buffers: RefCell<HashMap<usize, Vec<u8>>>,
}

impl Default for ClangBackend {
    fn default() -> Self {
        Self::new().unwrap()
    }
}

impl ClangBackend {
    pub fn new() -> Result<Self, BackendError> {
        let compiler = ClangCompiler;
        if !compiler.is_available() {
            return Err(BackendError::CompilerNotFound("clang".to_string()));
        }
        Ok(Self {
            compiler,
            renderer: CStyleRenderer,
            compile_options: RefCell::new(ClangCompileOptions::default()),
            buffer_counter: Cell::new(0),
            buffers: RefCell::new(HashMap::new()),
        })
    }

    pub fn compiler_options_mut(&self) -> RefMut<ClangCompileOptions> {
        self.compile_options.borrow_mut()
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
        self.buffers.borrow_mut().get_mut(&id).unwrap().as_mut_ptr()
    }

    fn compile_and_exec(&self, uops: &[UOp], args: &[&Buffer], shape_args: &[usize]) {
        debug!("Compiling and executing UOp kernel: {uops:?}");
        let code = self.renderer.render(uops);

        let options = self.compile_options.borrow();
        let kernel = self.compiler.compile(&code, &options).unwrap();
        debug!("Compilation successful, executing kernel");

        kernel.exec(args, shape_args);
        debug!("Execution finished");
    }
}

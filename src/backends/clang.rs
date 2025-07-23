use super::{Backend, Compiler, Renderer, Variable, Variable_};
use crate::uop::UOp;
use log::debug;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use super::c::compiler::{ClangCompiler, ClangCompileOptions};
use super::c::renderer::CStyleRenderer;


#[derive(Debug)]
pub struct ClangBackend {
    compiler: ClangCompiler,
    renderer: CStyleRenderer,
    compile_options: Mutex<ClangCompileOptions>,
    buffer_counter: AtomicUsize,
    buffers: Mutex<HashMap<usize, Vec<u8>>>,
}

impl Default for ClangBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ClangBackend {
    pub fn new() -> Self {
        let compiler = ClangCompiler;
        if !compiler.is_available() {
            panic!("clang not found");
        }
        Self {
            compiler,
            renderer: CStyleRenderer,
            compile_options: Mutex::new(ClangCompileOptions::default()),
            buffer_counter: AtomicUsize::new(0),
            buffers: Mutex::new(HashMap::new()),
        }
    }
}

impl Backend for ClangBackend {
    fn set_optimization_level(&self, level: u8) {
        let mut options = self.compile_options.lock().unwrap();
        options.optimization_level = level;
    }

    fn alloc(&self, size: usize, backend: Arc<dyn Backend>) -> Variable {
        let id = self.buffer_counter.fetch_add(1, Ordering::SeqCst);
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(id, vec![0; size]);
        Variable(Rc::new(Variable_ {
            id,
            size,
            backend,
        }))
    }

    fn free(&self, id: usize) {
        self.buffers.lock().unwrap().remove(&id);
    }

    fn get_buffer_ptr(&self, id: usize) -> *mut u8 {
        let mut buffers = self.buffers.lock().unwrap();
        buffers.get_mut(&id).unwrap().as_mut_ptr()
    }

    fn compile_and_exec(&self, uop: &UOp, args: &[&Variable]) {
        debug!("Compiling and executing UOp AST: {uop:?}");
        let code = self.renderer.render(uop);
        
        let options = self.compile_options.lock().unwrap();
        let kernel = self.compiler.compile(&code, &options).unwrap();
        debug!("Compilation successful, executing kernel");

        kernel.exec(args);
        debug!("Execution finished");
    }
}

use super::{Backend, Compiler, Renderer, Variable, Variable_};
use crate::uop::UOp;
use log::debug;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use super::c::compiler::{GccCompiler, GccCompileOptions};
use super::c::renderer::CStyleRenderer;


#[derive(Debug)]
pub struct CpuBackend {
    compiler: GccCompiler,
    renderer: CStyleRenderer,
    compile_options: Mutex<GccCompileOptions>,
    buffer_counter: AtomicUsize,
    buffers: Mutex<HashMap<usize, Vec<u8>>>,
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuBackend {
    pub fn new() -> Self {
        let compiler = GccCompiler;
        if !compiler.is_available() {
            panic!("gcc not found");
        }
        Self {
            compiler,
            renderer: CStyleRenderer,
            compile_options: Mutex::new(GccCompileOptions::default()),
            buffer_counter: AtomicUsize::new(0),
            buffers: Mutex::new(HashMap::new()),
        }
    }
}

impl Backend for CpuBackend {
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
        debug!("Compiling and executing UOp AST: {:?}", uop);
        let code = self.renderer.render(uop);
        
        let options = self.compile_options.lock().unwrap();
        let kernel = self.compiler.compile(&code, &options).unwrap();
        debug!("Compilation successful, executing kernel");

        kernel.exec(args);
        debug!("Execution finished");
    }
}

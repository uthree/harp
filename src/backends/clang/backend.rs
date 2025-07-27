use crate::backends::clang::compiler::ClangCompiler;
use crate::backends::clang::renderer::CStyleRenderer;
use crate::backends::{Backend, Buffer, Buffer_, Compiler, Renderer};
use crate::optimization::autotuner::BackendOptions;
use crate::uop::UOp;
use log::debug;
use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;
use which::which;

#[derive(Debug)]
pub struct ClangBackend {
    compiler: ClangCompiler,
    renderer: CStyleRenderer,
    id_counter: Rc<std::cell::Cell<usize>>,
    buffers: Rc<std::cell::RefCell<HashMap<usize, Vec<u8>>>>,
}

impl Clone for ClangBackend {
    fn clone(&self) -> Self {
        Self {
            compiler: ClangCompiler::new(self.compiler.compiler_cmd.clone()),
            renderer: CStyleRenderer,
            id_counter: self.id_counter.clone(),
            buffers: self.buffers.clone(),
        }
    }
}

impl Default for ClangBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ClangBackend {
    pub fn new() -> Self {
        let compilers = ["clang", "gcc"];
        let found_compiler = compilers
            .iter()
            .find(|cmd| which(cmd).is_ok())
            .expect("No suitable C compiler (clang, gcc) found in PATH.");

        debug!("Using C compiler: {found_compiler}");

        Self {
            compiler: ClangCompiler::new(found_compiler.to_string()),
            renderer: CStyleRenderer,
            id_counter: Rc::new(std::cell::Cell::new(0)),
            buffers: Rc::new(std::cell::RefCell::new(HashMap::new())),
        }
    }
}

impl Backend for ClangBackend {
    fn alloc(&self, size: usize, backend: Rc<dyn Backend>) -> Buffer {
        let id = self.id_counter.get();
        self.id_counter.set(id + 1);
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
    ) -> Duration {
        let BackendOptions::Clang(clang_options) = options;

        debug!("Compiling and executing UOp kernel: {uops:?}");
        let code = self.renderer.render(uops);

        println!("--- Generated C Code ---\n{code}\n------------------------");
        debug!("--- Generated C Code ---\n{code}\n------------------------");

        let kernel = self.compiler.compile(&code, clang_options).unwrap();
        debug!("Compilation successful, executing kernel");

        let exec_time = kernel.exec(args, shape_args);
        debug!("Execution finished in {exec_time:?}");
        exec_time
    }
}

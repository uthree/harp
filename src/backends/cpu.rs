use super::{Backend, Compiler, Kernel, Renderer, Variable};
use crate::dtype::{DType, IsNumber};
use crate::uop::UOp;
use log::debug;
use rustc_hash::FxHashMap;
use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use super::c::compiler::{GccCompiler, GccCompileOptions};
use super::c::renderer::CStyleRenderer;

#[derive(Debug, Clone)]
pub struct CpuBackend {
    compiler: GccCompiler,
    renderer: CStyleRenderer,
    compile_options: Arc<Mutex<GccCompileOptions>>,
    buffer_counter: Arc<AtomicUsize>,
    buffers: Arc<Mutex<FxHashMap<usize, Vec<u8>>>>,
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
            panic!("gcc not found, which is required for the CPU backend");
        }
        Self {
            compiler,
            renderer: CStyleRenderer,
            compile_options: Arc::new(Mutex::new(GccCompileOptions::default())),
            buffer_counter: Arc::new(AtomicUsize::new(0)),
            buffers: Arc::new(Mutex::new(FxHashMap::default())),
        }
    }
}

impl Backend for CpuBackend {
    fn compile_and_exec(&self, uop: &UOp, args: &[&Variable]) {
        debug!("Compiling and executing UOp AST: {:?}", uop);
        let code = self.renderer.render(uop);
        
        let options = self.compile_options.lock().unwrap();
        let kernel = self.compiler.compile(&code, &options).unwrap();
        debug!("Compilation successful, executing kernel");

        let mut buffers_lock = self.buffers.lock().unwrap();
        let mut bufs: Vec<*mut std::ffi::c_void> = args
            .iter()
            .map(|var| {
                // This is unsafe because we are bypassing the borrow checker to get multiple
                // mutable pointers from the HashMap. This is safe because we know that each
                // `var.id` is unique, so we are accessing different elements.
                let vec_ptr = buffers_lock.get_mut(&var.id).unwrap() as *mut Vec<u8>;
                unsafe { (*vec_ptr).as_mut_ptr() as *mut std::ffi::c_void }
            })
            .collect();
        
        kernel.exec(&mut bufs, &[]);
        debug!("Execution finished");
    }

    fn alloc(&self, size: usize, dtype: DType) -> Variable {
        let id = self.buffer_counter.fetch_add(1, Ordering::SeqCst);
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(id, vec![0; size]);
        Variable {
            id,
            size,
            dtype,
            backend: Arc::new(self.clone()),
        }
    }

    fn copy_to_device(&self, data: &dyn Any, dtype: DType) -> Variable {
        let id = self.buffer_counter.fetch_add(1, Ordering::SeqCst);
        
        let (ptr, len) = match dtype {
            DType::F32 => { let s: &Vec<f32> = data.downcast_ref().unwrap(); (s.as_ptr() as *const u8, s.len() * 4) },
            DType::F64 => { let s: &Vec<f64> = data.downcast_ref().unwrap(); (s.as_ptr() as *const u8, s.len() * 8) },
            _ => todo!(),
        };
        let byte_slice = unsafe { std::slice::from_raw_parts(ptr, len) };

        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(id, byte_slice.to_vec());

        Variable {
            id,
            size: len,
            dtype,
            backend: Arc::new(self.clone()),
        }
    }

    fn copy_from_device(&self, var: &Variable) -> Box<dyn Any> {
        let buffers = self.buffers.lock().unwrap();
        let byte_buffer = buffers.get(&var.id).expect("Variable ID not found in backend buffers");

        assert_eq!(byte_buffer.len(), var.size, "Buffer size mismatch");

        match var.dtype {
            DType::F32 => {
                let num_elements = var.size / 4;
                let mut result_vec = Vec::with_capacity(num_elements);
                unsafe {
                    let typed_slice = std::slice::from_raw_parts(byte_buffer.as_ptr() as *const f32, num_elements);
                    result_vec.extend_from_slice(typed_slice);
                }
                Box::new(result_vec)
            },
            _ => todo!(),
        }
    }
}

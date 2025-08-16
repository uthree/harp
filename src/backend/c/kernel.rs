use crate::backend::Kernel;
use crate::backend::c::CBuffer;
use crate::graph::GraphSignature;
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::sync::Arc;

pub struct CKernel {
    /// The dynamic library containing the compiled kernel.
    pub library: Arc<Library>,
    /// The name of the kernel function.
    pub func_name: String,
    /// Details about the kernel's inputs and outputs.
    pub details: GraphSignature,
}

impl CKernel {
    fn launch(&self, buffers: &[&CBuffer]) {
        unsafe {
            // This is a simplification. A more robust implementation would handle
            // a variable number of arguments based on `self.details.num_params`.
            match buffers.len() {
                1 => {
                    let func: Symbol<unsafe extern "C" fn(*mut c_void)> =
                        self.library.get(self.func_name.as_bytes()).unwrap();
                    func(buffers[0].ptr);
                }
                2 => {
                    let func: Symbol<unsafe extern "C" fn(*mut c_void, *mut c_void)> =
                        self.library.get(self.func_name.as_bytes()).unwrap();
                    func(buffers[0].ptr, buffers[1].ptr);
                }
                3 => {
                    let func: Symbol<unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void)> =
                        self.library.get(self.func_name.as_bytes()).unwrap();
                    func(buffers[0].ptr, buffers[1].ptr, buffers[2].ptr);
                }
                _ => unimplemented!(
                    "Kernel launch with {} buffers is not implemented",
                    buffers.len()
                ),
            }
        }
    }
}

impl Kernel<CBuffer> for CKernel {
    fn details(&self) -> &GraphSignature {
        &self.details
    }

    fn call(&mut self, buffers: Vec<CBuffer>, _shape_variables: &[usize]) -> Vec<CBuffer> {
        // For C backend, we assume the buffers are passed in the correct order.
        // The kernel will operate on them in place or write to output buffers.
        let buffer_refs: Vec<&CBuffer> = buffers.iter().collect();
        self.launch(&buffer_refs);
        // Return the buffers, as they might have been mutated.
        buffers
    }
}

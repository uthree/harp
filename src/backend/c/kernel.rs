use crate::backend::Kernel;
use crate::backend::c::CBuffer;
use crate::graph::GraphSignature;
use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::sync::Arc;

pub struct CKernel {
    /// The dynamic library containing the compiled kernel.
    pub library: Arc<Library>,
    /// The name of the kernel function (should be "kernel_main").
    pub func_name: String,
    /// Details about the kernel's inputs and outputs.
    pub details: GraphSignature,
}

type KernelMainFn = unsafe extern "C" fn(*mut *mut c_void, *const usize);

impl Kernel for CKernel {
    type Buffer = CBuffer;

    fn details(&self) -> &GraphSignature {
        &self.details
    }

    fn call(&mut self, buffers: Vec<CBuffer>, shape_variables: &[usize]) -> Vec<CBuffer> {
        unsafe {
            let func: Symbol<KernelMainFn> = self.library.get(self.func_name.as_bytes()).unwrap();

            let mut buf_ptrs: Vec<*mut c_void> = buffers.iter().map(|b| b.ptr).collect();

            func(buf_ptrs.as_mut_ptr(), shape_variables.as_ptr());
        }
        buffers
    }
}

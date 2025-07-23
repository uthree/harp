use crate::dtype::Number;
use crate::dtype::{DType, IsNumber};
use crate::uop::UOp;
use std::any::Any;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

// Re-export submodules
pub mod c;

// --- Top-level Backend Controller ---
mod cpu;
pub use cpu::CpuBackend;

// --- Common Backend Traits ---

pub trait Backend: Debug + Send + Sync {
    fn compile_and_exec(&self, uop: &UOp, args: &[&Variable]);
    fn copy_to_device(&self, data: &dyn Any, dtype: DType) -> Variable;
    fn copy_from_device(&self, var: &Variable) -> Box<dyn Any>;
    fn alloc(&self, size: usize, dtype: DType) -> Variable;
}

pub trait Compiler {
    type Options: Default + Clone + Debug;
    fn is_available(&self) -> bool;
    fn compile(
        &self,
        source_code: &str,
        options: &Self::Options,
    ) -> Result<Arc<dyn Kernel>, Box<dyn Error>>;
}

pub trait Renderer {
    fn render(&self, uop: &UOp) -> String;
}

pub trait Kernel {
    fn exec(&self, bufs: &mut [*mut std::ffi::c_void], ints: &[i32]);
    fn metadata(&self) -> &KernelMetadata;
}

// --- Common Kernel Structs ---

#[derive(Debug)]
pub struct ArgInfo {
    pub dtype: crate::dtype::DType,
    pub size: usize,
}

#[derive(Debug)]
pub struct KernelMetadata {
    pub args_info: Vec<ArgInfo>,
    pub global_work_size: usize,
    pub local_work_size: usize,
}

// --- Variable ---

#[derive(Debug, Clone)]
pub struct Variable {
    pub id: usize,
    pub size: usize,
    pub dtype: DType,
    pub backend: Arc<dyn Backend>,
}

impl Drop for Variable {
    // In a real framework, the backend would manage memory and Drop would trigger a free.
    // For now, our CPU backend's HashMap handles this when the backend is dropped.
    fn drop(&mut self) {}
}

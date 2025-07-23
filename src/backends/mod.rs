use crate::uop::UOp;
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;
use std::error::Error;

// Re-export submodules
pub mod c;

// --- Top-level Backend Controller ---
mod clang;
pub use clang::ClangBackend;


// --- Common Backend Traits ---

pub trait Backend: Debug {
    fn compile_and_exec(&self, uop: &UOp, args: &[&Variable]);
    fn alloc(&self, size: usize, backend: Rc<dyn Backend>) -> Variable;
    fn free(&self, id: usize);
    fn get_buffer_ptr(&self, id: usize) -> *mut u8;
}

pub trait Compiler {
    type Options: Default + Clone + Debug;
    fn is_available(&self) -> bool;
    fn compile(
        &self,
        source_code: &str,
        options: &Self::Options,
    ) -> Result<Rc<dyn Kernel>, Box<dyn Error>>;
}

pub trait Renderer {
    fn render(&self, uop: &UOp) -> String;
}

pub trait Kernel {
    fn exec(&self, args: &[&Variable]);
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

pub struct Variable_ {
    pub id: usize,
    pub size: usize,
    pub backend: Rc<dyn Backend>,
}

impl Drop for Variable_ {
    fn drop(&mut self) {
        self.backend.free(self.id);
    }
}

#[derive(Clone)]
pub struct Variable(pub Rc<Variable_>);

impl Deref for Variable {
    type Target = Variable_;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

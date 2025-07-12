use std::error::Error;

pub mod c;
pub mod codegen;
pub mod node_renderer;

/// A trait for a backend that can generate and compile code.
pub trait Backend {
    /// Returns the name of the backend.
    fn name(&self) -> &str;

    /// Checks if the backend is available on the system.
    fn is_available(&self) -> bool;
}

/// A trait for a compiled kernel that can be executed.
pub trait Kernel {
    /// Executes the kernel and returns the result.
    /// For now, we assume a simple function with no arguments and a float result.
    fn execute(&self) -> f32;
}

/// A trait for a compiler that can turn source code into a `Kernel`.
pub trait Compiler {
    type Kernel: Kernel;
    /// Checks if the compiler toolchain is available on the system.
    fn is_available(&self) -> bool;

    /// Compiles the given source code into a runnable kernel.
    ///
    /// # Arguments
    ///
    /// * `code` - A string containing the source code to compile.
    ///
    /// # Returns
    ///
    /// A `Result` containing a concrete `Kernel` type, or an error.
    fn compile(&self, code: &str) -> Result<Self::Kernel, Box<dyn Error>>;
}

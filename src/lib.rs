pub mod ast;
pub mod backend;
pub mod graph;
pub mod opt;
pub mod tensor;

pub mod cbuffer {
    pub use crate::backend::c::buffer::CBuffer;
}

/// Initializes the logger for the harp library.
///
///
///
/// This function should be called by the application using this library
/// to enable logging. The log level can be controlled via the `RUST_LOG`
/// environment variable.
///
/// This will ignore errors if the logger is already initialized.
pub fn init_logger() {
    let _ = env_logger::builder().is_test(false).try_init();
}

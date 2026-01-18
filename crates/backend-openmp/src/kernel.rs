//! OpenMP kernel implementation
//!
//! This module provides kernel execution by loading compiled shared libraries
//! with OpenMP parallel code.

use crate::buffer::OpenMPBuffer;
use eclat::backend::global::DeviceKind;
use eclat::backend::traits::{Buffer, Kernel};
use eclat::backend::KernelConfig;
use libloading::{Library, Symbol};
use std::any::Any;
use std::error::Error;
use std::mem::ManuallyDrop;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

/// Type alias for the kernel function signature
type KernelFn = unsafe extern "C" fn();

/// Internal state of OpenMPKernel (shared via Arc)
///
/// Note: We use ManuallyDrop for library and temp_dir to prevent cleanup issues.
/// On Linux, dlclose() on OpenMP libraries can cause SIGSEGV during thread cleanup
/// in libgomp. By leaking the library, we avoid the crash at the cost of minor
/// memory/file leaks that are cleaned up when the process exits.
struct OpenMPKernelInner {
    /// Temporary directory containing the compiled library (kept alive)
    /// Wrapped in ManuallyDrop to prevent deletion while library might still be in use
    #[allow(dead_code)]
    temp_dir: ManuallyDrop<TempDir>,
    /// Loaded library
    /// Wrapped in ManuallyDrop to avoid dlclose() which crashes libgomp on Linux
    library: ManuallyDrop<Library>,
}

// Safety: OpenMPKernelInner can be sent between threads as the library is self-contained
unsafe impl Send for OpenMPKernelInner {}
unsafe impl Sync for OpenMPKernelInner {}

/// OpenMP kernel for parallel CPU execution
///
/// Loads a compiled shared library with OpenMP pragmas and executes the kernel function.
pub struct OpenMPKernel {
    /// Shared internal state (library and temp directory)
    inner: Arc<OpenMPKernelInner>,
    /// Kernel configuration
    config: KernelConfig,
}

impl Clone for OpenMPKernel {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            config: self.config.clone(),
        }
    }
}

// Safety: OpenMPKernel can be sent between threads as the library is self-contained
unsafe impl Send for OpenMPKernel {}
unsafe impl Sync for OpenMPKernel {}

impl OpenMPKernel {
    /// Create a new OpenMPKernel from a compiled library
    pub fn new(
        temp_dir: TempDir,
        lib_path: PathBuf,
        config: KernelConfig,
    ) -> Result<Self, Box<dyn Error + Send + Sync>> {
        // Load the library
        let library = unsafe {
            Library::new(&lib_path)
                .map_err(|e| format!("Failed to load library {:?}: {}", lib_path, e))?
        };

        // Verify entry point exists
        let entry_point = &config.entry_point;
        unsafe {
            let _: Symbol<KernelFn> = library
                .get(entry_point.as_bytes())
                .map_err(|e| format!("Entry point '{}' not found: {}", entry_point, e))?;
        }

        Ok(Self {
            inner: Arc::new(OpenMPKernelInner {
                temp_dir: ManuallyDrop::new(temp_dir),
                library: ManuallyDrop::new(library),
            }),
            config,
        })
    }

    /// Get a reference to the library
    fn library(&self) -> &Library {
        &self.inner.library
    }

    /// Execute the kernel with the given buffer pointers
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - All pointers in input_ptrs and output_ptrs are valid
    /// - The buffers have sufficient size for the kernel operation
    unsafe fn execute_with_ptrs(
        &self,
        input_ptrs: &[*const u8],
        output_ptrs: &[*mut u8],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Build the argument list: inputs first, then outputs
        let mut args: Vec<*mut u8> = Vec::with_capacity(input_ptrs.len() + output_ptrs.len());

        for ptr in input_ptrs {
            args.push(*ptr as *mut u8);
        }
        for ptr in output_ptrs {
            args.push(*ptr);
        }

        let entry_point = &self.config.entry_point;

        // Get the function with the proper signature
        match args.len() {
            1 => {
                type Fn1 = unsafe extern "C" fn(*mut u8);
                let func: Symbol<Fn1> = unsafe { self.library().get(entry_point.as_bytes())? };
                unsafe { func(args[0]) };
            }
            2 => {
                type Fn2 = unsafe extern "C" fn(*mut u8, *mut u8);
                let func: Symbol<Fn2> = unsafe { self.library().get(entry_point.as_bytes())? };
                unsafe { func(args[0], args[1]) };
            }
            3 => {
                type Fn3 = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8);
                let func: Symbol<Fn3> = unsafe { self.library().get(entry_point.as_bytes())? };
                unsafe { func(args[0], args[1], args[2]) };
            }
            4 => {
                type Fn4 = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8);
                let func: Symbol<Fn4> = unsafe { self.library().get(entry_point.as_bytes())? };
                unsafe { func(args[0], args[1], args[2], args[3]) };
            }
            5 => {
                type Fn5 = unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8, *mut u8);
                let func: Symbol<Fn5> = unsafe { self.library().get(entry_point.as_bytes())? };
                unsafe { func(args[0], args[1], args[2], args[3], args[4]) };
            }
            6 => {
                type Fn6 =
                    unsafe extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8, *mut u8, *mut u8);
                let func: Symbol<Fn6> = unsafe { self.library().get(entry_point.as_bytes())? };
                unsafe { func(args[0], args[1], args[2], args[3], args[4], args[5]) };
            }
            n => {
                return Err(format!(
                    "Unsupported number of kernel arguments: {}. Maximum supported is 6.",
                    n
                )
                .into());
            }
        }

        Ok(())
    }
}

impl Kernel for OpenMPKernel {
    fn clone_kernel(&self) -> Box<dyn Kernel> {
        Box::new(self.clone())
    }

    fn config(&self) -> &KernelConfig {
        &self.config
    }

    fn device_kind(&self) -> DeviceKind {
        DeviceKind::OpenMP
    }

    fn execute(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // Get input buffer pointers
        let input_ptrs: Vec<*const u8> = inputs
            .iter()
            .map(|b| {
                b.as_any()
                    .downcast_ref::<OpenMPBuffer>()
                    .expect("Input buffer must be OpenMPBuffer")
                    .as_ptr()
            })
            .collect();

        // Get output buffer pointers
        let output_ptrs: Vec<*mut u8> = outputs
            .iter_mut()
            .map(|b| {
                b.as_any_mut()
                    .downcast_mut::<OpenMPBuffer>()
                    .expect("Output buffer must be OpenMPBuffer")
                    .as_mut_ptr()
            })
            .collect();

        // Execute kernel
        unsafe { self.execute_with_ptrs(&input_ptrs, &output_ptrs) }
    }

    fn execute_with_sizes(
        &self,
        inputs: &[&dyn Buffer],
        outputs: &mut [&mut dyn Buffer],
        _global_size: [usize; 3],
        _local_size: [usize; 3],
    ) -> Result<(), Box<dyn Error + Send + Sync>> {
        // OpenMP backend ignores work sizes (uses internal parallelization)
        self.execute(inputs, outputs)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl std::fmt::Debug for OpenMPKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenMPKernel")
            .field("entry_point", &self.config.entry_point)
            .finish()
    }
}

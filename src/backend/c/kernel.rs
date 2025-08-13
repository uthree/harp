use super::CBuffer;
use crate::backend::{Buffer, Kernel, KernelDetails};
use libc::c_void;
use libloading::{Library, Symbol};
use std::{collections::HashMap, sync::Arc};

/// A kernel representing a function loaded from a C dynamic library.
///
/// It owns the `libloading::Library` instance to ensure that the dynamic library
/// remains loaded in memory for the entire lifetime of the kernel. This prevents
/// dangling pointers to the function.
#[derive(Clone)]
pub struct CKernel {
    /// The loaded dynamic library.
    pub(super) library: Arc<Library>,
    /// The name of the function to be called within the library (e.g., "kernel_main").
    pub(super) func_name: String,
    /// Detailed information about the kernel's inputs and outputs.
    pub details: KernelDetails,
}

impl Kernel<CBuffer> for CKernel {
    fn details(&self) -> &KernelDetails {
        &self.details
    }

    fn call(&mut self, mut buffers: Vec<CBuffer>, shape_variables: &[usize]) -> Vec<CBuffer> {
        let num_inputs = self.details.inputs.len();
        let num_outputs = self.details.outputs.len();

        // 1. Validate the incoming buffers against the kernel's details.
        assert_eq!(
            buffers.len(),
            num_inputs + num_outputs,
            "Mismatched number of buffers: expected {}, got {}",
            num_inputs + num_outputs,
            buffers.len()
        );

        assert_eq!(
            shape_variables.len(),
            self.details.shape_variables.len(),
            "Mismatched number of shape variables: expected {}, got {}",
            self.details.shape_variables.len(),
            shape_variables.len()
        );

        // Create a map from shape variable names to their concrete values.
        let shape_vars_map: HashMap<String, i64> = self
            .details
            .shape_variables
            .iter()
            .cloned()
            .zip(shape_variables.iter().map(|&v| v as i64))
            .collect();

        for (i, (buffer, buffer_info)) in buffers
            .iter()
            .zip(
                self.details
                    .inputs
                    .iter()
                    .chain(self.details.outputs.iter()),
            )
            .enumerate()
        {
            // Validate dtype
            assert_eq!(
                buffer.dtype(),
                buffer_info.dtype,
                "Mismatched dtype for buffer {i}: expected {:?}, got {:?}",
                buffer_info.dtype,
                buffer.dtype()
            );

            // Validate shape
            let expected_shape: Vec<usize> = buffer_info
                .shape
                .iter()
                .map(|expr| expr.evaluate(&shape_vars_map) as usize)
                .collect();
            assert_eq!(
                buffer.shape(),
                expected_shape,
                "Mismatched shape for buffer {i}: expected {:?}, got {:?}",
                expected_shape,
                buffer.shape()
            );
        }

        // 2. If validation passes, execute the C function.
        type CFunc = unsafe extern "C" fn(*mut *mut c_void, *mut *mut c_void, *const usize);

        let (input_buffers, output_buffers) = buffers.split_at_mut(num_inputs);
        let mut input_ptrs: Vec<*mut c_void> = input_buffers.iter_mut().map(|b| b.ptr).collect();
        let mut output_ptrs: Vec<*mut c_void> = output_buffers.iter_mut().map(|b| b.ptr).collect();

        unsafe {
            let func: Symbol<CFunc> =
                self.library
                    .get(self.func_name.as_bytes())
                    .unwrap_or_else(|e| {
                        panic!(
                            "Failed to load symbol '{}' from library: {}",
                            self.func_name, e
                        )
                    });

            func(
                input_ptrs.as_mut_ptr(),
                output_ptrs.as_mut_ptr(),
                shape_variables.as_ptr(),
            );
        }

        // Reconstruct the original buffers vec to return it
        let mut result_buffers = Vec::with_capacity(num_inputs + num_outputs);
        result_buffers.extend_from_slice(input_buffers);
        result_buffers.extend_from_slice(output_buffers);
        result_buffers
    }
}

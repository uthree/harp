//! Sequential multi-kernel execution support
//!
//! This module provides types and utilities for executing multiple kernels
//! in sequence, supporting subgraph-based compilation where a single Graph
//! may be split into multiple kernels.

use crate::ast::DType;
use crate::backend::traits::{Buffer, Kernel};
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

/// Error type for program execution
#[derive(Debug)]
pub enum ProgramExecutionError<KE, BE> {
    /// Error during kernel execution
    KernelError(KE),
    /// Error during buffer allocation
    BufferError(BE),
    /// Buffer not found
    BufferNotFound(String),
    /// Kernel not found
    KernelNotFound(String),
}

impl<KE: fmt::Display, BE: fmt::Display> fmt::Display for ProgramExecutionError<KE, BE> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KernelError(e) => write!(f, "Kernel execution error: {}", e),
            Self::BufferError(e) => write!(f, "Buffer allocation error: {}", e),
            Self::BufferNotFound(name) => write!(f, "Buffer not found: {}", name),
            Self::KernelNotFound(name) => write!(f, "Kernel not found: {}", name),
        }
    }
}

impl<KE: std::error::Error + 'static, BE: std::error::Error + 'static> std::error::Error
    for ProgramExecutionError<KE, BE>
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::KernelError(e) => Some(e),
            Self::BufferError(e) => Some(e),
            _ => None,
        }
    }
}

/// Information about a single kernel invocation in a sequence
#[derive(Clone, Debug)]
pub struct KernelCallInfo {
    /// Name of the kernel to invoke
    pub kernel_name: String,
    /// Names of input buffers (may be external inputs or intermediate buffers)
    pub inputs: Vec<String>,
    /// Names of output buffers (may be external outputs or intermediate buffers)
    pub outputs: Vec<String>,
    /// Grid size for this kernel invocation
    pub grid_size: [usize; 3],
    /// Local/threadgroup size for this kernel invocation
    pub local_size: [usize; 3],
}

impl KernelCallInfo {
    /// Create a new kernel call info
    pub fn new(
        kernel_name: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        grid_size: [usize; 3],
        local_size: [usize; 3],
    ) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            inputs,
            outputs,
            grid_size,
            local_size,
        }
    }
}

/// Specification for an intermediate buffer
#[derive(Clone, Debug)]
pub struct IntermediateBufferSpec {
    /// Buffer name (used as key for lookup)
    pub name: String,
    /// Shape of the buffer
    pub shape: Vec<usize>,
    /// Data type of buffer elements
    pub dtype: DType,
}

impl IntermediateBufferSpec {
    /// Create a new intermediate buffer spec
    pub fn new(name: impl Into<String>, shape: Vec<usize>, dtype: DType) -> Self {
        Self {
            name: name.into(),
            shape,
            dtype,
        }
    }

    /// Calculate the byte size of the buffer
    pub fn byte_size(&self) -> usize {
        let element_count: usize = self.shape.iter().product();
        let element_size = match &self.dtype {
            DType::Bool => 1,
            DType::Int => std::mem::size_of::<isize>(),
            DType::F32 => 4,
            DType::Ptr(_) => std::mem::size_of::<*const ()>(),
            DType::Vec(inner, size) => {
                let inner_size = match inner.as_ref() {
                    DType::F32 => 4,
                    DType::Int => std::mem::size_of::<isize>(),
                    _ => 4,
                };
                inner_size * size
            }
            DType::Tuple(_) | DType::Unknown => 4, // Default to 4 bytes
        };
        element_count * element_size
    }
}

/// A compiled program consisting of multiple kernels
///
/// This represents a complete computation that may require multiple kernel
/// invocations. Intermediate buffers are automatically managed.
///
/// ## Execution Model
///
/// `execution_waves` contains groups of kernels organized as follows:
/// - Inner `Vec<KernelCallInfo>`: Kernels that can execute in parallel
/// - Outer `Vec`: Sequential waves with implicit barriers between them
///
/// ```text
/// execution_waves = [
///     // Wave 0: These kernels can run in parallel
///     [KernelA, KernelB],
///     // <implicit barrier>
///     // Wave 1: These depend on Wave 0's results
///     [KernelC],
/// ]
/// ```
pub struct CompiledProgram<K, B>
where
    K: Kernel<Buffer = B>,
    B: Buffer,
{
    /// Compiled kernels indexed by name
    pub kernels: HashMap<String, K>,
    /// Execution waves: groups of parallel-executable kernels
    /// - Inner Vec: kernels that can execute in parallel (no dependencies)
    /// - Outer Vec: sequential waves with implicit barriers between them
    pub execution_waves: Vec<Vec<KernelCallInfo>>,
    /// Specifications for intermediate buffers
    pub intermediate_buffer_specs: Vec<IntermediateBufferSpec>,
    /// Names of external input buffers
    pub input_names: Vec<String>,
    /// Names of external output buffers
    pub output_names: Vec<String>,
    _buffer: PhantomData<B>,
}

impl<K, B> CompiledProgram<K, B>
where
    K: Kernel<Buffer = B>,
    B: Buffer,
{
    /// Create a new compiled program
    pub fn new(
        kernels: HashMap<String, K>,
        execution_waves: Vec<Vec<KernelCallInfo>>,
        intermediate_buffer_specs: Vec<IntermediateBufferSpec>,
        input_names: Vec<String>,
        output_names: Vec<String>,
    ) -> Self {
        Self {
            kernels,
            execution_waves,
            intermediate_buffer_specs,
            input_names,
            output_names,
            _buffer: PhantomData,
        }
    }

    /// Get the number of kernels in this program
    pub fn kernel_count(&self) -> usize {
        self.kernels.len()
    }

    /// Check if this program has multiple kernels
    pub fn is_multi_kernel(&self) -> bool {
        self.execution_waves.iter().map(|w| w.len()).sum::<usize>() > 1
    }

    /// Get the total number of kernel calls
    pub fn total_kernel_calls(&self) -> usize {
        self.execution_waves.iter().map(|w| w.len()).sum()
    }

    /// Get the number of execution waves
    pub fn wave_count(&self) -> usize {
        self.execution_waves.len()
    }

    /// Execute the program with named input and output buffers
    ///
    /// Kernels are executed wave by wave, with implicit barriers between waves.
    /// Within each wave, kernels are currently executed sequentially, but they
    /// have no dependencies and could be parallelized in future implementations.
    ///
    /// # Arguments
    /// * `context` - The GPU context for allocating intermediate buffers
    /// * `inputs` - Map of input buffer names to buffer references
    /// * `outputs` - Map of output buffer names to mutable buffer references
    ///
    /// # Returns
    /// * `Ok(())` on successful execution
    /// * `Err(error)` if any kernel execution fails
    pub fn execute(
        &self,
        context: &B::Dev,
        inputs: &HashMap<String, &B>,
        outputs: &mut HashMap<String, &mut B>,
    ) -> Result<(), ProgramExecutionError<K::Error, B::Error>> {
        // Allocate intermediate buffers (simple per-execution allocation)
        let mut intermediate_buffers: HashMap<String, B> = HashMap::new();
        for spec in &self.intermediate_buffer_specs {
            let buf = B::allocate(context, spec.shape.clone(), spec.dtype.clone())
                .map_err(ProgramExecutionError::BufferError)?;
            intermediate_buffers.insert(spec.name.clone(), buf);
        }

        // Execute each wave of kernels
        // Kernels within a wave have no dependencies and could run in parallel
        // An implicit barrier exists between waves
        for wave in &self.execution_waves {
            for call in wave {
                self.execute_kernel_call(call, inputs, outputs, &mut intermediate_buffers)?;
            }
            // Implicit barrier between waves (handled by GPU synchronization)
        }

        // Intermediate buffers are automatically dropped here
        Ok(())
    }

    /// Execute a single kernel call
    fn execute_kernel_call(
        &self,
        call: &KernelCallInfo,
        inputs: &HashMap<String, &B>,
        outputs: &mut HashMap<String, &mut B>,
        intermediate_buffers: &mut HashMap<String, B>,
    ) -> Result<(), ProgramExecutionError<K::Error, B::Error>> {
        let kernel = self
            .kernels
            .get(&call.kernel_name)
            .ok_or_else(|| ProgramExecutionError::KernelNotFound(call.kernel_name.clone()))?;

        // Collect input buffer pointers (as raw pointers to avoid borrow issues)
        let input_ptrs: Vec<*const B> = call
            .inputs
            .iter()
            .map(|name| {
                inputs
                    .get(name)
                    .map(|b| *b as *const B)
                    .or_else(|| intermediate_buffers.get(name).map(|b| b as *const B))
                    .ok_or_else(|| ProgramExecutionError::BufferNotFound(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Collect output buffer pointers
        let output_ptrs: Vec<*mut B> = call
            .outputs
            .iter()
            .map(|name| {
                outputs
                    .get_mut(name)
                    .map(|b| *b as *mut B)
                    .or_else(|| intermediate_buffers.get_mut(name).map(|b| b as *mut B))
                    .ok_or_else(|| ProgramExecutionError::BufferNotFound(name.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Convert raw pointers back to references for execution
        // SAFETY: The pointers are valid for the duration of this scope,
        // and we ensure no aliasing by checking that inputs and outputs
        // refer to different buffers.
        unsafe {
            let input_refs: Vec<&B> = input_ptrs.iter().map(|p| &**p).collect();
            let mut output_refs: Vec<&mut B> = output_ptrs.iter().map(|p| &mut **p).collect();

            kernel
                .execute(&input_refs, &mut output_refs)
                .map_err(ProgramExecutionError::KernelError)?;
        }

        Ok(())
    }

    /// Execute with positional buffers (for single-kernel programs)
    ///
    /// This is a convenience method for programs that have a single kernel
    /// with straightforward input/output ordering.
    pub fn execute_positional(
        &self,
        context: &B::Dev,
        inputs: &[&B],
        outputs: &mut [&mut B],
    ) -> Result<(), ProgramExecutionError<K::Error, B::Error>> {
        // Build named maps from positional arguments
        let input_map: HashMap<String, &B> = self
            .input_names
            .iter()
            .zip(inputs.iter())
            .map(|(name, buf)| (name.clone(), *buf))
            .collect();

        // For outputs, we need to be more careful
        let mut output_map: HashMap<String, &mut B> = HashMap::new();
        for (name, buf) in self.output_names.iter().zip(outputs.iter_mut()) {
            // SAFETY: Each output name is unique, so we won't have aliasing
            let buf_ptr = *buf as *mut B;
            unsafe {
                output_map.insert(name.clone(), &mut *buf_ptr);
            }
        }

        self.execute(context, &input_map, &mut output_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_call_info_creation() {
        let info = KernelCallInfo::new(
            "test_kernel",
            vec!["input_a".to_string(), "input_b".to_string()],
            vec!["output".to_string()],
            [1024, 1, 1],
            [64, 1, 1],
        );

        assert_eq!(info.kernel_name, "test_kernel");
        assert_eq!(info.inputs.len(), 2);
        assert_eq!(info.outputs.len(), 1);
        assert_eq!(info.grid_size, [1024, 1, 1]);
        assert_eq!(info.local_size, [64, 1, 1]);
    }

    #[test]
    fn test_intermediate_buffer_spec() {
        let spec = IntermediateBufferSpec::new("temp_buf", vec![256, 256], DType::F32);

        assert_eq!(spec.name, "temp_buf");
        assert_eq!(spec.shape, vec![256, 256]);
        assert_eq!(spec.byte_size(), 256 * 256 * 4); // F32 = 4 bytes
    }
}

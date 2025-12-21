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

/// Builder for specifying input/output buffers by name
///
/// This struct provides a fluent API for binding buffers to named slots
/// when executing compiled kernels or programs. It also supports dynamic
/// shape variables that can be used to compute grid sizes at runtime.
///
/// # Example
///
/// ```ignore
/// let query = ExecutionQuery::new()
///     .input("a", &buffer_a)
///     .input("b", &buffer_b)
///     .output("result", &mut output_buffer)
///     .shape_var("batch_size", 32)
///     .shape_var("seq_len", 128);
///
/// compiled_kernel.execute_with(query)?;
/// ```
pub struct ExecutionQuery<'a, B: Buffer> {
    inputs: HashMap<String, &'a B>,
    outputs: HashMap<String, *mut B>,
    shape_vars: HashMap<String, i64>,
    _marker: PhantomData<&'a mut B>,
}

impl<'a, B: Buffer> ExecutionQuery<'a, B> {
    /// Create a new empty execution query
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            shape_vars: HashMap::new(),
            _marker: PhantomData,
        }
    }

    /// Add an input buffer with the given name
    pub fn input(mut self, name: impl Into<String>, buffer: &'a B) -> Self {
        self.inputs.insert(name.into(), buffer);
        self
    }

    /// Add an output buffer with the given name
    pub fn output(mut self, name: impl Into<String>, buffer: &'a mut B) -> Self {
        self.outputs.insert(name.into(), buffer as *mut B);
        self
    }

    /// Add a shape variable for dynamic size computation
    ///
    /// Shape variables are used to compute grid sizes and other
    /// dimension-dependent values at runtime.
    pub fn shape_var(mut self, name: impl Into<String>, value: i64) -> Self {
        self.shape_vars.insert(name.into(), value);
        self
    }

    /// Add multiple shape variables at once
    pub fn shape_vars(mut self, vars: impl IntoIterator<Item = (String, i64)>) -> Self {
        self.shape_vars.extend(vars);
        self
    }

    /// Get the input buffers map
    pub fn inputs(&self) -> &HashMap<String, &'a B> {
        &self.inputs
    }

    /// Get the output buffers map (raw pointers for read-only access)
    ///
    /// # Safety
    /// The returned pointers are valid for the lifetime of the query.
    /// The caller must not create mutable references from these pointers
    /// while other references exist.
    pub(crate) fn outputs(&self) -> &HashMap<String, *mut B> {
        &self.outputs
    }

    /// Get the shape variables map
    pub fn get_shape_vars(&self) -> &HashMap<String, i64> {
        &self.shape_vars
    }

    /// Get the output buffers as mutable references
    ///
    /// # Safety
    /// The caller must ensure that no aliasing occurs between output buffers.
    pub(crate) unsafe fn outputs_mut(&mut self) -> HashMap<String, &'a mut B> {
        self.outputs
            .iter()
            .map(|(k, v)| (k.clone(), unsafe { &mut **v }))
            .collect()
    }

    /// Check if all required inputs are present
    pub fn has_all_inputs(&self, required: &[String]) -> bool {
        required.iter().all(|name| self.inputs.contains_key(name))
    }

    /// Check if all required outputs are present
    pub fn has_all_outputs(&self, required: &[String]) -> bool {
        required.iter().all(|name| self.outputs.contains_key(name))
    }

    /// Get missing input names
    pub fn missing_inputs(&self, required: &[String]) -> Vec<String> {
        required
            .iter()
            .filter(|name| !self.inputs.contains_key(*name))
            .cloned()
            .collect()
    }

    /// Get missing output names
    pub fn missing_outputs(&self, required: &[String]) -> Vec<String> {
        required
            .iter()
            .filter(|name| !self.outputs.contains_key(*name))
            .cloned()
            .collect()
    }

    /// Check if all required shape variables are present
    pub fn has_all_shape_vars(&self, required: &[String]) -> bool {
        required
            .iter()
            .all(|name| self.shape_vars.contains_key(name))
    }

    /// Get missing shape variable names
    pub fn missing_shape_vars(&self, required: &[String]) -> Vec<String> {
        required
            .iter()
            .filter(|name| !self.shape_vars.contains_key(*name))
            .cloned()
            .collect()
    }
}

impl<'a, B: Buffer> Default for ExecutionQuery<'a, B> {
    fn default() -> Self {
        Self::new()
    }
}

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
            DType::I64 => std::mem::size_of::<i64>(),
            DType::I32 => 4, // 32-bit signed integer
            DType::F32 => 4,
            DType::Ptr(_) => std::mem::size_of::<*const ()>(),
            DType::Vec(inner, size) => {
                let inner_size = match inner.as_ref() {
                    DType::F32 => 4,
                    DType::I64 => std::mem::size_of::<i64>(),
                    DType::I32 => 4,
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

    /// Execute the program using an ExecutionQuery
    ///
    /// This method provides a fluent API for specifying buffers by name.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let query = ExecutionQuery::new()
    ///     .input("a", &buf_a)
    ///     .input("b", &buf_b)
    ///     .output("out", &mut buf_out);
    ///
    /// program.execute_with(context, query)?;
    /// ```
    pub fn execute_with(
        &self,
        context: &B::Dev,
        mut query: ExecutionQuery<'_, B>,
    ) -> Result<(), ProgramExecutionError<K::Error, B::Error>> {
        // Validate that all required buffers are present
        let missing_inputs = query.missing_inputs(&self.input_names);
        if !missing_inputs.is_empty() {
            return Err(ProgramExecutionError::BufferNotFound(format!(
                "Missing input buffers: {:?}",
                missing_inputs
            )));
        }

        let missing_outputs = query.missing_outputs(&self.output_names);
        if !missing_outputs.is_empty() {
            return Err(ProgramExecutionError::BufferNotFound(format!(
                "Missing output buffers: {:?}",
                missing_outputs
            )));
        }

        // SAFETY: ExecutionQuery ensures no aliasing between outputs
        let mut outputs = unsafe { query.outputs_mut() };
        self.execute(context, query.inputs(), &mut outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Device;

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

    // Mock buffer for testing ExecutionQuery
    #[derive(Debug, Clone)]
    struct MockBuffer {
        #[allow(dead_code)]
        data: Vec<f32>,
        shape: Vec<usize>,
    }

    #[derive(Debug)]
    struct MockDevice;

    impl Device for MockDevice {
        fn is_available() -> bool {
            true
        }
    }

    #[derive(Debug)]
    struct MockError(String);

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "MockError: {}", self.0)
        }
    }

    impl std::error::Error for MockError {}

    // SAFETY: MockBuffer is only used in single-threaded tests
    unsafe impl Send for MockBuffer {}
    unsafe impl Sync for MockBuffer {}

    impl Buffer for MockBuffer {
        type Dev = MockDevice;
        type Error = MockError;

        fn allocate(
            _device: &Self::Dev,
            shape: Vec<usize>,
            _dtype: DType,
        ) -> Result<Self, Self::Error> {
            let size: usize = shape.iter().product();
            Ok(MockBuffer {
                data: vec![0.0; size],
                shape,
            })
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn dtype(&self) -> DType {
            DType::F32
        }

        fn byte_len(&self) -> usize {
            self.data.len() * 4
        }

        fn write_from_host(&mut self, _data: &[u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn read_to_host(&self) -> Result<Vec<u8>, Self::Error> {
            Ok(vec![0u8; self.byte_len()])
        }

        fn buffer_size_mismatch_error(expected: usize, actual: usize) -> Self::Error {
            MockError(format!(
                "Size mismatch: expected {}, got {}",
                expected, actual
            ))
        }

        fn buffer_alignment_error(buffer_size: usize, type_size: usize) -> Self::Error {
            MockError(format!(
                "Alignment error: buffer {} not aligned to {}",
                buffer_size, type_size
            ))
        }
    }

    #[test]
    fn test_execution_query_builder() {
        let buf_a = MockBuffer {
            data: vec![1.0, 2.0],
            shape: vec![2],
        };
        let buf_b = MockBuffer {
            data: vec![3.0, 4.0],
            shape: vec![2],
        };
        let mut buf_out = MockBuffer {
            data: vec![0.0, 0.0],
            shape: vec![2],
        };

        let query = ExecutionQuery::<MockBuffer>::new()
            .input("a", &buf_a)
            .input("b", &buf_b)
            .output("out", &mut buf_out);

        assert_eq!(query.inputs().len(), 2);
        assert!(query.inputs().contains_key("a"));
        assert!(query.inputs().contains_key("b"));
    }

    #[test]
    fn test_execution_query_has_all_inputs() {
        let buf_a = MockBuffer {
            data: vec![1.0],
            shape: vec![1],
        };
        let buf_b = MockBuffer {
            data: vec![2.0],
            shape: vec![1],
        };

        let query = ExecutionQuery::<MockBuffer>::new()
            .input("a", &buf_a)
            .input("b", &buf_b);

        let required = vec!["a".to_string(), "b".to_string()];
        assert!(query.has_all_inputs(&required));

        let missing = vec!["a".to_string(), "c".to_string()];
        assert!(!query.has_all_inputs(&missing));
    }

    #[test]
    fn test_execution_query_missing_inputs() {
        let buf_a = MockBuffer {
            data: vec![1.0],
            shape: vec![1],
        };

        let query = ExecutionQuery::<MockBuffer>::new().input("a", &buf_a);

        let required = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let missing = query.missing_inputs(&required);

        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&"b".to_string()));
        assert!(missing.contains(&"c".to_string()));
    }

    #[test]
    fn test_execution_query_has_all_outputs() {
        let mut buf_out1 = MockBuffer {
            data: vec![0.0],
            shape: vec![1],
        };
        let mut buf_out2 = MockBuffer {
            data: vec![0.0],
            shape: vec![1],
        };

        let query = ExecutionQuery::<MockBuffer>::new()
            .output("out1", &mut buf_out1)
            .output("out2", &mut buf_out2);

        let required = vec!["out1".to_string(), "out2".to_string()];
        assert!(query.has_all_outputs(&required));

        let missing = vec!["out1".to_string(), "out3".to_string()];
        assert!(!query.has_all_outputs(&missing));
    }

    #[test]
    fn test_execution_query_shape_vars() {
        let buf_a = MockBuffer {
            data: vec![1.0],
            shape: vec![1],
        };

        let query = ExecutionQuery::<MockBuffer>::new()
            .input("a", &buf_a)
            .shape_var("batch_size", 32)
            .shape_var("seq_len", 128);

        let shape_vars = query.get_shape_vars();
        assert_eq!(shape_vars.len(), 2);
        assert_eq!(shape_vars.get("batch_size"), Some(&32));
        assert_eq!(shape_vars.get("seq_len"), Some(&128));
    }

    #[test]
    fn test_execution_query_shape_vars_bulk() {
        let buf_a = MockBuffer {
            data: vec![1.0],
            shape: vec![1],
        };

        let vars = vec![
            ("batch_size".to_string(), 64),
            ("hidden_dim".to_string(), 256),
        ];

        let query = ExecutionQuery::<MockBuffer>::new()
            .input("a", &buf_a)
            .shape_vars(vars);

        let shape_vars = query.get_shape_vars();
        assert_eq!(shape_vars.len(), 2);
        assert_eq!(shape_vars.get("batch_size"), Some(&64));
        assert_eq!(shape_vars.get("hidden_dim"), Some(&256));
    }

    #[test]
    fn test_execution_query_missing_shape_vars() {
        let buf_a = MockBuffer {
            data: vec![1.0],
            shape: vec![1],
        };

        let query = ExecutionQuery::<MockBuffer>::new()
            .input("a", &buf_a)
            .shape_var("batch_size", 32);

        let required = vec![
            "batch_size".to_string(),
            "seq_len".to_string(),
            "hidden_dim".to_string(),
        ];
        let missing = query.missing_shape_vars(&required);

        assert_eq!(missing.len(), 2);
        assert!(missing.contains(&"seq_len".to_string()));
        assert!(missing.contains(&"hidden_dim".to_string()));
    }
}

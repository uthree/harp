//! Execution query support
//!
//! This module provides `ExecutionQuery` for binding buffers and shape variables
//! when executing compiled kernels.

use crate::backend::traits::TypedBuffer;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

/// Builder for specifying input/output buffers by name
///
/// This struct provides a fluent API for binding buffers to named slots
/// when executing compiled kernels. It also supports dynamic
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
pub struct ExecutionQuery<'a, B: TypedBuffer> {
    inputs: HashMap<String, &'a B>,
    outputs: HashMap<String, *mut B>,
    shape_vars: HashMap<String, i64>,
    _marker: PhantomData<&'a mut B>,
}

impl<'a, B: TypedBuffer> ExecutionQuery<'a, B> {
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

impl<'a, B: TypedBuffer> Default for ExecutionQuery<'a, B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Error type for kernel/program execution
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::backend::Device;

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

    impl TypedBuffer for MockBuffer {
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
            self.data.len() * std::mem::size_of::<f32>()
        }

        fn read_to_host(&self) -> Result<Vec<u8>, Self::Error> {
            Ok(vec![0u8; self.byte_len()])
        }

        fn write_from_host(&mut self, _data: &[u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn buffer_size_mismatch_error(expected: usize, actual: usize) -> Self::Error {
            MockError(format!(
                "Buffer size mismatch: expected {}, got {}",
                expected, actual
            ))
        }

        fn buffer_alignment_error(buffer_size: usize, type_size: usize) -> Self::Error {
            MockError(format!(
                "Buffer alignment error: buffer size {} not aligned to type size {}",
                buffer_size, type_size
            ))
        }
    }

    #[test]
    fn test_execution_query_inputs() {
        let buf_a = MockBuffer {
            data: vec![1.0, 2.0],
            shape: vec![2],
        };
        let buf_b = MockBuffer {
            data: vec![3.0, 4.0],
            shape: vec![2],
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
    fn test_execution_query_outputs() {
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

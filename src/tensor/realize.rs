//! Tensor realization and data management.
//!
//! This module provides methods for:
//! - Setting input data (`set_data()`)
//! - Executing computation graphs (`realize()`)
//! - Reading results back to host (`to_vec()`)

use std::cell::Ref;
use std::collections::HashMap;
use std::sync::RwLock;

use crate::ast::{DType, TensorDType};
use crate::backend::{Buffer, ExecutionError, has_default_device};
use crate::graph::GraphInner;

use super::dim::Dimension;
use super::tensor::Tensor;

// ============================================================================
// Global Buffer Cache
// ============================================================================

/// Global cache for input tensor buffers.
///
/// When `set_data()` is called on an input tensor, its buffer is registered here
/// so that `realize()` can access it when executing computation graphs.
///
/// Keys are GraphNode pointers cast to usize for Send/Sync compatibility.
/// Note: Buffer trait already requires Send + Sync.
static INPUT_BUFFER_CACHE: std::sync::LazyLock<RwLock<HashMap<usize, Box<dyn Buffer>>>> =
    std::sync::LazyLock::new(|| RwLock::new(HashMap::new()));

/// Convert a GraphInner pointer to a usize key.
fn ptr_to_key(ptr: *const GraphInner) -> usize {
    ptr as usize
}

/// Register a buffer in the global cache for later retrieval.
fn register_input_buffer(graph_ptr: *const GraphInner, buffer: Box<dyn Buffer>) {
    let mut cache = INPUT_BUFFER_CACHE.write().unwrap();
    cache.insert(ptr_to_key(graph_ptr), buffer);
}

/// Get a clone of a buffer from the cache.
fn clone_input_buffer(graph_ptr: *const GraphInner) -> Option<Box<dyn Buffer>> {
    let cache = INPUT_BUFFER_CACHE.read().unwrap();
    cache.get(&ptr_to_key(graph_ptr)).map(|b| b.clone_buffer())
}

/// Remove a buffer from the cache.
#[allow(dead_code)]
fn remove_input_buffer(graph_ptr: *const GraphInner) {
    let mut cache = INPUT_BUFFER_CACHE.write().unwrap();
    cache.remove(&ptr_to_key(graph_ptr));
}

/// Check if a buffer exists in the cache.
fn has_input_buffer(graph_ptr: *const GraphInner) -> bool {
    let cache = INPUT_BUFFER_CACHE.read().unwrap();
    cache.contains_key(&ptr_to_key(graph_ptr))
}

// ============================================================================
// Realization Methods
// ============================================================================

impl<D: Dimension, T: TensorDType> Tensor<D, T> {
    /// Check if this tensor has been realized (has data in a buffer).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
    /// assert!(!x.is_realized());
    /// x.set_data(&data)?;
    /// assert!(x.is_realized());
    /// ```
    pub fn is_realized(&self) -> bool {
        self.inner.buffer.borrow().is_some()
    }

    /// Get a reference to the realized buffer.
    ///
    /// # Panics
    ///
    /// Panics if the tensor has not been realized.
    pub fn buffer(&self) -> Ref<'_, Box<dyn Buffer>> {
        Ref::map(self.inner.buffer.borrow(), |opt| {
            opt.as_ref()
                .expect("Tensor not realized. Call realize() or set_data() first.")
        })
    }

    /// Try to get a reference to the realized buffer.
    ///
    /// Returns `None` if the tensor has not been realized.
    pub fn try_buffer(&self) -> Option<Ref<'_, Box<dyn Buffer>>> {
        let borrow = self.inner.buffer.borrow();
        if borrow.is_some() {
            Some(Ref::map(borrow, |opt| opt.as_ref().unwrap()))
        } else {
            None
        }
    }

    /// Execute the computation graph and materialize the tensor data.
    ///
    /// This method:
    /// 1. Checks if the tensor is already realized (returns early if so)
    /// 2. Collects all input tensors that need data
    /// 3. Executes the computation graph using the current device
    /// 4. Stores the result buffer
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No device is set
    /// - Required input data is missing
    /// - Execution fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2> = Tensor::input([32, 64], DType::F32);
    /// let y: Tensor<D2> = Tensor::input([32, 64], DType::F32);
    /// x.set_data(&x_data)?;
    /// y.set_data(&y_data)?;
    ///
    /// let z = &x + &y;
    /// z.realize()?;  // Executes the computation
    ///
    /// let result: Vec<f32> = z.to_vec()?;
    /// ```
    pub fn realize(&self) -> Result<&Self, ExecutionError> {
        use crate::backend::execute_graph;
        use crate::graph::collect_inputs;

        // Check if already realized
        if self.is_realized() {
            return Ok(self);
        }

        // Check device is set
        if !has_default_device() {
            return Err(ExecutionError::NoDevice);
        }

        // For input tensors (external nodes), check if data was set
        if self.inner.graph.is_external() {
            let ptr = std::rc::Rc::as_ptr(&self.inner.graph.0);
            if has_input_buffer(ptr) {
                // Copy buffer from cache to self
                let buffer = clone_input_buffer(ptr).ok_or_else(|| {
                    ExecutionError::Internal("Buffer disappeared from cache".to_string())
                })?;
                *self.inner.buffer.borrow_mut() = Some(buffer);
                return Ok(self);
            }
            return Err(ExecutionError::MissingInput(
                "Input tensor has no data. Call set_data() first.".to_string(),
            ));
        }

        // For computed tensors, collect input buffers and execute graph
        let input_nodes = collect_inputs(std::slice::from_ref(&self.inner.graph));

        // Build input buffer map
        let mut input_buffers: HashMap<*const GraphInner, Box<dyn Buffer>> =
            HashMap::with_capacity(input_nodes.len());

        for input_node in &input_nodes {
            let ptr = std::rc::Rc::as_ptr(&input_node.0);
            let name = input_node.name();

            #[cfg(debug_assertions)]
            eprintln!(
                "[realize] Input node: name={:?}, ptr={:p}, has_cached_buffer={}",
                name,
                ptr,
                has_input_buffer(ptr)
            );

            if let Some(buffer) = clone_input_buffer(ptr) {
                input_buffers.insert(ptr, buffer);
            } else {
                // Check if this is a special constant node (ones, zeros)
                if let Some(buffer) = create_constant_buffer(input_node, name)? {
                    #[cfg(debug_assertions)]
                    eprintln!("[realize] Created constant buffer for {:?}", name);
                    input_buffers.insert(ptr, buffer);
                } else {
                    let name_str = name
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| format!("node@{:p}", ptr));
                    return Err(ExecutionError::MissingInput(format!(
                        "Input tensor '{}' has no data. Call set_data() first.",
                        name_str
                    )));
                }
            }
        }

        #[cfg(debug_assertions)]
        {
            eprintln!(
                "[realize] Executing graph with {} inputs",
                input_buffers.len()
            );
            for (ptr, buf) in &input_buffers {
                if let Ok(data) = buf.read_to_host() {
                    let len = std::cmp::min(data.len(), 32);
                    eprintln!(
                        "[realize] Input buffer {:p}: {} bytes, first bytes: {:?}",
                        ptr,
                        data.len(),
                        &data[..len]
                    );
                }
            }
        }

        // Execute the computation graph
        let mut result = execute_graph(std::slice::from_ref(&self.inner.graph), &input_buffers)?;

        #[cfg(debug_assertions)]
        {
            if let Some(out_buf) = result.get(0)
                && let Ok(data) = out_buf.read_to_host()
            {
                let len = std::cmp::min(data.len(), 32);
                eprintln!(
                    "[realize] Output buffer: {} bytes, first bytes: {:?}",
                    data.len(),
                    &data[..len]
                );
            }
        }

        // Store the result buffer (output index 0 corresponds to our root)
        let buffer = result.take(0).ok_or_else(|| {
            ExecutionError::Internal("No output buffer returned from execution".to_string())
        })?;

        *self.inner.buffer.borrow_mut() = Some(buffer);

        Ok(self)
    }

    /// Read tensor data back to host as a Vec<T>.
    ///
    /// The type `T` is determined at compile time by the tensor's type parameter.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor has not been realized
    /// - Reading from the buffer fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D1, f32> = Tensor::input([4]);
    /// x.set_data(&[1.0f32, 2.0, 3.0, 4.0])?;
    ///
    /// let data: Vec<f32> = x.to_vec()?;
    /// assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn to_vec(&self) -> Result<Vec<T>, ExecutionError>
    where
        T: Clone,
    {
        // Check if realized
        if !self.is_realized() {
            return Err(ExecutionError::NotRealized);
        }

        // Read from buffer
        let buffer = self.inner.buffer.borrow();
        let buffer = buffer.as_ref().unwrap();

        let bytes = buffer
            .read_to_host()
            .map_err(|e| ExecutionError::ExecutionFailed(e.to_string()))?;

        // Convert bytes to Vec<T>
        let element_size = std::mem::size_of::<T>();
        let count = bytes.len() / element_size;

        let mut result = Vec::with_capacity(count);
        let ptr = bytes.as_ptr() as *const T;
        unsafe {
            for i in 0..count {
                result.push((*ptr.add(i)).clone());
            }
        }

        Ok(result)
    }

    /// Set the input data for this tensor.
    ///
    /// This allocates a buffer on the current device and copies the data.
    /// The data type is determined at compile time by the tensor's type parameter `T`.
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data that matches the tensor's shape
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No device is set
    /// - Data size doesn't match tensor size
    /// - Buffer allocation fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D2, f32> = Tensor::input([2, 3]);
    /// x.set_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// ```
    pub fn set_data(&self, data: &[T]) -> Result<(), ExecutionError> {
        // Check device is set
        if !has_default_device() {
            return Err(ExecutionError::NoDevice);
        }

        // Verify size matches
        let expected_size = self.numel();
        if data.len() != expected_size {
            return Err(ExecutionError::ShapeMismatch(format!(
                "Data size {} doesn't match tensor size {}",
                data.len(),
                expected_size
            )));
        }

        // Allocate buffer and write data
        let buffer = allocate_buffer(self.shape(), T::DTYPE)?;
        let mut buffer = buffer;

        // Convert data to bytes
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        buffer
            .write_from_host(bytes)
            .map_err(|e| ExecutionError::ExecutionFailed(e.to_string()))?;

        // Register buffer in global cache for realize() to access
        let graph_ptr = std::rc::Rc::as_ptr(&self.inner.graph.0);
        register_input_buffer(graph_ptr, buffer.clone_buffer());

        // Store buffer locally
        *self.inner.buffer.borrow_mut() = Some(buffer);

        Ok(())
    }

    /// Clear the realized buffer, freeing device memory.
    ///
    /// After calling this, the tensor will need to be realized again
    /// before data can be read.
    pub fn clear_buffer(&self) {
        *self.inner.buffer.borrow_mut() = None;
    }
}

// ============================================================================
// Buffer Allocation Helper
// ============================================================================

/// Allocate a buffer on the current device.
///
/// This function dispatches to the appropriate backend based on the
/// currently set device.
fn allocate_buffer(shape: Vec<usize>, dtype: DType) -> Result<Box<dyn Buffer>, ExecutionError> {
    use crate::backend::allocate_buffer_on_default_device;

    allocate_buffer_on_default_device(shape, dtype)
        .map_err(|e| ExecutionError::AllocationFailed(format!("Failed to allocate buffer: {}", e)))
}

/// Create a buffer for special constant nodes (ones, zeros).
///
/// These nodes are created internally during backward pass and need to be
/// initialized with their respective constant values.
///
/// Returns `None` if the node is not a recognized constant type.
fn create_constant_buffer(
    node: &crate::graph::GraphNode,
    name: Option<&str>,
) -> Result<Option<Box<dyn Buffer>>, ExecutionError> {
    let shape: Vec<usize> = node
        .shape()
        .iter()
        .map(|e| {
            e.evaluate()
                .expect("Cannot evaluate shape expression at runtime") as usize
        })
        .collect();
    let dtype = node.dtype().clone();

    match name {
        Some("ones") => {
            let buffer = create_filled_buffer(shape, dtype, 1.0)?;
            Ok(Some(buffer))
        }
        Some("zeros") => {
            let buffer = create_filled_buffer(shape, dtype, 0.0)?;
            Ok(Some(buffer))
        }
        Some("scalar") => {
            // Scalars default to 1.0 (used for initial gradient)
            let buffer = create_filled_buffer(shape, dtype, 1.0)?;
            Ok(Some(buffer))
        }
        _ => Ok(None),
    }
}

/// Create a buffer filled with a constant value.
fn create_filled_buffer(
    shape: Vec<usize>,
    dtype: DType,
    value: f64,
) -> Result<Box<dyn Buffer>, ExecutionError> {
    let mut buffer = allocate_buffer(shape.clone(), dtype.clone())?;
    let numel: usize = shape.iter().product();

    // Create data based on dtype
    let bytes: Vec<u8> = match dtype {
        DType::F32 => {
            let val = value as f32;
            let data: Vec<f32> = vec![val; numel];
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    numel * std::mem::size_of::<f32>(),
                )
                .to_vec()
            }
        }
        DType::F64 => {
            let val = value;
            let data: Vec<f64> = vec![val; numel];
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    numel * std::mem::size_of::<f64>(),
                )
                .to_vec()
            }
        }
        DType::I32 => {
            let val = value as i32;
            let data: Vec<i32> = vec![val; numel];
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    numel * std::mem::size_of::<i32>(),
                )
                .to_vec()
            }
        }
        DType::I64 => {
            let val = value as i64;
            let data: Vec<i64> = vec![val; numel];
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    numel * std::mem::size_of::<i64>(),
                )
                .to_vec()
            }
        }
        DType::U32 => {
            let val = value as u32;
            let data: Vec<u32> = vec![val; numel];
            unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    numel * std::mem::size_of::<u32>(),
                )
                .to_vec()
            }
        }
        _ => {
            return Err(ExecutionError::Internal(format!(
                "Unsupported dtype {:?} for constant buffer",
                dtype
            )));
        }
    };

    buffer
        .write_from_host(&bytes)
        .map_err(|e| ExecutionError::ExecutionFailed(e.to_string()))?;

    Ok(buffer)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dim::{D1, D2};

    #[test]
    fn test_is_realized() {
        let x: Tensor<D2, f32> = Tensor::input([2, 3]);
        assert!(!x.is_realized());
    }

    #[test]
    fn test_realize_no_device() {
        // Clear any existing device
        crate::backend::clear_default_device();

        let x: Tensor<D1, f32> = Tensor::input([4]);
        let result = x.realize();
        assert!(matches!(result, Err(ExecutionError::NoDevice)));
    }

    #[test]
    fn test_to_vec_not_realized() {
        let x: Tensor<D1, f32> = Tensor::input([4]);
        let result = x.to_vec();
        assert!(matches!(result, Err(ExecutionError::NotRealized)));
    }

    #[test]
    fn test_set_data_no_device() {
        // Clear any existing device
        crate::backend::clear_default_device();

        let x: Tensor<D1, f32> = Tensor::input([4]);
        let result = x.set_data(&[1.0f32, 2.0, 3.0, 4.0]);
        assert!(matches!(result, Err(ExecutionError::NoDevice)));
    }

    #[test]
    fn test_set_data_size_mismatch() {
        // Need a device for this test - try to set up a backend
        let _ = crate::backend::set_device_str("c");

        if has_default_device() {
            let x: Tensor<D1, f32> = Tensor::input([4]);
            let result = x.set_data(&[1.0f32, 2.0, 3.0]); // Wrong size

            // If buffer allocation is implemented, we should get ShapeMismatch
            // If not implemented, we'll get Internal error from allocate_buffer
            match &result {
                Err(ExecutionError::ShapeMismatch(_)) => {}
                Err(ExecutionError::Internal(_)) => {} // Buffer allocation not implemented yet
                other => panic!("Unexpected result: {:?}", other),
            }
        }
    }

    /// Integration test for buffer allocation and data transfer with GPU backends
    /// This test is only run when a GPU backend (Metal or OpenCL) is available
    #[test]
    fn test_set_data_and_to_vec_with_gpu() {
        // Try to set up a GPU backend
        // Priority: Metal > OpenCL
        let device_result = crate::backend::set_device_str("metal")
            .or_else(|_| crate::backend::set_device_str("opencl"));

        if device_result.is_err() {
            println!("No GPU backend available, skipping test");
            return;
        }

        // Verify device supports runtime
        let device_kind = crate::backend::get_default_device_kind();
        println!("Testing with device: {:?}", device_kind);

        // Test 1D tensor
        let x: Tensor<D1, f32> = Tensor::input([4]);
        let input_data = [1.0f32, 2.0, 3.0, 4.0];

        let result = x.set_data(&input_data);
        assert!(result.is_ok(), "set_data failed: {:?}", result.err());
        assert!(x.is_realized());

        let output = x.to_vec().expect("to_vec failed");
        assert_eq!(output, input_data.to_vec());

        // Test 2D tensor
        let y: Tensor<D2, f32> = Tensor::input([2, 3]);
        let input_data_2d = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        y.set_data(&input_data_2d).expect("set_data for 2D failed");
        assert!(y.is_realized());

        let output_2d = y.to_vec().expect("to_vec for 2D failed");
        assert_eq!(output_2d, input_data_2d.to_vec());

        // Test clear_buffer
        y.clear_buffer();
        assert!(!y.is_realized());
    }

    /// Test buffer allocation with different data types
    #[test]
    fn test_set_data_different_dtypes() {
        let device_result = crate::backend::set_device_str("metal")
            .or_else(|_| crate::backend::set_device_str("opencl"));

        if device_result.is_err() {
            println!("No GPU backend available, skipping test");
            return;
        }

        // Test f64
        let x_f64: Tensor<D1, f64> = Tensor::input([4]);
        let input_f64 = [1.0f64, 2.0, 3.0, 4.0];
        x_f64.set_data(&input_f64).expect("set_data for f64 failed");
        let output_f64 = x_f64.to_vec().expect("to_vec for f64 failed");
        assert_eq!(output_f64, input_f64.to_vec());

        // Test i32
        let x_i32: Tensor<D1, i32> = Tensor::input([4]);
        let input_i32 = [1i32, 2, 3, 4];
        x_i32.set_data(&input_i32).expect("set_data for i32 failed");
        let output_i32 = x_i32.to_vec().expect("to_vec for i32 failed");
        assert_eq!(output_i32, input_i32.to_vec());
    }

    // ========================================================================
    // Integration Tests for Computation Graph Execution
    // ========================================================================

    /// Test realize with simple element-wise addition
    #[test]
    fn test_realize_add() {
        let device_result = crate::backend::set_device_str("metal")
            .or_else(|_| crate::backend::set_device_str("opencl"));

        if device_result.is_err() {
            println!("No GPU backend available, skipping test");
            return;
        }

        let x: Tensor<D1, f32> = Tensor::input([4]);
        let y: Tensor<D1, f32> = Tensor::input([4]);

        x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
            .expect("set_data for x failed");
        y.set_data(&[5.0f32, 6.0, 7.0, 8.0])
            .expect("set_data for y failed");

        let z = &x + &y;
        let realize_result = z.realize();
        assert!(
            realize_result.is_ok(),
            "realize failed: {:?}",
            realize_result.err()
        );

        let result = z.to_vec().expect("to_vec failed");
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    /// Test realize with element-wise multiplication
    #[test]
    fn test_realize_mul() {
        let device_result = crate::backend::set_device_str("metal")
            .or_else(|_| crate::backend::set_device_str("opencl"));

        if device_result.is_err() {
            println!("No GPU backend available, skipping test");
            return;
        }

        let x: Tensor<D1, f32> = Tensor::input([4]);
        let y: Tensor<D1, f32> = Tensor::input([4]);

        x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
            .expect("set_data for x failed");
        y.set_data(&[2.0f32, 3.0, 4.0, 5.0])
            .expect("set_data for y failed");

        let z = &x * &y;
        z.realize().expect("realize failed");

        let result = z.to_vec().expect("to_vec failed");
        assert_eq!(result, vec![2.0, 6.0, 12.0, 20.0]);
    }

    /// Test realize with 2D tensors
    #[test]
    fn test_realize_2d() {
        let device_result = crate::backend::set_device_str("metal")
            .or_else(|_| crate::backend::set_device_str("opencl"));

        if device_result.is_err() {
            println!("No GPU backend available, skipping test");
            return;
        }

        let x: Tensor<D2, f32> = Tensor::input([2, 3]);
        let y: Tensor<D2, f32> = Tensor::input([2, 3]);

        x.set_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("set_data for x failed");
        y.set_data(&[6.0f32, 5.0, 4.0, 3.0, 2.0, 1.0])
            .expect("set_data for y failed");

        let z = &x + &y;
        z.realize().expect("realize failed");

        let result = z.to_vec().expect("to_vec failed");
        assert_eq!(result, vec![7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
    }

    /// Test that realize is idempotent
    #[test]
    fn test_realize_idempotent() {
        let device_result = crate::backend::set_device_str("metal")
            .or_else(|_| crate::backend::set_device_str("opencl"));

        if device_result.is_err() {
            println!("No GPU backend available, skipping test");
            return;
        }

        let x: Tensor<D1, f32> = Tensor::input([4]);
        x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
            .expect("set_data failed");

        // First realize
        x.realize().expect("first realize failed");
        let result1 = x.to_vec().expect("to_vec failed");

        // Second realize should be a no-op
        x.realize().expect("second realize failed");
        let result2 = x.to_vec().expect("to_vec failed");

        assert_eq!(result1, result2);
    }

    /// Test realize with missing input
    #[test]
    fn test_realize_missing_input() {
        let device_result = crate::backend::set_device_str("metal")
            .or_else(|_| crate::backend::set_device_str("opencl"));

        if device_result.is_err() {
            println!("No GPU backend available, skipping test");
            return;
        }

        let x: Tensor<D1, f32> = Tensor::input([4]);
        let y: Tensor<D1, f32> = Tensor::input([4]);

        // Only set data for x, not y
        x.set_data(&[1.0f32, 2.0, 3.0, 4.0])
            .expect("set_data for x failed");

        let z = &x + &y;
        let result = z.realize();
        assert!(matches!(result, Err(ExecutionError::MissingInput(_))));
    }
}

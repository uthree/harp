//! Tensor realization and data management.
//!
//! This module provides methods for:
//! - Setting input data (`set_data()`)
//! - Executing computation graphs (`realize()`)
//! - Reading results back to host (`to_vec()`)

use std::cell::Ref;

use crate::ast::{DType, TensorDType};
use crate::backend::{Buffer, ExecutionError, has_default_device};

use super::dim::Dimension;
use super::tensor::Tensor;

// ============================================================================
// Realization Methods
// ============================================================================

impl<D: Dimension> Tensor<D> {
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
        // Check if already realized
        if self.is_realized() {
            return Ok(self);
        }

        // Check device is set
        if !has_default_device() {
            return Err(ExecutionError::NoDevice);
        }

        // TODO: Implement full execution pipeline
        // For now, return error for non-input tensors
        if !self.inner.graph.is_external() {
            return Err(ExecutionError::Internal(
                "realize() for computed tensors is not yet implemented. \
                 Only input tensors with set_data() are currently supported."
                    .to_string(),
            ));
        }

        // For input tensors without data, this is an error
        Err(ExecutionError::MissingInput(
            "Input tensor has no data. Call set_data() first.".to_string(),
        ))
    }

    /// Read tensor data back to host as a Vec<T>.
    ///
    /// # Type Safety
    ///
    /// The type `T` must match the tensor's DType:
    /// - `f32` for `DType::F32`
    /// - `f64` for `DType::F64`
    /// - `i32` for `DType::I32`
    /// - etc.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor has not been realized
    /// - The type `T` doesn't match the tensor's DType
    /// - Reading from the buffer fails
    ///
    /// # Example
    ///
    /// ```ignore
    /// let x: Tensor<D1> = Tensor::input([4], DType::F32);
    /// x.set_data(&[1.0f32, 2.0, 3.0, 4.0])?;
    ///
    /// let data: Vec<f32> = x.to_vec()?;
    /// assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn to_vec<T: TensorDType + Clone>(&self) -> Result<Vec<T>, ExecutionError> {
        // Check if realized
        if !self.is_realized() {
            return Err(ExecutionError::NotRealized);
        }

        // Check dtype matches
        if T::DTYPE != self.dtype() {
            return Err(ExecutionError::DTypeMismatch(format!(
                "Expected {:?}, tensor has {:?}",
                T::DTYPE,
                self.dtype()
            )));
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
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of data that matches the tensor's shape and dtype
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
    /// let x: Tensor<D2> = Tensor::input([2, 3], DType::F32);
    /// x.set_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    /// ```
    pub fn set_data<T: TensorDType>(&self, data: &[T]) -> Result<(), ExecutionError> {
        // Check device is set
        if !has_default_device() {
            return Err(ExecutionError::NoDevice);
        }

        // Verify dtype matches
        if T::DTYPE != self.dtype() {
            return Err(ExecutionError::DTypeMismatch(format!(
                "Data type {:?} doesn't match tensor dtype {:?}",
                T::DTYPE,
                self.dtype()
            )));
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
        let buffer = allocate_buffer(self.shape(), self.dtype())?;
        let mut buffer = buffer;

        // Convert data to bytes
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
        };

        buffer
            .write_from_host(bytes)
            .map_err(|e| ExecutionError::ExecutionFailed(e.to_string()))?;

        // Store buffer
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dim::{D1, D2};

    #[test]
    fn test_is_realized() {
        let x: Tensor<D2> = Tensor::input([2, 3], DType::F32);
        assert!(!x.is_realized());
    }

    #[test]
    fn test_realize_no_device() {
        // Clear any existing device
        crate::backend::clear_default_device();

        let x: Tensor<D1> = Tensor::input([4], DType::F32);
        let result = x.realize();
        assert!(matches!(result, Err(ExecutionError::NoDevice)));
    }

    #[test]
    fn test_to_vec_not_realized() {
        let x: Tensor<D1> = Tensor::input([4], DType::F32);
        let result: Result<Vec<f32>, _> = x.to_vec();
        assert!(matches!(result, Err(ExecutionError::NotRealized)));
    }

    #[test]
    fn test_set_data_no_device() {
        // Clear any existing device
        crate::backend::clear_default_device();

        let x: Tensor<D1> = Tensor::input([4], DType::F32);
        let result = x.set_data(&[1.0f32, 2.0, 3.0, 4.0]);
        assert!(matches!(result, Err(ExecutionError::NoDevice)));
    }

    #[test]
    fn test_set_data_size_mismatch() {
        // Need a device for this test - try to set up a backend
        let _ = crate::backend::set_device_str("c");

        if has_default_device() {
            let x: Tensor<D1> = Tensor::input([4], DType::F32);
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

    #[test]
    fn test_set_data_dtype_mismatch() {
        let _ = crate::backend::set_device_str("c");

        if has_default_device() {
            let x: Tensor<D1> = Tensor::input([4], DType::F32);
            let result = x.set_data(&[1i32, 2, 3, 4]); // Wrong dtype

            // DType check happens before buffer allocation, so this should always work
            assert!(matches!(result, Err(ExecutionError::DTypeMismatch(_))));
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
        let x: Tensor<D1> = Tensor::input([4], DType::F32);
        let input_data = [1.0f32, 2.0, 3.0, 4.0];

        let result = x.set_data(&input_data);
        assert!(result.is_ok(), "set_data failed: {:?}", result.err());
        assert!(x.is_realized());

        let output: Vec<f32> = x.to_vec().expect("to_vec failed");
        assert_eq!(output, input_data.to_vec());

        // Test 2D tensor
        let y: Tensor<D2> = Tensor::input([2, 3], DType::F32);
        let input_data_2d = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        y.set_data(&input_data_2d).expect("set_data for 2D failed");
        assert!(y.is_realized());

        let output_2d: Vec<f32> = y.to_vec().expect("to_vec for 2D failed");
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
        let x_f64: Tensor<D1> = Tensor::input([4], DType::F64);
        let input_f64 = [1.0f64, 2.0, 3.0, 4.0];
        x_f64.set_data(&input_f64).expect("set_data for f64 failed");
        let output_f64: Vec<f64> = x_f64.to_vec().expect("to_vec for f64 failed");
        assert_eq!(output_f64, input_f64.to_vec());

        // Test i32
        let x_i32: Tensor<D1> = Tensor::input([4], DType::I32);
        let input_i32 = [1i32, 2, 3, 4];
        x_i32.set_data(&input_i32).expect("set_data for i32 failed");
        let output_i32: Vec<i32> = x_i32.to_vec().expect("to_vec for i32 failed");
        assert_eq!(output_i32, input_i32.to_vec());
    }
}

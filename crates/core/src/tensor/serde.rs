//! Tensor serialization and deserialization
//!
//! This module provides serialization support for Tensor types using serde.
//! Note that only the computed data is serialized; the computation graph is not preserved.
//!
//! # Examples
//!
//! ```ignore
//! use harp_core::tensor::{Tensor, DimDyn};
//! use harp_core::tensor::serde::TensorData;
//!
//! // Create and compute a tensor
//! let t = Tensor::<f32, DimDyn>::full_dyn(&[2, 3], 1.0);
//! t.realize().unwrap();
//!
//! // Serialize
//! let data = t.to_tensor_data().unwrap();
//! let json = serde_json::to_string(&data).unwrap();
//!
//! // Deserialize
//! let loaded: TensorData<f32> = serde_json::from_str(&json).unwrap();
//! let t2 = Tensor::<f32, DimDyn>::from_tensor_data(loaded);
//! ```

use ndarray::{Array, ArrayD, Dimension as NdDimension, IxDyn};
use serde::{Deserialize, Serialize};

use super::{DimDyn, Dimension, Tensor};
use crate::ast::DType;

// ============================================================================
// TensorData - Serializable representation of tensor data
// ============================================================================

/// Serializable representation of tensor data
///
/// This struct captures the essential data needed to reconstruct a tensor:
/// - shape: The dimensions of the tensor
/// - dtype: The data type (F32, F64, etc.)
/// - data: The raw element data
///
/// Note: The computation graph and autograd metadata are NOT preserved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorData<T> {
    /// Shape of the tensor
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DType,
    /// Flattened tensor data in row-major order
    pub data: Vec<T>,
}

impl<T: Clone> TensorData<T> {
    /// Create a new TensorData from components
    pub fn new(shape: Vec<usize>, dtype: DType, data: Vec<T>) -> Self {
        Self { shape, dtype, data }
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Convert to ndarray with dynamic dimensions
    pub fn to_ndarray(&self) -> ArrayD<T> {
        let shape = IxDyn(&self.shape);
        Array::from_shape_vec(shape, self.data.clone())
            .expect("Shape mismatch in TensorData::to_ndarray")
    }

    /// Convert to ndarray with static dimensions
    pub fn to_ndarray_d<D: NdDimension>(&self) -> Array<T, D> {
        let shape = D::from_dimension(&IxDyn(&self.shape)).expect("Dimension mismatch");
        Array::from_shape_vec(shape, self.data.clone()).expect("Shape mismatch")
    }
}

// ============================================================================
// Tensor -> TensorData conversion
// ============================================================================

impl<D: Dimension> Tensor<f32, D> {
    /// Convert the tensor to a serializable TensorData
    ///
    /// Returns None if the tensor has not been realized yet.
    ///
    /// # Example
    /// ```ignore
    /// let t = Tensor::<f32, Dim2>::full([2, 3], 1.0);
    /// t.realize().unwrap();
    /// let data = t.to_tensor_data().unwrap();
    /// ```
    pub fn to_tensor_data(&self) -> Option<TensorData<f32>> {
        let data = self.data()?;
        Some(TensorData {
            shape: self.shape().to_vec(),
            dtype: self.dtype().clone(),
            data,
        })
    }
}

impl<D: Dimension> Tensor<f64, D> {
    /// Convert the tensor to a serializable TensorData
    ///
    /// Returns None if the tensor has not been realized yet.
    pub fn to_tensor_data(&self) -> Option<TensorData<f64>> {
        let data = self.data()?;
        Some(TensorData {
            shape: self.shape().to_vec(),
            dtype: self.dtype().clone(),
            data,
        })
    }
}

// ============================================================================
// TensorData -> Tensor conversion (DimDyn only)
// ============================================================================

impl Tensor<f32, DimDyn> {
    /// Create a tensor from TensorData
    ///
    /// Creates an already-realized tensor with the given data.
    ///
    /// # Example
    /// ```ignore
    /// let data = TensorData::new(vec![2, 3], DType::F32, vec![1.0; 6]);
    /// let t = Tensor::<f32, DimDyn>::from_tensor_data(data);
    /// ```
    pub fn from_tensor_data(data: TensorData<f32>) -> Self {
        Self::from_data(data.data, data.shape)
    }

    /// Create a tensor from an ndarray
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::arr2;
    ///
    /// let arr = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    /// let t = Tensor::<f32, DimDyn>::from_ndarray(&arr);
    /// ```
    pub fn from_ndarray<D: NdDimension>(array: &Array<f32, D>) -> Self {
        let shape = array.shape().to_vec();
        let data: Vec<f32> = array
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| array.iter().cloned().collect());
        Self::from_data(data, shape)
    }

    /// Create a tensor from a dynamic ndarray
    ///
    /// # Example
    /// ```ignore
    /// use ndarray::ArrayD;
    ///
    /// let arr = ArrayD::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
    /// let t = Tensor::<f32, DimDyn>::from_ndarray_dyn(&arr);
    /// ```
    pub fn from_ndarray_dyn(array: &ArrayD<f32>) -> Self {
        Self::from_ndarray(array)
    }
}

impl Tensor<f64, DimDyn> {
    /// Create a tensor from TensorData
    pub fn from_tensor_data(data: TensorData<f64>) -> Self {
        Self::from_data(data.data, data.shape)
    }

    /// Create a tensor from an ndarray
    pub fn from_ndarray<D: NdDimension>(array: &Array<f64, D>) -> Self {
        let shape = array.shape().to_vec();
        let data: Vec<f64> = array
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_else(|| array.iter().cloned().collect());
        Self::from_data(data, shape)
    }

    /// Create a tensor from a dynamic ndarray
    pub fn from_ndarray_dyn(array: &ArrayD<f64>) -> Self {
        Self::from_ndarray(array)
    }
}

// ============================================================================
// File I/O convenience functions
// ============================================================================

/// Error type for tensor serialization/deserialization
#[derive(Debug)]
pub enum TensorSerdeError {
    /// Tensor has not been realized yet
    NotRealized,
    /// IO error during file operations
    Io(std::io::Error),
    /// JSON serialization/deserialization error
    Json(serde_json::Error),
}

impl std::fmt::Display for TensorSerdeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorSerdeError::NotRealized => write!(f, "Tensor has not been realized yet"),
            TensorSerdeError::Io(e) => write!(f, "IO error: {}", e),
            TensorSerdeError::Json(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl std::error::Error for TensorSerdeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TensorSerdeError::Io(e) => Some(e),
            TensorSerdeError::Json(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TensorSerdeError {
    fn from(e: std::io::Error) -> Self {
        TensorSerdeError::Io(e)
    }
}

impl From<serde_json::Error> for TensorSerdeError {
    fn from(e: serde_json::Error) -> Self {
        TensorSerdeError::Json(e)
    }
}

impl<D: Dimension> Tensor<f32, D> {
    /// Save the tensor to a JSON file
    ///
    /// # Example
    /// ```ignore
    /// let t = Tensor::<f32, Dim2>::full([2, 3], 1.0);
    /// t.realize().unwrap();
    /// t.save("tensor.json").unwrap();
    /// ```
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TensorSerdeError> {
        let data = self.to_tensor_data().ok_or(TensorSerdeError::NotRealized)?;
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &data)?;
        Ok(())
    }
}

impl Tensor<f32, DimDyn> {
    /// Load a tensor from a JSON file
    ///
    /// # Example
    /// ```ignore
    /// let t = Tensor::<f32, DimDyn>::load("tensor.json").unwrap();
    /// ```
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TensorSerdeError> {
        let file = std::fs::File::open(path)?;
        let data: TensorData<f32> = serde_json::from_reader(file)?;
        Ok(Self::from_tensor_data(data))
    }
}

impl<D: Dimension> Tensor<f64, D> {
    /// Save the tensor to a JSON file
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), TensorSerdeError> {
        let data = self.to_tensor_data().ok_or(TensorSerdeError::NotRealized)?;
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, &data)?;
        Ok(())
    }
}

impl Tensor<f64, DimDyn> {
    /// Load a tensor from a JSON file
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TensorSerdeError> {
        let file = std::fs::File::open(path)?;
        let data: TensorData<f64> = serde_json::from_reader(file)?;
        Ok(Self::from_tensor_data(data))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_roundtrip() {
        let original = TensorData::new(vec![2, 3], DType::F32, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Serialize to JSON
        let json = serde_json::to_string(&original).unwrap();

        // Deserialize back
        let loaded: TensorData<f32> = serde_json::from_str(&json).unwrap();

        assert_eq!(original.shape, loaded.shape);
        assert_eq!(original.dtype, loaded.dtype);
        assert_eq!(original.data, loaded.data);
    }

    #[test]
    fn test_tensor_data_to_ndarray() {
        let data = TensorData::new(vec![2, 3], DType::F32, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let arr = data.to_ndarray();

        assert_eq!(arr.shape(), &[2, 3]);
        assert_eq!(arr[[0, 0]], 1.0);
        assert_eq!(arr[[1, 2]], 6.0);
    }

    #[test]
    fn test_tensor_from_ndarray() {
        use ndarray::arr2;

        let arr = arr2(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let t = Tensor::<f32, DimDyn>::from_ndarray(&arr);

        assert_eq!(t.shape(), &[2, 3]);

        // Data should be available immediately (from_data creates an executed tensor)
        let data = t.data().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_tensor_save_load() {
        use std::io::Write;

        // Create a temporary file
        let mut temp_file = tempfile::NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Create TensorData and save directly (since we don't have a device set up)
        let data = TensorData::new(vec![2, 3], DType::F32, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let json = serde_json::to_string_pretty(&data).unwrap();
        temp_file.write_all(json.as_bytes()).unwrap();
        temp_file.flush().unwrap();

        // Load the tensor
        let loaded = Tensor::<f32, DimDyn>::load(&path).unwrap();

        assert_eq!(loaded.shape(), &[2, 3]);
        let loaded_data = loaded.data().unwrap();
        assert_eq!(loaded_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}

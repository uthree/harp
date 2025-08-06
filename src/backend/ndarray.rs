//! This module provides interoperability with the `ndarray` crate.
use crate::{ast::DType, backend::Buffer, tensor::shape::expr::IntoDType};
use ndarray::{Array, Dimension};
use std::ffi::c_void;

/// Implements the `Buffer` trait for `ndarray::Array`.
///
/// This allows `ndarray` arrays to be used directly as input/output buffers
/// for compiled kernels.
impl<A, D> Buffer for Array<A, D>
where
    A: IntoDType + 'static, // 'static is needed for TypeId
    D: Dimension,
{
    fn as_mut_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr() as *mut c_void
    }

    fn dtype(&self) -> DType {
        A::into_dtype()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::TryIntoNdarray;

    use ndarray::{ArrayD, arr2};

    #[test]
    fn test_ndarray_buffer_trait_f32() {
        let mut arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let buffer: &mut dyn Buffer = &mut arr;

        assert_eq!(buffer.dtype(), DType::F32);
        assert_eq!(buffer.shape(), vec![2, 2]);
        assert_eq!(buffer.size(), 4);
    }

    #[test]
    fn test_ndarray_buffer_trait_i64() {
        let mut arr = arr2(&[[1i64, 2], [3, 4]]);
        let buffer: &mut dyn Buffer = &mut arr;

        assert_eq!(buffer.dtype(), DType::I64);
        assert_eq!(buffer.shape(), vec![2, 2]);
    }

    #[test]
    fn test_buffer_to_ndarray_roundtrip() {
        let mut arr = arr2(&[[1.0f32, 2.0], [3.0, 4.0]]);
        let original_arr_dyn = arr.clone().into_dyn();

        // Convert back to ndarray using the extension trait
        let new_arr_f32 = arr.try_into_ndarray::<f32>().unwrap();
        assert_eq!(new_arr_f32.shape(), &[2, 2]);
        assert_eq!(new_arr_f32, original_arr_dyn);

        // Try converting to a wrong type
        let new_arr_i64 = arr.try_into_ndarray::<i64>();
        assert!(new_arr_i64.is_none());
    }

    #[test]
    fn test_empty_buffer_to_ndarray() {
        let mut arr = ArrayD::<f32>::from_shape_vec(vec![2, 0, 2], vec![]).unwrap();

        assert_eq!(arr.size(), 0);
        let new_arr = arr.try_into_ndarray::<f32>().unwrap();
        assert_eq!(new_arr.shape(), &[2, 0, 2]);
        assert_eq!(new_arr.len(), 0);
    }
}

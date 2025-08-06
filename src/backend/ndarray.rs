//! This module provides interoperability with the `ndarray` crate.
use crate::{ast::DType, backend::Buffer, tensor::shape::expr::IntoDType};
use ndarray::{Array, Dimension};

/// Implements the `Buffer` trait for `ndarray::Array`.
///
/// This allows `ndarray` arrays to be used directly as input/output buffers
/// for compiled kernels.
impl<A, D> Buffer for Array<A, D>
where
    A: IntoDType + 'static, // 'static is needed for TypeId
    D: Dimension,
{
    fn as_mut_bytes(&mut self) -> &mut [u8] {
        let slice = self.as_slice_mut().unwrap();
        unsafe {
            std::slice::from_raw_parts_mut(
                slice.as_mut_ptr() as *mut u8,
                std::mem::size_of_val(slice),
            )
        }
    }

    fn dtype(&self) -> DType {
        A::into_dtype()
    }

    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
}

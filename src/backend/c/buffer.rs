use crate::ast::DType;
use crate::backend::Buffer;
use std::ffi::c_void;

#[derive(Debug)]
pub struct CBuffer {
    /// Raw pointer to the allocated memory.
    pub ptr: *mut c_void,
    /// The shape of the buffer.
    pub shape: Vec<usize>,
    /// The data type of the elements.
    pub dtype: DType,
}

impl CBuffer {
    /// Creates a new buffer from a slice of data.
    pub fn from_slice<T>(data: &[T], shape: &[usize], dtype: DType) -> Self {
        let buffer = Self::allocate(dtype.clone(), shape.to_vec());
        let numel = shape.iter().product::<usize>();

        // 要素数と型サイズの整合性をチェック
        if data.len() != numel {
            panic!(
                "Data length {} doesn't match shape elements {}",
                data.len(),
                numel
            );
        }

        // 型サイズの整合性をチェック
        let expected_element_size = dtype.size();
        let actual_element_size = std::mem::size_of::<T>();
        if expected_element_size != actual_element_size {
            panic!(
                "Type size mismatch: expected {} bytes, got {} bytes",
                expected_element_size, actual_element_size
            );
        }

        let total_size = numel * expected_element_size;
        unsafe {
            // より安全なメモリコピーのため、ソースとデスティネーションの境界チェック
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.ptr as *mut u8,
                total_size,
            );
        }
        buffer
    }

    /// Copies the buffer data to a new `Vec`.
    pub fn to_vec<T>(&self) -> Vec<T> {
        let numel = self.shape.iter().product::<usize>();

        // 型サイズの整合性をチェック
        let expected_element_size = self.dtype.size();
        let actual_element_size = std::mem::size_of::<T>();
        if expected_element_size != actual_element_size {
            panic!(
                "Type size mismatch: expected {} bytes, got {} bytes",
                expected_element_size, actual_element_size
            );
        }

        let mut vec = Vec::with_capacity(numel);
        let total_size = numel * expected_element_size;
        unsafe {
            vec.set_len(numel);
            // より安全なメモリコピー
            std::ptr::copy_nonoverlapping(
                self.ptr as *const u8,
                vec.as_mut_ptr() as *mut u8,
                total_size,
            );
        }
        vec
    }
}

impl Buffer for CBuffer {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn allocate(dtype: DType, shape: Vec<usize>) -> Self {
        let numel = shape.iter().product::<usize>();
        let element_size = dtype.size();
        let size = numel * element_size;

        let ptr = if size == 0 {
            // ゼロサイズの場合は非nullポインタを返す（標準的な動作）
            std::ptr::NonNull::dangling().as_ptr() as *mut libc::c_void
        } else {
            let ptr = unsafe { libc::malloc(size) };
            if ptr.is_null() {
                panic!("Failed to allocate {} bytes for CBuffer", size);
            }
            ptr
        };

        CBuffer { ptr, shape, dtype }
    }
}

impl Drop for CBuffer {
    fn drop(&mut self) {
        unsafe {
            // ゼロサイズの場合は dangling ポインタなので free しない
            let numel = self.shape.iter().product::<usize>();
            let size = numel * self.dtype.size();
            if size > 0 {
                libc::free(self.ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_allocation() {
        let shape = vec![2, 3];
        let dtype = DType::F32;
        let buffer = CBuffer::allocate(dtype, shape);
        assert!(!buffer.ptr.is_null());
        assert_eq!(buffer.shape, vec![2, 3]);
        assert_eq!(buffer.dtype, DType::F32);
    }

    #[test]
    fn test_buffer_from_slice_and_to_vec() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let dtype = DType::F32;
        let buffer = CBuffer::from_slice(&data, &shape, dtype);
        let result_vec = buffer.to_vec::<f32>();
        assert_eq!(data, result_vec);
    }

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::Usize.size(), std::mem::size_of::<usize>());
        assert_eq!(DType::Isize.size(), std::mem::size_of::<isize>());
        assert_eq!(
            DType::Ptr(Box::new(DType::Void)).size(),
            std::mem::size_of::<*const ()>()
        );
    }
}

impl DType {
    /// Returns the size of the data type in bytes.
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::Isize => std::mem::size_of::<isize>(),
            DType::Usize => std::mem::size_of::<usize>(),
            // Pointers are assumed to be 64-bit for now.
            DType::Ptr(_) => std::mem::size_of::<*const c_void>(),
            _ => unimplemented!("Size for dtype {:?} is not implemented", self),
        }
    }
}

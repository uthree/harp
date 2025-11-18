use crate::ast::DType;
use crate::backend::Buffer;

/// Metalバックエンド用のホストメモリバッファ
///
/// C/OpenCLバックエンドと同様に、ホストメモリを使用する。
/// Metal APIを使った実行時に、このデータがGPUに転送される。
pub struct MetalBuffer {
    data: Vec<u8>,
    shape: Vec<usize>,
    element_size: usize,
}

impl MetalBuffer {
    /// 新しいバッファを作成
    pub fn new(shape: Vec<usize>, element_size: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let byte_size = total_elements * element_size;
        let data = vec![0u8; byte_size];

        Self {
            data,
            shape,
            element_size,
        }
    }

    /// f32データから直接バッファを作成
    pub fn from_f32_vec(data: Vec<f32>) -> Self {
        let shape = vec![data.len()];
        let element_size = std::mem::size_of::<f32>();
        let byte_size = data.len() * element_size;

        let mut buffer = Self {
            data: vec![0u8; byte_size],
            shape,
            element_size,
        };

        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                buffer.data.as_mut_ptr(),
                byte_size,
            );
        }

        buffer
    }

    /// データポインタを取得（可変）
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// データポインタを取得（不変）
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// f32スライスとして取得
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.element_size != std::mem::size_of::<f32>() {
            return None;
        }
        unsafe {
            Some(std::slice::from_raw_parts(
                self.data.as_ptr() as *const f32,
                self.data.len() / self.element_size,
            ))
        }
    }

    /// f32スライスとして取得（可変）
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.element_size != std::mem::size_of::<f32>() {
            return None;
        }
        unsafe {
            Some(std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut f32,
                self.data.len() / self.element_size,
            ))
        }
    }
}

impl Buffer for MetalBuffer {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn dtype(&self) -> DType {
        // element_sizeから型を推測
        match self.element_size {
            4 => DType::F32,
            8 => DType::Int, // i64相当
            _ => DType::F32,
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        self.data.clone()
    }

    fn from_bytes(&mut self, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() != self.data.len() {
            return Err(format!(
                "Byte length mismatch: expected {}, got {}",
                self.data.len(),
                bytes.len()
            ));
        }

        self.data.copy_from_slice(bytes);
        Ok(())
    }

    fn byte_len(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_creation() {
        let buffer = MetalBuffer::new(vec![10, 20], 4);
        assert_eq!(buffer.shape(), vec![10, 20]);
        assert_eq!(buffer.byte_len(), 10 * 20 * 4);
    }

    #[test]
    fn test_buffer_from_f32() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let buffer = MetalBuffer::from_f32_vec(data.clone());

        assert_eq!(buffer.shape(), vec![10]);
        assert_eq!(buffer.element_size, std::mem::size_of::<f32>());

        let read_data = buffer.as_f32_slice().unwrap();
        assert_eq!(read_data, data.as_slice());
    }

    #[test]
    fn test_buffer_read_write() {
        let mut buffer = MetalBuffer::new(vec![10], std::mem::size_of::<f32>());

        // データを書き込み
        let write_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let slice = buffer.as_f32_slice_mut().unwrap();
        slice.copy_from_slice(&write_data);

        // データを読み出し
        let read_data = buffer.as_f32_slice().unwrap();

        // 確認
        assert_eq!(write_data.as_slice(), read_data);
    }
}

use crate::backend::Buffer;
use log::{debug, trace};
use metal::{Buffer as MTLBuffer, Device, MTLResourceOptions};

/// Metal デバイスバッファのラッパー
pub struct MetalBuffer {
    buffer: MTLBuffer,
    shape: Vec<usize>,
    element_size: usize,
}

impl MetalBuffer {
    /// 新しいバッファを作成
    pub fn new(device: &Device, shape: Vec<usize>, element_size: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let byte_size = total_elements * element_size;

        debug!(
            "Creating Metal buffer: shape={:?}, element_size={}, total_bytes={}",
            shape, element_size, byte_size
        );

        let buffer = device.new_buffer(byte_size as u64, MTLResourceOptions::StorageModeShared);

        Self {
            buffer,
            shape,
            element_size,
        }
    }

    /// 既存の MTLBuffer からラップ
    pub fn from_buffer(buffer: MTLBuffer, shape: Vec<usize>, element_size: usize) -> Self {
        Self {
            buffer,
            shape,
            element_size,
        }
    }

    /// 内部の MTLBuffer への参照を取得
    pub fn inner(&self) -> &MTLBuffer {
        &self.buffer
    }

    /// バッファのバイトサイズを取得
    pub fn byte_size(&self) -> usize {
        self.buffer.length() as usize
    }

    /// 要素のサイズを取得（バイト単位）
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// データを CPU から GPU へコピー
    pub fn write_data<T: Copy>(&mut self, data: &[T]) {
        trace!(
            "Writing {} elements to Metal buffer (shape={:?})",
            data.len(),
            self.shape
        );
        let ptr = self.buffer.contents() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        trace!("Write completed");
    }

    /// データを GPU から CPU へコピー
    pub fn read_data<T: Copy>(&self, data: &mut [T]) {
        trace!(
            "Reading {} elements from Metal buffer (shape={:?})",
            data.len(),
            self.shape
        );
        let ptr = self.buffer.contents() as *const T;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), data.len());
        }
        trace!("Read completed");
    }
}

impl Buffer for MetalBuffer {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::MetalCompiler;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_buffer_creation() {
        if let Some(compiler) = MetalCompiler::with_default_device() {
            let buffer = compiler.create_buffer(vec![10, 20], 4);
            assert_eq!(buffer.shape(), vec![10, 20]);
            assert_eq!(buffer.byte_size(), 10 * 20 * 4);
        }
    }

    #[test]
    #[serial]
    fn test_buffer_read_write() {
        if let Some(compiler) = MetalCompiler::with_default_device() {
            let mut buffer = compiler.create_buffer(vec![10], 4);

            // データを書き込み
            let write_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
            buffer.write_data(&write_data);

            // データを読み出し
            let mut read_data = vec![0.0f32; 10];
            buffer.read_data(&mut read_data);

            // 確認
            assert_eq!(write_data, read_data);
        }
    }
}

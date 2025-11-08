use crate::backend::Buffer;

/// C言語/OpenMPバックエンド用のバッファ
///
/// CPUメモリ上のデータバッファを表します。
#[derive(Debug, Clone)]
pub struct CBuffer {
    data: Vec<u8>,
    shape: Vec<usize>,
    element_size: usize,
}

impl CBuffer {
    /// 新しいCBufferを作成
    pub fn new(shape: Vec<usize>, element_size: usize) -> Self {
        let total_elements: usize = shape.iter().product();
        let byte_size = total_elements * element_size;
        Self {
            data: vec![0; byte_size],
            shape,
            element_size,
        }
    }

    /// データへの可変参照を取得
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// データへの参照を取得
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// データのポインタを取得（unsafe）
    pub fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// データの可変ポインタを取得（unsafe）
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data.as_mut_ptr()
    }

    /// 要素サイズを取得
    pub fn element_size(&self) -> usize {
        self.element_size
    }

    /// 総バイト数を取得
    pub fn byte_len(&self) -> usize {
        self.data.len()
    }

    /// f32スライスとして取得（要素サイズが4バイトの場合）
    pub fn as_f32_slice(&self) -> Option<&[f32]> {
        if self.element_size == 4 {
            let len = self.data.len() / 4;
            Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const f32, len) })
        } else {
            None
        }
    }

    /// f32スライスとして可変取得（要素サイズが4バイトの場合）
    pub fn as_f32_slice_mut(&mut self) -> Option<&mut [f32]> {
        if self.element_size == 4 {
            let len = self.data.len() / 4;
            Some(unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut f32, len) })
        } else {
            None
        }
    }

    /// i32スライスとして取得（要素サイズが4バイトの場合）
    pub fn as_i32_slice(&self) -> Option<&[i32]> {
        if self.element_size == 4 {
            let len = self.data.len() / 4;
            Some(unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const i32, len) })
        } else {
            None
        }
    }

    /// i32スライスとして可変取得（要素サイズが4バイトの場合）
    pub fn as_i32_slice_mut(&mut self) -> Option<&mut [i32]> {
        if self.element_size == 4 {
            let len = self.data.len() / 4;
            Some(unsafe { std::slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut i32, len) })
        } else {
            None
        }
    }
}

impl Buffer for CBuffer {
    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cbuffer_creation() {
        let buffer = CBuffer::new(vec![10, 20], 4);
        assert_eq!(buffer.shape(), vec![10, 20]);
        assert_eq!(buffer.element_size(), 4);
        assert_eq!(buffer.byte_len(), 10 * 20 * 4);
    }

    #[test]
    fn test_cbuffer_f32_slice() {
        let mut buffer = CBuffer::new(vec![10], 4);

        // f32スライスとして書き込み
        if let Some(slice) = buffer.as_f32_slice_mut() {
            for (i, val) in slice.iter_mut().enumerate() {
                *val = i as f32;
            }
        }

        // f32スライスとして読み込み
        if let Some(slice) = buffer.as_f32_slice() {
            assert_eq!(slice[0], 0.0);
            assert_eq!(slice[5], 5.0);
            assert_eq!(slice[9], 9.0);
        } else {
            panic!("Failed to get f32 slice");
        }
    }

    #[test]
    fn test_cbuffer_wrong_element_size() {
        let buffer = CBuffer::new(vec![10], 2);
        assert!(buffer.as_f32_slice().is_none());
        assert!(buffer.as_i32_slice().is_none());
    }
}

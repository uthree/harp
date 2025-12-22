//! バッファ型
//!
//! バックエンド非依存のバッファ抽象化を提供します。

use super::ArrayError;

/// バックエンドバリアント生成マクロ
///
/// 新しいバックエンドを追加する際は、このマクロに1行追加するだけでOK。
macro_rules! define_buffer {
    ($($feature:literal => $variant:ident($type:ty)),* $(,)?) => {
        /// バッファ（デバイス非依存）
        #[derive(Clone)]
        pub enum Buffer {
            $(
                #[cfg(feature = $feature)]
                $variant($type),
            )*
        }

        impl Buffer {
            /// バッファの形状を取得
            pub fn shape(&self) -> &[usize] {
                match self {
                    $(
                        #[cfg(feature = $feature)]
                        Buffer::$variant(buf) => {
                            use harp_core::backend::Buffer;
                            buf.shape()
                        }
                    )*
                    #[allow(unreachable_patterns)]
                    _ => &[],
                }
            }

            /// バッファからデータを読み出し
            pub fn read_vec<T: Clone + 'static>(&self) -> Result<Vec<T>, ArrayError> {
                match self {
                    $(
                        #[cfg(feature = $feature)]
                        Buffer::$variant(buf) => {
                            use harp_core::backend::Buffer;
                            buf.read_vec()
                                .map_err(|e| ArrayError::Execution(e.to_string()))
                        }
                    )*
                    #[allow(unreachable_patterns)]
                    _ => Err(ArrayError::NoBackend),
                }
            }
        }
    };
}

define_buffer! {
    "opencl" => OpenCL(harp_backend_opencl::OpenCLBuffer),
    "metal" => Metal(harp_backend_metal::MetalBuffer),
}

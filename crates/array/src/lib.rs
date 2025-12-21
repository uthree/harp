//! Harp Array: ndarray/PyTorchライクなAPIで配列計算を行うクレート
//!
//! # 特徴
//!
//! - **動的バックエンド**: 実行時にデバイスを選択可能
//! - **型付け次元**: `Array1`, `Array2` など、次元数をコンパイル時に検証
//! - **動的次元**: `ArrayD` で実行時に次元数が決まる配列もサポート
//! - **デバイス転送**: `.to(Device::Metal)` でデバイス間転送
//!
//! # Example
//!
//! ```ignore
//! use harp_array::prelude::*;
//!
//! // 配列の作成
//! let a: Array2<f32> = Array2::zeros([100, 100]);
//! let b: Array2<f32> = Array2::ones([100, 100]);
//!
//! // デバイス転送
//! if Device::Metal.is_available() {
//!     let metal_arr = a.to(Device::Metal)?;
//! }
//! ```

pub mod device;
pub mod dim;
pub mod dyn_backend;
pub mod execution;
pub mod generators;

// Re-exports
pub use device::Device;
pub use dim::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
pub use dyn_backend::{
    Array, Array0, Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayElement, ArrayError,
};
pub use generators::IntoShape;

/// Prelude module - 主要な型をまとめてインポート
pub mod prelude {
    // 配列型
    pub use crate::dyn_backend::{
        Array, Array0, Array1, Array2, Array3, Array4, Array5, Array6, ArrayD, ArrayElement,
        ArrayError,
    };

    // デバイス
    pub use crate::device::Device;

    // 次元型
    pub use crate::dim::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
    pub use crate::dim::{DimensionMismatch, IntoDimensionality, IntoDyn};
    pub use crate::generators::IntoShape;
}

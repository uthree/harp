//! Harp Array: ndarray/PyTorchライクなAPIで配列計算を行うクレート
//!
//! このクレートは遅延評価による計算グラフの構築と、
//! キャッシュ機構による効率的なカーネル再利用を提供します。
//!
//! # 特徴
//!
//! - **遅延評価**: 演算はグラフとして構築され、必要になったときに実行
//! - **キャッシュ**: 同じ計算グラフは再コンパイルせずに再利用
//! - **型付け次元**: `Array1`, `Array2` など、次元数をコンパイル時に検証
//! - **動的次元**: `ArrayD` で実行時に次元数が決まる配列もサポート
//!
//! # Example
//!
//! ```ignore
//! use harp_array::prelude::*;
//!
//! // 配列の作成（遅延評価：この時点では計算しない）
//! let a: Array2<f32> = zeros([100, 100]);
//! let b: Array2<f32> = ones([100, 100]);
//!
//! // 演算（グラフ構築のみ）
//! let c = &a + &b;
//! let d = &c * 2.0f32;
//!
//! // データ取得時に初めて計算が実行される
//! let data: Vec<f32> = d.to_vec()?;
//! ```

pub mod array;
pub mod cache;
pub mod context;
pub mod dim;
pub mod generators;
pub mod ops;

// Re-exports
pub use array::{Array, ArrayElement, ArrayError, ArrayState};
pub use cache::{GraphSignature, KernelCache};
pub use context::{ContextError, ExecutionConfig, ExecutionContext};
pub use dim::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
pub use generators::{
    IntoShape, arange, arange_f32, full_f32, full_i32, ones, ones_i32, ones_like, rand, zeros,
    zeros_i32, zeros_like,
};

/// Prelude module - 主要な型をまとめてインポート
pub mod prelude {
    pub use crate::dim::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
    pub use crate::dim::{DimensionMismatch, IntoDimensionality, IntoDyn};
}

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
//! - **グローバルコンテキスト**: バックエンド単位でコンテキストを共有
//!
//! # Example
//!
//! ```ignore
//! use harp_array::prelude::*;
//!
//! // バックエンド固有の型エイリアスを使用
//! type Array2F = Array<f32, Dim2, MyBackend>;
//!
//! // 配列の作成（遅延評価：この時点では計算しない）
//! let a: Array2F = Array::zeros([100, 100]);
//! let b: Array2F = Array::ones([100, 100]);
//!
//! // 演算（グラフ構築のみ）
//! let c = &a + &b;
//! let d = &c * 2.0f32;
//!
//! // データ取得時に初めて計算が実行される
//! let data: Vec<f32> = d.to_vec()?;
//! ```

pub mod array;
pub mod backend;
pub mod cache;
pub mod context;
pub mod dim;
pub mod generators;
pub mod ops;

// Re-exports
pub use array::{Array, ArrayElement, ArrayError, ArrayState};
pub use backend::Backend;
pub use cache::{GraphSignature, KernelCache};
pub use context::{ContextError, ExecutionConfig, ExecutionContext};
pub use dim::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
pub use generators::IntoShape;

/// Prelude module - 主要な型をまとめてインポート
pub mod prelude {
    pub use crate::array::{Array, ArrayElement, ArrayError};
    pub use crate::backend::Backend;
    pub use crate::dim::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension};
    pub use crate::dim::{DimensionMismatch, IntoDimensionality, IntoDyn};
    pub use crate::generators::IntoShape;
}

//! Harp: High-level Array Processor
//!
//! Harpは計算グラフを構築し、様々なバックエンド（OpenCL、Metal等）で実行するライブラリです。
//!
//! # 基本的な使い方
//!
//! Tensorを使用した推奨される方法:
//!
//! ```ignore
//! use harp::prelude::*;
//! use harp::backend::set_default_device;
//!
//! // デバイスの設定
//! // set_default_device(device, DeviceKind::Metal);
//!
//! // テンソルの作成
//! let a = Tensor::<Dim2>::full([10, 20], 1.0);
//! let b = Tensor::<Dim2>::full([10, 20], 2.0);
//!
//! // 遅延評価される演算
//! let result = &a + &b;
//!
//! // 実行（realize()で計算を実行）
//! let computed = result.realize().unwrap();
//! ```
//!
//! # バックエンド
//!
//! バックエンドはfeature flagで有効化します:
//! - `opencl`: OpenCLバックエンド（`harp::backend::opencl`）
//! - `metal`: Metalバックエンド（`harp::backend::metal`、macOSのみ）
//!
//! ```toml
//! [dependencies]
//! harp = { version = "0.1", features = ["opencl"] }
//! ```

// Core modules
pub mod ast;
pub mod backend;
pub mod opt;
pub mod renderer;
pub mod tensor;

// Re-export types
pub use ast::DType;

// Re-export renderer traits
pub use renderer::Renderer;

// Re-export backend traits
pub use backend::{Buffer, Compiler, Device, Kernel, KernelConfig, Pipeline};

/// Prelude module with commonly used types and traits
///
/// このモジュールをインポートすることで、Harpを使う上で必要な
/// 主要な型やトレイトを一括でインポートできます。
///
/// # 推奨: Tensor API
///
/// Tensorはharpの主要なAPIです。遅延評価、自動微分、演算融合をサポートします。
///
/// # Example
///
/// ```ignore
/// use harp::prelude::*;
/// ```
pub mod prelude {
    // Tensor types (recommended API)
    pub use crate::tensor::{
        Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension, Tensor,
    };

    // Data types
    pub use crate::ast::DType;

    // Backend traits
    pub use crate::backend::{
        Buffer, BufferSignature, Compiler, Device, Kernel, KernelSignature, Pipeline, Renderer,
    };

    // Shape expressions (for advanced tensor operations)
    pub use crate::tensor::shape::{Expr, View};
}

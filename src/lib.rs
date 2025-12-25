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
//! # Graph API（上級者向け）
//!
//! 低レベルのGraph APIも利用可能です。計算グラフを直接操作したい場合に使用します。
//! `harp::graph`モジュールからGraph, GraphNode等をインポートしてください。
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
pub mod core;
pub mod graph;
pub mod lowerer;
pub mod opt;
pub mod renderer;
pub mod tensor;

// Re-export commonly used types from graph module
// Note: GraphNode is available via `harp::graph::GraphNode` for advanced usage
pub use graph::{DType, Graph, ReduceStrategy};

// Re-export renderer traits
pub use renderer::Renderer;

// Re-export backend traits
pub use backend::{Buffer, Compiler, Device, Kernel, KernelConfig, Pipeline};

// Re-export lowerer
pub use lowerer::{create_lowering_optimizer, create_signature, create_simple_lowering_optimizer};

/// Prelude module with commonly used types and traits
///
/// このモジュールをインポートすることで、Harpを使う上で必要な
/// 主要な型やトレイトを一括でインポートできます。
///
/// # 推奨: Tensor API
///
/// Tensorはharpの主要なAPIです。遅延評価、自動微分、演算融合をサポートします。
///
/// # 上級者向け: Graph API
///
/// 低レベルのGraph APIが必要な場合は、`harp::graph`モジュールを直接使用してください。
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
    pub use crate::graph::DType;

    // Graph types (for advanced usage - prefer Tensor API for most use cases)
    // GraphNode is intentionally excluded; use harp::graph::GraphNode if needed
    pub use crate::graph::{Graph, ReduceStrategy};

    // Backend traits
    pub use crate::backend::{
        Buffer, BufferSignature, Compiler, Device, Kernel, KernelSignature, Pipeline, Renderer,
    };

    // Shape expressions (for advanced tensor operations)
    pub use crate::graph::shape::{Expr, View};
}

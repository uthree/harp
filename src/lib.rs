//! Harp: High-level Array Processor
//!
//! Harpは計算グラフを構築し、様々なバックエンド（OpenCL、Metal等）で実行するライブラリです。
//!
//! # 基本的な使い方
//!
//! ```no_run
//! use harp::prelude::*;
//!
//! // グラフの作成
//! let mut graph = Graph::new();
//!
//! // 入力ノードの作成
//! let a = graph.input("a", DType::F32, vec![10, 20]);
//! let b = graph.input("b", DType::F32, vec![10, 20]);
//!
//! // 演算
//! let result = a + b;
//!
//! // 出力ノードの登録
//! graph.output("result", result);
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
pub mod graph;
pub mod lowerer;
pub mod opt;
pub mod renderer;
pub mod tensor;

// Re-export commonly used types from graph module
pub use graph::{CumulativeStrategy, DType, Graph, GraphNode, ReduceStrategy};

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
/// # Example
///
/// ```
/// use harp::prelude::*;
/// ```
pub mod prelude {
    // Graph types
    pub use crate::graph::{
        CumulativeStrategy, DType, Graph, GraphNode, GraphOp, ReduceOp, ReduceStrategy,
    };

    // Graph operations (helper functions)
    pub use crate::graph::ops::{max, recip, reduce, reduce_max, reduce_mul, reduce_sum};

    // Backend traits
    pub use crate::backend::{
        Buffer, BufferSignature, Compiler, Device, Kernel, KernelSignature, Pipeline, Renderer,
    };

    // Lowerer
    pub use crate::lowerer::{
        create_lowering_optimizer, create_signature, create_simple_lowering_optimizer,
    };

    // Shape expressions
    pub use crate::graph::shape::{Expr, View};

    // AST types (for advanced usage)
    pub use crate::ast::{AstNode, DType as AstDType, Literal};

    // Tensor types
    pub use crate::tensor::{
        Dim, Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn, Dimension, Tensor,
    };
}

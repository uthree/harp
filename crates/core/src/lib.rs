//! Harp Core: A high-performance tensor computation library
//!
//! Harpは計算グラフを構築し、様々なバックエンド（OpenCL、Metal等）で実行するライブラリです。
//!
//! # 基本的な使い方
//!
//! ```no_run
//! use harp_core::prelude::*;
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

// Core modules
pub mod ast;
pub mod backend;
pub mod graph;
pub mod lowerer;
pub mod opt;

// Re-export commonly used types from graph module
pub use graph::{CumulativeStrategy, DType, Graph, GraphNode, ReduceStrategy};

// Re-export backend traits
pub use backend::Renderer;

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
/// use harp_core::prelude::*;
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
}

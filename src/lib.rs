//! Harp: A high-performance tensor computation library
//!
//! Harpは計算グラフを構築し、様々なバックエンド（Metal等）で実行するライブラリです。
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
//! let a = graph.input("a")
//!     .with_dtype(DType::F32)
//!     .with_shape(vec![10, 20])
//!     .build();
//!
//! let b = graph.input("b")
//!     .with_dtype(DType::F32)
//!     .with_shape(vec![10, 20])
//!     .build();
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
pub use graph::{CumulativeStrategy, DType, ElementwiseStrategy, Graph, GraphNode, ReduceStrategy};

// Re-export backend traits
pub use backend::{Buffer, Compiler, Kernel, Renderer};

// Re-export lowerer
pub use lowerer::Lowerer;

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
        CumulativeStrategy, DType, ElementwiseStrategy, Graph, GraphNode, GraphOp, ReduceOp,
        ReduceStrategy,
    };

    // Graph operations (helper functions)
    pub use crate::graph::ops::{
        fused_elementwise, fused_elementwise_reduce, fused_reduce, max, recip, reduce, reduce_max,
        reduce_mul, reduce_sum,
    };

    // Fused operation types
    pub use crate::graph::ops::{FusedElementwiseOp, FusedInput};

    // Backend traits
    pub use crate::backend::{Buffer, Compiler, Kernel, KernelSignature, Query, QueryBuilder};

    // Lowerer
    pub use crate::lowerer::Lowerer;

    // Shape expressions
    pub use crate::graph::shape::{Expr, View};

    // AST types (for advanced usage)
    pub use crate::ast::{AstNode, DType as AstDType, Function, Literal};
}

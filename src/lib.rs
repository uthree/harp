//! # harp: A Tensor Computation Library with JIT Compilation
//!
//! `harp` is a library for performing tensor (multi-dimensional array) computations,
//! inspired by projects like tinygrad. It constructs a computation graph from tensor
//! operations, allowing for optimizations and Just-In-Time (JIT) compilation to
//! high-performance C code via a backend like Clang.
//!
//! ## Core Concepts
//!
//! - **Tensor:** The primary user-facing data structure, representing a multi-dimensional
//!   array. All operations on Tensors are lazily evaluated.
//! - **Computation Graph:** Tensor operations build a directed acyclic graph (DAG) of
//!   operations (`UOp`). This graph is not executed immediately.
//! - **Lazy Evaluation:** Computations are only performed when the result is explicitly
//!   requested by calling the `.realize()` method on a `Tensor`.
//! - **Backend:** An execution engine responsible for taking the computation graph,
//!   optimizing it, compiling it to a native kernel (e.g., C via Clang), and
//!   executing it.
//!
//! ## Example
//!
//! Here is a simple example of adding two tensors.
//!
//! ```rust
//! use harp::prelude::*;
//!
//! // 1. Get a backend instance for the current thread.
//! let backend = backend("clang");
//!
//! // 2. Create two source tensors from data.
//! let a = Tensor::from_vec(vec![1.0f32; 10], &[10]);
//! let b = Tensor::from_vec(vec![2.0f32; 10], &[10]);
//!
//! // 3. Perform an operation. This builds the graph but doesn't compute anything.
//! let c = &a + &b;
//!
//! // 4. Realize the result. This triggers the entire pipeline:
//! //    Graph -> Optimization -> Lowering -> Code Generation -> Compilation -> Execution
//! let result_tensor = c.realize();
//!
//! // 5. Copy the result back to the host and verify.
//! let result_vec = c.to_vec();
//! assert_eq!(result_vec, vec![3.0f32; 10]);
//! ```

pub mod backends;
pub mod autotuner;
pub mod context;
pub mod dot;
pub mod dtype;
pub mod linearizer;
pub mod lowerizer;
pub mod optimizer;
pub mod pattern;
pub mod prelude;
pub mod shapetracker;
pub mod tensor;
pub mod uop;
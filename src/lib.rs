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
//! use ndarray::ArrayD;
//!
//! // 1. Get a backend instance for the current thread.
//! let backend = backend("clang");
//!
//! // 2. Create two source tensors from ndarray arrays.
//! let arr_a: ArrayD<f32> = ArrayD::from_elem(vec![10], 1.0f32);
//! let arr_b: ArrayD<f32> = ArrayD::from_elem(vec![10], 2.0f32);
//! let a: Tensor = arr_a.clone().into();
//! let b: Tensor = arr_b.clone().into();
//!
//! // 3. Perform an operation. This builds the graph but doesn't compute anything.
//! let c = &a + &b;
//!
//! // 4. Realize the result. This triggers the entire pipeline:
//! //    Graph -> Optimization -> Lowering -> Code Generation -> Compilation -> Execution
//! let result_buffer = c.realize();
//!
//! // 5. Copy the result back to an ndarray and verify.
//! let result_arr: ArrayD<f32> = c.into();
//! assert_eq!(result_arr, &arr_a + &arr_b);
//! ```

pub mod backends;
pub mod context;
pub mod dot;
pub mod dtype;
pub mod lowerizer;
pub mod optimization;
pub mod prelude;
pub mod shapetracker;
pub mod tensor;
pub mod uop;

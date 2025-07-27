# harp: A Tensor Computation Library with JIT Compilation

WORK IN PROGRESS  

`harp` is a library for performing tensor (multi-dimensional array) computations,
inspired by projects like tinygrad. It constructs a computation graph from tensor
operations, allowing for optimizations and Just-In-Time (JIT) compilation to
high-performance C code via a backend like Clang.

## Core Concepts

- **Tensor:** The primary user-facing data structure, representing a multi-dimensional
  array. All operations on Tensors are lazily evaluated.
- **Computation Graph:** Tensor operations build a directed acyclic graph (DAG) of
  operations. This graph is not executed immediately.
- **Lazy Evaluation:** Computations are only performed when the result is explicitly
  requested by calling the `.realize()` method on a `Tensor`.
- **Backend:** An execution engine responsible for taking the computation graph,
  optimizing it, compiling it to a native kernel (e.g., C via Clang), and
  executing it.

## Usage

Here is a simple example of adding two tensors.

```rust
use harp::prelude::*;
use std::rc::Rc;

fn main() {
    // 1. Initialize a backend. This requires Clang to be installed.
    let backend = Rc::new(ClangBackend::new().expect("Clang backend failed to initialize."));

    // 2. Create two source tensors. Data is not allocated yet.
    let a = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(vec![10]),
        DType::F32,
        backend.clone(),
    );
    let b = Tensor::new(
        TensorOp::Load,
        vec![],
        ShapeTracker::new(vec![10]),
        DType::F32,
        backend.clone(),
    );

    // 3. Perform an operation. This builds the graph but doesn't compute anything.
    let c = &a + &b;

    // 4. Realize the result. This triggers the entire pipeline:
    //    Graph -> Optimization -> Lowering -> Code Generation -> Compilation -> Execution
    let result_buffer = c.realize();

    // The result_buffer now holds a handle to the memory containing the result of the addition.
    println!("Successfully realized tensor c!");
}
```

## License

This project is licensed under the MIT License.

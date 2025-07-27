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
use ndarray::ArrayD;

fn main() {
    // 1. Create two source tensors.
    //    You can use `ndarray::ArrayD` and the `.into()` trait.
    let a: Tensor = ArrayD::from_elem(vec![10], 1.0f32).into();
    let b: Tensor = ArrayD::from_elem(vec![10], 2.0f32).into();

    // 2. Perform an operation. This builds the graph but doesn't compute anything.
    let c = &a + &b;

    // 3. Convert the result back to an ndarray to inspect it.
    //    This implicitly calls `.realize()` to execute the computation.
    let result_array: ArrayD<f32> = c.into();

    println!("Successfully computed tensor c!");
    println!("Result: {:?}", result_array);
    // Expected output: array([3., 3., 3., 3., 3., 3., 3., 3., 3., 3.], dim: [10])
}
```

## License

This project is dual licensed under Apache 2.0 and MIT, so you can use it either way if you need to.

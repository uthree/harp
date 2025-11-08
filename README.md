[![Rust CI](https://github.com/uthree/harp/actions/workflows/rust.yml/badge.svg)](https://github.com/uthree/harp/actions/workflows/rust.yml)
[![Rust](https://img.shields.io/badge/rust-stable-blue.svg)](https://www.rust-lang.org/)

# harp: A Tensor Computation Library with JIT Compilation
**WORK IN PROGRESS**

"Harp" is a transpiler for multidimensional array computation, deep learning, and computer science, inspired by [tinygrad](https://github.com/tinygrad/tinygrad) and [luminal](https://github.com/luminal-ai/luminal).
It automatically generates optimal C kernels for CPUs or GPUs from a computational graph of a multidimensional array.

## Architecture
![architecture](assets/images/harp_architecture_overview.png)
- Tensor / Autograd Frontend: A frontend that makes it easy to handle tensor calculations, automatic differentiation, etc.
- Lowerer: Responsible for converting the computational graph into a C-like abstract syntax tree (AST).
- Renderer: Renders the AST as source code in the target language.
- Compiler: An interface that abstracts compiler calls.

## Optimization
In Harp, optimization is done in two main stages:
- Computational graph optimization: Tensor-wise graph optimizations are performed (e.g., fusing multiple element-wise operations into a single operation node).
- AST optimization: Removing unnecessary operations found in the abstract syntax tree (such as meaningless operations like a + 0) and unrolling loops.

## Quick Start

### Installation

Add Harp to your `Cargo.toml`:

```toml
[dependencies]
harp = { git = "https://github.com/uthree/harp" }
```

### Basic Usage

```rust
use harp::prelude::*;

fn main() {
    // Create a computation graph
    let mut graph = Graph::new();

    // Create input nodes
    let a = graph.input("a")
        .with_dtype(DType::F32)
        .with_shape(vec![10, 20])
        .build();

    let b = graph.input("b")
        .with_dtype(DType::F32)
        .with_shape(vec![10, 20])
        .build();

    // Perform operations (using operator overloading)
    let result = a + b;

    // Register output node
    graph.output("result", result);
}
```

### Using Helper Functions

```rust
use harp::prelude::*;

// Reduce operations
let sum = reduce_sum(input, 0);      // Sum along axis 0
let max_val = reduce_max(input, 1);  // Max along axis 1

// Element-wise operations
let reciprocal = recip(input);       // Reciprocal
let maximum = max(a, b);             // Element-wise maximum
```

### View Operations

```rust
use harp::prelude::*;

// Transpose
let transposed = input.view.clone().permute(vec![1, 0]);

// Add/remove dimensions
let unsqueezed = input.view.clone().unsqueeze(0);
let squeezed = input.view.clone().squeeze(2);

// Flip
let flipped = input.view.clone().flip(0);
```

## API Documentation

The library exports the following main modules:

- **`prelude`**: Commonly used types and traits (recommended to import with `use harp::prelude::*`)
- **`graph`**: Computation graph construction
- **`lowerer`**: Graph to AST conversion
- **`backend`**: Backend implementations (Metal, etc.)
- **`ast`**: Intermediate representation
- **`opt`**: Optimization passes

Top-level re-exports for convenience:
- `Graph`, `GraphNode`, `DType`, `AxisStrategy` from `graph`
- `Compiler`, `Renderer`, `Kernel`, `Buffer` from `backend`
- `Lowerer` from `lowerer`

# License
This repository is dual licensed under [Apache 2.0](./LICENSE_APACHE) and [MIT](./LICENSE_MIT), so you can use it either way if you need to.

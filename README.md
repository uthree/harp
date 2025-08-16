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

# License
This repository is dual licensed under [Apache 2.0](./LICENSE_APACHE) and [MIT](./LICENSE_MIT), so you can use it either way if you need to.
# Harp Tensor Design

This document outlines the design for implementing a tensor-like interface in Harp, inspired by libraries like tinygrad.

## Core Concepts

The design revolves around `Tensor`, a unified `Operator` set, and `ShapeTracker`.

### 1. `Tensor` and `TensorData`

-   **`Tensor`**: The primary user-facing struct. It's a lightweight wrapper that holds a reference-counted pointer (`Arc`) to its underlying data and computation definition.
-   **`TensorData`**: Contains the actual information about the computation graph at a specific node. It holds the operator and the source tensors (inputs).

```rust
// In src/tensor.rs
pub struct TensorData {
    // Note: This now holds `Box<dyn Operator>` instead of a separate trait.
    pub op: Box<dyn Operator>,
    pub src: Vec<Tensor>,
}
```

### 2. Unified Operators and Key Traits

To avoid redundancy, Harp will use a single set of operator structs (e.g., `OpAdd`, `OpMul`) for both the low-level `Node` graph and the high-level `Tensor` graph.

A set of marker traits and utility traits will be used to define the capabilities of each operator.

-   **`TensorOperator`**: A marker trait to designate which operators are permitted in the construction of a `Tensor` graph.
-   **`Elementwise`**: A marker trait to indicate that an operator is applied element-by-element (e.g., `OpAdd`, `Sin`).
-   **`HasIdentityElement`**: A trait for binary operators that have an identity element. This is crucial for operations like `Reduce`.
    ```rust
    trait HasIdentityElement {
        fn identity_element() -> Node;
    }
    ```
    For example, for `OpAdd`, the identity element is `0`. For `OpMul`, it is `1`.

### 3. Specialized Operators

Beyond standard arithmetic, the following specialized operators are required.

-   **`LOAD`**: Represents reading data from an input source.
-   **`STORE`**: Represents writing data to a memory buffer.
-   **`SINK`**: A special operator to mark a `Node` as a final result of a computation.
-   **`LOOP`**: Represents a looping construct for iterative computations.
-   **`Max`**: A binary, commutative, element-wise operator that returns the greater of its two inputs.
-   **`Reduce`**: A higher-order operator for performing reductions (like sum, product, max) along a specified axis. It collapses one dimension of a tensor. This is analogous to `pytorch.sum(axis, keepdim=False)`.
    ```rust
    struct Reduce {
        // The reduction operation (e.g., OpAdd, OpMul, Max).
        // The trait bounds ensure the reduction is well-defined.
        op: Box<dyn BinaryOperator + Commutative + HasIdentityElement + Elementwise>,
        axis: usize,
    }

    impl TensorOperator for Reduce {}
    ```

### 4. `ShapeTracker`

A crucial component for handling multi-dimensional arrays. It tracks the dimensions of a tensor and holds the expression for calculating the memory offset from a multi-dimensional index. This allows Harp to resolve tensor indexing into its core `Node`-based computation graph.

```rust
// In src/tensor.rs
use crate::node::Node;
use std::rc::Rc;

pub struct ShapeTracker {
    // The size of each dimension (e.g., [4, 3] for a 4x3 matrix).
    pub dims: Vec<Rc<Node>>,
    // The mathematical expression to convert a multi-dimensional index
    // into a linear memory offset.
    pub index_expr: Vec<Rc<Node>>,
}
```

### 5. Compilation (`Tensor` to `Node`)

`Tensor` operations build a high-level computation graph. To execute or optimize this graph using Harp's existing engine, it must be "compiled" into the lower-level `Node` graph.

A `compile` method on `Tensor` will perform this transformation recursively, using the `ShapeTracker` to resolve indexing logic into scalar arithmetic.

```rust
// In src/tensor.rs
impl Tensor {
    /// Compiles the tensor's computation graph into a traditional Node graph.
    /// This process resolves tensor indexing into scalar operations.
    pub fn compile(&self, shape_tracker: &ShapeTracker) -> Rc<Node> {
        // ... implementation details ...
        todo!()
    }
}
```
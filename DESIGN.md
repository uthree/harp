# Harp Design Document

This document outlines the core design principles of the Harp library, focusing on its computation graph, operator hierarchy, and tensor interface.

## Core Concepts

The design revolves around two main graph representations: the low-level `Node` graph for scalar operations and the high-level `Tensor` graph for multi-dimensional array operations.

### 1. `Node` Graph

-   The fundamental computation graph, where each node represents a scalar value and a single operation (e.g., `OpAdd`, `Sin`).
-   This graph is the target for compilation and optimization.

### 2. `Tensor` Graph

-   **`Tensor`**: The primary user-facing struct. It's a lightweight, reference-counted handle to its underlying computation definition.
-   **`TensorData`**: Contains the operator and source tensors (inputs) that define a node in the tensor graph.
-   **Lazy Evaluation**: Tensor operations do not perform computations immediately. Instead, they build a high-level graph representing the sequence of operations. Actual computation is deferred until a backend compiles and executes the graph.

### 3. Operator Hierarchy

Harp uses a unified set of operator structs (e.g., `OpAdd`, `OpMul`) for both graphs, categorized by a system of traits to define their behavior and properties. This is crucial for abstraction and optimization.

-   **`Operator`**: The base trait for all operations.
-   **`PrimitiveOp`**: A marker for the most basic operators that a compiler backend must implement (e.g., `OpAdd`, `OpMul`, `Load`, `OpUniform`). These are the fundamental building blocks of all computations.
-   **`FusedOp`**: A trait for composite operators that can be decomposed into a subgraph of more primitive ones. This allows for high-level abstractions without increasing the complexity of the compiler backend. The decomposition is defined in the `fallback` method.
    -   *Examples*: `OpSub` is fused into `a + (b * -1)`. `OpDiv` is fused into `a * recip(b)`. `OpRandn` is fused into a graph representing the Box-Muller transform, which uses `OpUniform` as its primitive source of randomness.
-   **Mathematical Property Traits**:
    -   **`BinaryOp`**: A marker for operators that take two operands.
    -   **`CommutativeOp`**: A marker for binary operators where the order of operands does not matter (e.g., `a + b == b + a`).
    -   **`AssociativeOp`**: A marker for binary operators where the grouping of operations does not matter (e.g., `(a + b) + c == a + (b * c)`). These properties are vital for graph optimizers, such as reordering operations or rebalancing computation trees.
-   **`TensorOperator`**: A marker trait to designate which operators are permitted in the construction of a `Tensor` graph.

### 4. Tensor Initialization and Shape Manipulation

-   **Shape Representation**: Shapes are represented by `Vec<usize>` for consistency with Rust's standard library indexing and memory management.
-   **Type-Safe Constants**: Generic methods like `Tensor::full` use the `DType` trait to preserve type information (e.g., `f32`, `i64`) within the graph, avoiding premature casting.
-   **Memory-Efficient Initialization**:
    -   `Tensor::full(shape, value)`, `zeros(shape)`, `ones(shape)` are implemented as memory-efficient view operations. They create a single scalar `Const` node and use the `Expand` operator to broadcast it to the target shape without allocating large amounts of memory.
-   **Declarative Randomness**:
    -   `Tensor::uniform(shape)` and `Tensor::randn(shape)` do not generate random numbers immediately. They place `OpUniform` or `OpRandn` operators in the graph, representing the *intent* to generate random numbers. The actual generation is deferred to the execution engine.

### 5. Compilation (`Tensor` to `Node`)

-   The high-level `Tensor` graph is compiled into a low-level `Node` graph for execution.
-   The `compile` method, with the help of a `ShapeTracker`, recursively traverses the tensor graph, resolving multi-dimensional indexing and tensor operations into scalar arithmetic on `Node`s.
-   **`ShapeTracker`**: A crucial component that tracks the dimensions of a tensor and holds the expression for calculating the linear memory offset from a multi-dimensional index.

### 6. API and Tooling

-   **`prelude` Module**: To improve ergonomics, a `prelude` module is provided to conveniently import the most commonly used items (`Tensor`, `Node`, `ToDot`, `DType`, various operators, etc.).
-   **`ToDot` Trait**: A common trait for graph visualization. It is implemented for both `Node` and `Tensor` to generate a string representation in DOT format, which can be used with tools like Graphviz.

## 7. Backend and Code Generation

The ultimate goal of Harp is to compile the `Node` graph into high-performance code for various backends like C, CUDA, or Metal.

### a. Directory Structure

-   All backend-related modules are organized under the `src/backend/` directory for clarity and future expansion.
    -   `src/backend/renderer.rs`: Defines the core rendering traits.
    -   `src/backend/codegen.rs`: Contains the main `CodeGenerator` engine.
    -   `src/backend/c/`: A directory for the C language backend, containing its `mod.rs` with the `CRenderer` implementation.

### b. Code Generation Engine (`CodeGenerator`)

-   The `CodeGenerator` is the central engine responsible for orchestrating the conversion of a `Node` graph into source code.
-   **Processing Strategy**: It adopts a **bottom-up** approach inspired by the `Rewriter`.
    1.  It performs a **topological sort** of the graph to determine the correct execution order of nodes.
    2.  It iterates through the sorted nodes, processing leaf nodes first.
-   **Intermediate Variable Management**: To avoid generating overly complex, nested expressions, the `CodeGenerator` manages the creation of intermediate variables. For each non-leaf operation, it generates a statement (e.g., `float v1 = v0 + 2.0;`) and stores the result in a new, unique variable (`v1`).
-   **Memoization**: The engine uses a `HashMap` to cache the resulting variable name for each rendered `Node`. This ensures that any node with multiple dependents is only processed once, preventing redundant code generation.

### c. Renderer Traits

-   **`Renderer` Trait**: An abstraction for a specific code generation target (e.g., C, Metal). Its primary role is to dispatch an `Operator` to the correct rendering logic.
-   **`Render<Op>` Trait**: A generic trait that defines how to render a *specific*, primitive operator `Op`. Each `Renderer` implements this trait for all the primitive operators it supports. This provides a high degree of type safety and modularity.

### d. `FusedOp` Fallback

-   The `CodeGenerator` handles the `FusedOp` fallback mechanism.
-   When a `Renderer` reports that it does not support a given operator (by returning `None`), the `CodeGenerator` checks if the operator implements `FusedOp`.
-   If it does, the `CodeGenerator` invokes the `fallback()` method to get the equivalent subgraph of primitive operators and recursively renders that subgraph instead.
-   This powerful abstraction allows renderers to support a wide range of complex operations by implementing only a minimal set of primitives.

### e. Future Components (Design)

-   **`Compiler` Trait**: Will be responsible for taking the generated source code string and invoking an external compiler (e.g., `gcc`, `nvcc`) to produce a runnable artifact. It will also check for the availability of the necessary toolchain.
-   **`Kernel` Trait**: A handle to a compiled, loadable kernel, likely representing a function pointer from a dynamic library (`.so`, `.dll`).
-   **`Backend` Trait**: A high-level interface統括 (e.g., `Cuda<0>`) that brings together the `Renderer`, `Compiler`, and device management.
-   **`DeviceTensor` Trait**: Represents a tensor whose data resides on a specific device (e.g., a GPU). It will manage device memory and provide methods for data transfer between host and device.
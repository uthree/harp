# Workspace Design

To improve maintainability, modularity, and enforce clear API boundaries, the `harp` project is structured as a multi-crate workspace. This approach separates concerns, making the library easier to develop, test, and extend.

The dependency chain is designed to be clear and acyclic:
`harp-tensor` → `harp-ir` → `harp-graph`
`harp-codegen` → `harp-ir` → `harp-graph`

## Crate Structure

### 1. `harp-graph` (Foundational Library)

-   **Responsibility**: Provides a generic, reusable graph data structure.
-   **Core Components**: `Graph<T>`, `Node<T>`, `NodeId`.
-   **Role**: Acts as a foundational library for creating and manipulating graph structures. It is agnostic to the concepts of "computation," "operators," or "tensors."
-   **Dependencies**: None within the workspace.

### 2. `harp-ir` (Intermediate Representation)

-   **Responsibility**: Defines the compiler's intermediate representations.
-   **Core Components**:
    -   **Computation Graph**: A `harp-graph::Graph` where the node data is a low-level `Operator`.
    -   **Procedural IR**: `Procedure`, `Instruction`, and `VReg` structs for linearized operations.
    -   **Shared Types**: `DType` and other common data types.
-   **Role**: Acts as the "common language" for the compiler. It defines the data structures that the frontend produces and the backend consumes.
-   **Dependencies**: `harp-graph`.

### 3. `harp-tensor` (Frontend)

-   **Responsibility**: Provides the public, user-facing `Tensor` API.
-   **Core Components**: `Tensor` struct and its associated mathematical operations.
-   **Role**: Constructs a high-level computation graph from user code. It then lowers this graph into the representation defined in `harp-ir`.
-   **Dependencies**: `harp-ir`.

### 4. `harp-codegen` (Backend)

-   **Responsibility**: Generates target-specific source code from the intermediate representations.
-   **Core Components**:
    -   **Linearizer**: Converts the computation graph from `harp-ir` into the procedural IR (also defined in `harp-ir`).
    -   **Renderers**: Translates the procedural IR into target source code (e.g., C, CUDA, Metal).
-   **Role**: Acts as the compilation backend, turning the abstract computations into executable code.
-   **Dependencies**: `harp-ir`.
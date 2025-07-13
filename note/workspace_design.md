# Workspace Design

To improve maintainability, modularity, and enforce clear API boundaries, the `harp` project will be restructured into a multi-crate workspace. This approach separates concerns, making the library easier to develop, test, and extend.

## Crate Structure

The workspace will consist of three primary crates, each with a distinct responsibility:

### 1. `harp-api`

-   **Responsibility**: Provides the public, user-facing API. This is the primary entry point for users of the library.
-   **Core Components**: `Tensor` struct and associated operations.
-   **Role**: Handles the construction of the high-level computation graph from user code. It will act as a bridge to the IR crate by lowering the `Tensor` graph into a `Node` graph.
-   **Dependencies**: Depends on `harp-ir`.

### 2. `harp-ir` (Intermediate Representation)

-   **Responsibility**: Defines the core, shared data structures and logic for the computation graph. This crate acts as the "common language" between the frontend API and the backend code generation.
-   **Core Components**: `Node`, `Operator`, `DType`, pattern matching, and graph simplification logic.
-   **Role**: Provides a stable, decoupled intermediate layer. It has no knowledge of the user-facing API or the final code generation targets.
-   **Dependencies**: None within the workspace.

### 3. `harp-codegen` (Code Generation)

-   **Responsibility**: Consumes the graph from `harp-ir` and generates target-specific source code.
-   **Core Components**:
    -   Linearizer: Converts the `Node` graph into a procedural Intermediate Representation (IR).
    -   Procedural IR: Defines `Procedure` and `Instruction` structs.
    -   Renderers: Translates the procedural IR into target code (e.g., C, CUDA).
-   **Role**: Acts as the compilation backend. It is responsible for the final stages of turning the abstract computation into executable code.
-   **Dependencies**: Depends on `harp-ir`.

This structure ensures that the frontend (`harp-api`) and backend (`harp-codegen`) are decoupled, communicating only through the well-defined structures in `harp-ir`.

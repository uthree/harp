# Harp Intermediate Representation (IR) Specification

This document outlines the design specification for Harp's Intermediate Representation (IR), which is used to translate the computation graph into an executable format.

## 1. Overall Philosophy

The primary goal of this IR is to create a representation that is:
- Close to the hardware execution model, especially for GPUs.
- Amenable to various optimizations.
- A suitable target for compiling the high-level computation graph.

The design adopts a static memory allocation model, where all necessary memory is allocated in a single arena before execution begins.

## 2. Module Structure

The project will be organized as follows:
- `src/graph/`: Contains the user-facing graph representation (Nodes, Tensors, Operators, etc.).
- `src/ir/`: Will contain the new Intermediate Representation and the logic for converting the graph to this IR.
- `src/shape/`: Remains at the top level for use by other modules.

## 3. IR Components

The IR is composed of three main hierarchical structures: `Function`, `Kernel`, and `Instruction`.

### 3.1. `Function`

A `Function` is the top-level container for a complete, executable computation. It manages memory and orchestrates the execution of `Kernel`s.

```rust
pub struct Function {
    pub name: String,
    pub kernels: Vec<Kernel>,
    // A list of all memory buffers required for the function.
    pub buffers: Vec<Buffer>,
    // The total size of the memory arena required for execution (in bytes).
    pub required_memory: usize,
    // A list of BufferIds that are arguments to the function.
    pub args: Vec<BufferId>,
    // The BufferId that holds the return value of the function.
    pub ret: BufferId,
}
```

### 3.2. `Kernel`

A `Kernel` represents a unit of computation that can be executed in parallel, mapping well to concepts like GPU kernels. A `Function` can contain multiple `Kernel`s, which can be executed sequentially or in parallel depending on their data dependencies.

```rust
pub struct Kernel {
    pub name: String,
    pub instructions: Vec<Instruction>,
    // Data types for each virtual register used in this kernel.
    pub vregs: Vec<DType>,
    // Dimensions for launching the kernel on a parallel device (e.g., GPU grid/block size).
    pub launch_dims: [usize; 3],
    // A list of BufferIds that this kernel reads from.
    pub reads: Vec<BufferId>,
    // A list of BufferIds that this kernel writes to.
    pub writes: Vec<BufferId>,
}
```

### 3.3. `Instruction`

An `Instruction` is the most basic unit of operation within a `Kernel`.

```rust
// Represents a virtual register holding an intermediate value.
pub type VReg = usize;

// Represents a unique identifier for a memory buffer.
pub type BufferId = usize;

// The set of operations that can be performed.
pub enum Instruction {
    // Loads a constant value into a virtual register.
    Const { out: VReg, val: Scalar },

    // Performs an ALU operation (e.g., Add, Mul, Sin).
    Alu {
        op: AluOp,
        out: VReg,
        lhs: VReg,
        rhs: Option<VReg>, // None for unary operations.
    },

    // Loads data from a memory buffer into a virtual register.
    Load { out: VReg, from: BufferId, shape: ShapeTracker },

    // Stores data from a virtual register into a memory buffer.
    Store { to: BufferId, from: VReg, shape: ShapeTracker },

    // Generates random numbers.
    Rand { op: RandOp, out: VReg, shape: ShapeTracker },

    // Future extensions for control flow.
    // Loop { ... },
    // If { ... },
}

// Enum for ALU operations.
pub enum AluOp {
    Add, Mul, Exp2, Log2, Sin, Sqrt, Recip, LessThan, // etc.
}

// Enum for random number generation operations.
pub enum RandOp {
    Uniform, // 0-1 uniform distribution
    Normal,  // Standard normal distribution
}
```

## 4. Memory Management

Memory is managed statically using a single memory arena per `Function`.

- **Static Allocation**: Before a `Function` is executed, a single contiguous block of memory (`arena`) of size `required_memory` is allocated. It is freed only after the function completes.
- **Buffers as Views**: `Buffer` objects do not own memory themselves. Instead, they represent a "view" or a slice of the main arena, defined by an `offset` and `size`.
- **Memory Reuse**: The graph-to-IR compiler will be responsible for analyzing the liveness of tensors to allow non-overlapping tensors to share the same memory regions, minimizing the total `required_memory`.

```rust
// Represents a view into the memory arena.
pub struct Buffer {
    pub id: BufferId,
    // The starting offset within the memory arena (in bytes).
    pub offset: usize,
    // The size of this buffer (in bytes).
    pub size: usize,
    pub dtype: DType,
    // The memory space where the buffer resides (e.g., CPU or GPU).
    pub memory_space: MemorySpace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    Host,   // CPU memory
    Device, // GPU memory
}
```

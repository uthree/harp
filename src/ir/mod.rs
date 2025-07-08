/// The Intermediate Representation (IR) module.
pub mod compiler;

use crate::graph::dtype::DType;
use crate::graph::dtype::Scalar;
use crate::shape::tracker::ShapeTracker;
use std::fmt;

// --- Memory Management ---

/// A unique identifier for a memory buffer.
pub type BufferId = usize;

/// Represents a view into a memory arena.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub id: BufferId,
    /// The starting offset within the memory arena (in bytes).
    pub offset: usize,
    /// The size of this buffer (in bytes).
    pub size: usize,
    pub dtype: DType,
    /// The memory space where the buffer resides (e.g., CPU or GPU).
    pub memory_space: MemorySpace,
}

/// Defines the memory space for a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    /// CPU memory
    Host,
    /// GPU memory
    Device,
}

// --- Instructions and Operations ---

/// Represents a virtual register holding an intermediate value.
pub type VReg = usize;

/// The set of operations that can be performed within a kernel.
#[derive(Debug, Clone)]
pub enum Instruction {
    /// Loads a constant value into a virtual register.
    Const { out: VReg, val: Scalar },

    /// Performs an ALU operation (e.g., Add, Mul, Sin).
    Alu {
        op: AluOp,
        out: VReg,
        lhs: VReg,
        /// None for unary operations.
        rhs: Option<VReg>,
    },

    /// Loads data from a memory buffer into a virtual register.
    Load {
        out: VReg,
        from: BufferId,
        shape: ShapeTracker,
    },

    /// Stores data from a virtual register into a memory buffer.
    Store {
        to: BufferId,
        from: VReg,
        shape: ShapeTracker,
    },

    /// Generates random numbers.
    Rand {
        op: RandOp,
        out: VReg,
        shape: ShapeTracker,
    },
    // Future extensions for control flow.
    // Loop { ... },
    // If { ... },
}

/// Enum for ALU (Arithmetic Logic Unit) operations.
#[derive(Debug, Clone, Copy)]
pub enum AluOp {
    Add,
    Mul,
    Exp2,
    Log2,
    Sin,
    Sqrt,
    Recip,
    LessThan,
}

/// Enum for random number generation operations.
#[derive(Debug, Clone, Copy)]
pub enum RandOp {
    /// 0-1 uniform distribution
    Uniform,
    /// Standard normal distribution
    Normal,
}

// --- Kernel and Function ---

/// Represents a unit of computation that can be executed in parallel.
#[derive(Debug, Clone)]
pub struct Kernel {
    pub name: String,
    pub instructions: Vec<Instruction>,
    /// Data types for each virtual register used in this kernel.
    pub vregs: Vec<DType>,
    /// Dimensions for launching the kernel on a parallel device.
    pub launch_dims: [usize; 3],
    /// A list of BufferIds that this kernel reads from.
    pub reads: Vec<BufferId>,
    /// A list of BufferIds that this kernel writes to.
    pub writes: Vec<BufferId>,
}

/// A `Function` is the top-level container for a complete, executable computation.
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub kernels: Vec<Kernel>,
    /// A list of all memory buffers required for the function.
    pub buffers: Vec<Buffer>,
    /// The total size of the memory arena required for execution (in bytes).
    pub required_memory: usize,
    /// A list of BufferIds that are arguments to the function.
    pub args: Vec<BufferId>,
    /// The BufferId that holds the return value of the function.
    pub ret: BufferId,
}

// --- Display Implementations ---

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scalar::Bool(v) => write!(f, "{}", v),
            Scalar::I8(v) => write!(f, "{}", v),
            Scalar::U8(v) => write!(f, "{}", v),
            Scalar::I16(v) => write!(f, "{}", v),
            Scalar::U16(v) => write!(f, "{}", v),
            Scalar::I32(v) => write!(f, "{}", v),
            Scalar::U32(v) => write!(f, "{}", v),
            Scalar::I64(v) => write!(f, "{}", v),
            Scalar::U64(v) => write!(f, "{}", v),
            Scalar::F32(v) => write!(f, "{}", v),
            Scalar::F64(v) => write!(f, "{}", v),
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instruction::Const { out, val } => {
                write!(f, "v{} = const {}", out, val)
            }
            Instruction::Alu { op, out, lhs, rhs } => {
                if let Some(rhs) = rhs {
                    write!(f, "v{} = {:?} v{}, v{}", out, op, lhs, rhs)
                } else {
                    write!(f, "v{} = {:?} v{}", out, op, lhs)
                }
            }
            Instruction::Load { out, from, shape } => {
                write!(f, "v{} = load buf{}[{}]", out, from, shape)
            }
            Instruction::Store { to, from, shape } => {
                write!(f, "store buf{}[{}] = v{}", to, shape, from)
            }
            Instruction::Rand { op, out, shape } => {
                write!(f, "v{} = {:?} [{}]", out, op, shape)
            }
        }
    }
}

impl fmt::Display for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "kernel {}:", self.name)?;
        for inst in &self.instructions {
            writeln!(f, "  {}", inst)?;
        }
        Ok(())
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "function {}:", self.name)?;
        for kernel in &self.kernels {
            write!(f, "{}", kernel)?;
        }
        Ok(())
    }
}
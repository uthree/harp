//! Defines the core traits and data structures for the backend infrastructure.
//!
//! This module provides abstractions for:
//! - `Buffer`: A generic container for data on a device.
//! - `Kernel`: A compiled, executable function.
//! - `Compiler`: A component that compiles code into a `Kernel`.
//! - `Renderer`: A component that translates an AST into source code.
//! - `Device`: A component that manages memory allocation.

use crate::ast::{AstNode, DType};
use crate::tensor::shape::expr::Expr;
use ::ndarray::ArrayD;
use std::any::TypeId;
use std::ffi::c_void;

// --- Core Data Structures ---

/// Information about a single buffer used in a kernel.
#[derive(Debug, Clone)]
pub struct BufferInfo {
    pub dtype: DType,
    pub shape: Vec<Expr>,
}

/// Detailed information about a compiled kernel's inputs and outputs.
#[derive(Debug, Clone, Default)]
pub struct KernelDetails {
    /// Information about each buffer (input and output).
    pub buffers: Vec<BufferInfo>,
    /// The names of variables used to define dynamic shapes.
    pub shape_variables: Vec<String>,
}

// --- Core Traits ---

/// A trait for a generic buffer that can be passed to a kernel.
/// This trait is object-safe.
/// should be free when dropped.
pub trait Buffer: Drop {
    /// Returns a mutable pointer to the buffer's raw data.
    fn as_mut_ptr(&mut self) -> *mut c_void;

    /// Returns the data type of the elements in the buffer.
    fn dtype(&self) -> DType;

    /// Returns the shape of the buffer as a `Vec` of symbolic expressions.
    fn shape(&self) -> Vec<Expr>;

    /// Returns the total number of elements in the buffer.
    fn size(&self) -> usize {
        self.shape()
            .iter()
            .map(|e| match e {
                Expr::Const(v) => *v as usize,
                // In a real scenario, you might need a way to resolve variables.
                // For now, we panic if the shape is not fully concrete.
                _ => panic!("Cannot calculate size of buffer with dynamic shapes"),
            })
            .product()
    }
}

/// An extension trait for `Buffer` providing `ndarray` conversion.
pub trait TryIntoNdarray: Buffer {
    /// Creates an `ndarray::ArrayD` (dynamically dimensioned array) from the
    /// buffer's data by cloning it.
    ///
    /// Returns `None` if the requested `ndarray` element type `T` does not match
    /// the buffer's `DType`.
    ///
    /// # Type Parameters
    ///
    /// * `T`: The desired element type of the output `ndarray::ArrayD`. Must implement `Clone`
    ///   and have a `'static` lifetime.
    fn try_into_ndarray<T: Clone + 'static>(&mut self) -> Option<ArrayD<T>> {
        if TypeId::of::<T>() != self.dtype().to_type_id() {
            return None;
        }

        let shape: Vec<usize> = self
            .shape()
            .iter()
            .map(|e| match e {
                Expr::Const(v) => *v as usize,
                _ => panic!("Cannot create ndarray from dynamic shape"),
            })
            .collect();

        let size = shape.iter().product();
        if size == 0 {
            return ArrayD::from_shape_vec(shape, vec![]).ok();
        }

        let data_slice = unsafe { std::slice::from_raw_parts(self.as_mut_ptr() as *const T, size) };
        let data_vec = data_slice.to_vec();
        ArrayD::from_shape_vec(shape, data_vec).ok()
    }
}

/// Blanket implementation of `TryIntoNdarray` for all types that implement `Buffer`.
impl<T: Buffer> TryIntoNdarray for T {}

/// A trait for a compiled, executable kernel.
pub trait Kernel<Var: Buffer> {
    /// Executes the kernel with the given buffers and shape variables.
    fn call(&self, buffers: Vec<Var>, shape_variables: Vec<usize>) -> Vec<Var>;
}

/// A trait for a compiler that turns a code representation into a `Kernel`.
pub trait Compiler<Var: Buffer, CodeRepr = String, CompilerOption = ()> {
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: CompilerOption);
    fn compile(&mut self, code: CodeRepr) -> impl Kernel<Var>;
}

/// A trait for a component that translates an `AstNode` into a code representation.
pub trait Renderer<CodeRepr = String> {
    fn new() -> Self;
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}

/// A trait for a device that can allocate and manage memory.
pub trait Device<Var: Buffer> {
    fn allocate(&mut self, dtype: DType, size: usize) -> Var;
    fn is_available(&self) -> bool;
}

// --- Submodules ---
pub mod c;
pub mod ndarray;

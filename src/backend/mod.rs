//! Defines the core traits and data structures for the backend infrastructure.
//!
//! This module provides abstractions for:
//! - `Buffer`: A generic container for data on a device.
//! - `Kernel`: A compiled, executable function.
//! - `Compiler`: A component that compiles code into a `Kernel`.
//! - `Renderer`: A component that translates an AST into source code.

// --- Submodules ---
pub mod c;
pub mod generic;

// --- Imports ---
use crate::{
    ast::{AstNode, DType},
    graph::{Graph, NodeId, shape::expr::Expr},
};
use ndarray::ArrayD;
use rustc_hash::FxHashMap;
use std::any::{Any, TypeId};
use std::collections::HashMap;

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
    /// A map from the graph's NodeId to the index in the `buffers` Vec.
    pub buffer_map: FxHashMap<NodeId, usize>,
}

// --- Core Traits ---

/// A trait for a generic buffer that can be passed to a kernel.
/// This trait is object-safe.
/// should be free when dropped.
pub trait Buffer: 'static + Sized + Clone {
    /// Provides a way to downcast the buffer to a concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Provides a way to downcast the buffer to a concrete type.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Returns a slice of the buffer's raw data as bytes.
    fn as_bytes(&self) -> &[u8];

    /// Returns a mutable slice of the buffer's raw data as bytes.
    fn as_mut_bytes(&mut self) -> &mut [u8];

    /// Returns the data type of the elements in the buffer.
    fn dtype(&self) -> DType;

    /// 形状を返す。Bufferとして実態を持っている時点でサイズは確定しているので、Exprではない。
    fn shape(&self) -> Vec<usize>;

    /// Allocates a new buffer with the given dtype and shape.
    fn allocate(dtype: DType, shape: Vec<usize>) -> Self;
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

        let shape = self.shape();
        let size = shape.iter().product();
        if size == 0 {
            return ArrayD::from_shape_vec(shape, vec![]).ok();
        }

        let bytes = self.as_mut_bytes();
        let data_slice = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const T, size) };
        let data_vec = data_slice.to_vec();
        ArrayD::from_shape_vec(shape, data_vec).ok()
    }
}

/// Blanket implementation of `TryIntoNdarray` for all types that implement `Buffer`.
impl<T: Buffer> TryIntoNdarray for T {}

/// A trait for a compiled, executable kernel.
pub trait Kernel<B: Buffer> {
    /// Returns detailed information about the kernel's inputs and outputs.
    fn details(&self) -> &KernelDetails;

    /// Executes the kernel with the given buffers and shape variables.
    fn call(&mut self, buffers: Vec<B>, shape_variables: &[usize]) -> Vec<B>;
}

/// A trait for a compiler that turns a code representation into a `Kernel`.
pub trait Compiler<B: Buffer, CodeRepr = String, CompilerOption = ()> {
    type KernelType: Kernel<B>;
    fn new() -> Self;
    fn is_available(&self) -> bool;
    fn with_option(&mut self, option: CompilerOption);
    fn compile(&mut self, code: &CodeRepr, details: KernelDetails) -> Self::KernelType;
}

/// A trait for a component that translates an `AstNode` into a code representation.
pub trait Renderer<CodeRepr = String> {
    fn new() -> Self;
    fn render(&mut self, ast: AstNode) -> CodeRepr;
}

pub trait Backend<B: Buffer> {
    fn new() -> Self
    where
        Self: Sized;
    fn is_available(&self) -> bool;
    fn execute(
        &self,
        graph: &Graph,
        inputs: Vec<B>,
        shape_variables: Vec<usize>,
    ) -> HashMap<NodeId, B>;
}

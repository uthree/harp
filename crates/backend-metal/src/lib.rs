//! Metal backend for Harp (macOS only)
//!
//! This crate provides native GPU execution using Apple's Metal API.
//!
//! # Usage
//!
//! ```ignore
//! use harp_backend_metal::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now Metal is available as a device
//! use harp_core::backend::HarpDevice;
//! let device = HarpDevice::metal(0).unwrap();
//! device.set_as_default();
//! ```

#![cfg(target_os = "macos")]

mod buffer;
mod compiler;
mod device;
mod kernel;
pub mod renderer;

pub use buffer::MetalBuffer;
pub use compiler::MetalCompiler;
pub use device::{MetalDevice, MetalError};
pub use kernel::MetalKernel;

// Re-export renderer types for convenience
pub use harp_core::backend::renderer::OptimizationLevel;
pub use renderer::{MetalCode, MetalRenderer};

use harp_core::ast::DType;
use harp_core::backend::cache::{
    CacheEntry, KernelCacheKey, get_cached_kernel, insert_cached_kernel,
};
use harp_core::backend::device::{BackendRegistry, DeviceError};
use harp_core::backend::global::{DeviceKind, get_default_device};
use harp_core::backend::traits::{Buffer, Compiler, Device, TypedBuffer};
use harp_core::backend::{BufferSignature, KernelSignature, Pipeline};
use harp_core::tensor::TensorInner;
use harp_core::tensor::forward::{ForwardError, collect_input_data_inner, register_realizer};
use harp_core::tensor::lowerer::lower_tensor_inner;
use harp_core::tensor::shape::Expr;
use harp_core::tensor::stringify::stringify_graph_inner;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// Metal backend registry implementation
struct MetalBackendRegistry;

impl BackendRegistry for MetalBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::Metal
    }

    fn name(&self) -> &str {
        "Metal"
    }

    fn is_available(&self) -> bool {
        MetalDevice::is_available()
    }

    fn create_device(&self, index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        let device = MetalDevice::with_device(index).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to create Metal device: {}", e))
        })?;
        Ok(Arc::new(device))
    }

    fn list_devices(&self) -> Vec<String> {
        metal::Device::all()
            .into_iter()
            .map(|d| d.name().to_string())
            .collect()
    }
}

// ============================================================================
// Realizer Implementation
// ============================================================================

/// Realize a TensorInner on a Metal device
fn realize_metal(inner: &TensorInner) -> Result<(), ForwardError> {
    // Get the Metal device
    let device: Arc<MetalDevice> = get_default_device::<MetalDevice>()
        .ok_or_else(|| ForwardError::DeviceUnavailable("Metal device not found".to_string()))?;

    // Generate cache key from graph structure (include device identity)
    let graph_repr = stringify_graph_inner(inner);
    let device_id = Arc::as_ptr(&device) as usize;
    let cache_key = KernelCacheKey::new(graph_repr, DeviceKind::Metal, device_id);

    // Check cache for compiled kernel
    let compiled: CacheEntry = if let Some(cached) = get_cached_kernel(&cache_key) {
        log::debug!("Kernel cache hit: {}", cache_key.graph_repr());
        cached
    } else {
        log::debug!("Kernel cache miss: {}", cache_key.graph_repr());

        // Collect input tensor data
        let input_data = collect_input_data_inner(inner);

        // Lower TensorInner to AST directly
        let ast = lower_tensor_inner(inner);

        // Create input signatures
        let input_signatures: Vec<BufferSignature> = input_data
            .iter()
            .enumerate()
            .map(|(i, (_, shape))| {
                let shape_expr: Vec<Expr> = shape.iter().map(|&s| Expr::from(s as i64)).collect();
                BufferSignature::new(format!("input{}", i), shape_expr)
            })
            .collect();

        // Create signature from tensor metadata
        let output_shape_expr: Vec<Expr> = inner
            .shape()
            .iter()
            .map(|&s| Expr::from(s as i64))
            .collect();
        let signature = KernelSignature::new(
            input_signatures,
            vec![BufferSignature::new(
                "output".to_string(),
                output_shape_expr,
            )],
        );

        // Create pipeline and compile from AST
        let renderer = MetalRenderer::default();
        let compiler = MetalCompiler::new();
        let mut pipeline: Pipeline<MetalRenderer, MetalDevice, MetalCompiler> =
            Pipeline::new(renderer, compiler, device.as_ref().clone());

        let compiled = pipeline
            .compile_ast(ast, signature)
            .map_err(|e| ForwardError::CompilationError(format!("Failed to compile: {:?}", e)))?;

        // Insert into cache
        insert_cached_kernel(cache_key, compiled.clone());

        compiled
    };

    // Collect input tensor data (needed for buffer creation)
    let input_data = collect_input_data_inner(inner);

    // Create input buffers from collected data
    let mut input_buffers: Vec<MetalBuffer> = Vec::new();
    for (data, shape) in &input_data {
        let buffer = MetalBuffer::from_vec(device.as_ref(), shape.clone(), DType::F32, data)
            .map_err(|e| {
                ForwardError::ExecutionError(format!("Failed to create input buffer: {}", e))
            })?;
        input_buffers.push(buffer);
    }

    // Allocate output buffer
    let output_shape = inner.shape().to_vec();
    let mut output_buffer = MetalBuffer::allocate(device.as_ref(), output_shape, DType::F32)
        .map_err(|e| {
            ForwardError::ExecutionError(format!("Failed to create output buffer: {}", e))
        })?;

    // Execute kernel with input and output buffers using dyn Buffer
    let input_refs: Vec<&dyn Buffer> = input_buffers.iter().map(|b| b as &dyn Buffer).collect();
    let mut output_refs: Vec<&mut dyn Buffer> = vec![&mut output_buffer as &mut dyn Buffer];

    // Compute dispatch sizes (no shape vars for now since shapes are static)
    let shape_vars: HashMap<String, i64> = HashMap::new();
    let grid_size = compiled.dispatch_config.evaluate_grid_size(&shape_vars);
    let local_size = compiled.dispatch_config.evaluate_local_size(&shape_vars);

    compiled
        .kernel
        .execute_with_sizes(&input_refs, &mut output_refs, grid_size, local_size)
        .map_err(|e| ForwardError::ExecutionError(format!("Execution failed: {:?}", e)))?;

    // Store GPU buffer directly in inner
    if let Ok(mut guard) = inner.buffer().write() {
        *guard = Some(Box::new(output_buffer) as Box<dyn Buffer>);
    }

    Ok(())
}

// ============================================================================
// Initialization
// ============================================================================

/// Initialize the Metal backend
///
/// This function registers the Metal backend with harp-core, making it
/// available for device selection via `HarpDevice::auto()` or `HarpDevice::metal()`.
///
/// This should be called once at program startup. When using the `harp` facade
/// crate, this is done automatically via the `ctor` attribute.
pub fn init() {
    // Register the backend
    harp_core::backend::register_backend(Box::new(MetalBackendRegistry));

    // Register the realizer
    register_realizer(DeviceKind::Metal, realize_metal);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        // Just ensure init doesn't panic
        init();
    }
}

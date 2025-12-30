//! OpenCL backend for Harp
//!
//! This crate provides native GPU execution using the `ocl` crate.
//!
//! # Usage
//!
//! ```ignore
//! use harp_backend_opencl::init;
//!
//! // Initialize the backend (typically done automatically via ctor)
//! init();
//!
//! // Now OpenCL is available as a device
//! use harp_core::backend::HarpDevice;
//! let device = HarpDevice::opencl(0).unwrap();
//! device.set_as_default();
//! ```

mod buffer;
mod compiler;
mod device;
mod kernel;
pub mod renderer;

pub use buffer::OpenCLBuffer;
pub use compiler::OpenCLCompiler;
pub use device::{OpenCLDevice, OpenCLError};
pub use kernel::OpenCLKernel;

// Re-export renderer types for convenience
pub use harp_core::backend::renderer::OptimizationLevel;
pub use renderer::{OpenCLCode, OpenCLRenderer};

use harp_core::ast::DType;
use harp_core::backend::cache::disk::{
    DiskCacheMetadata, compute_cache_hash, harp_version, load_binary, save_binary,
};
use harp_core::backend::cache::{
    CacheEntry, KernelCacheKey, get_cached_kernel, insert_cached_kernel,
};
use harp_core::backend::device::{BackendRegistry, DeviceError};
use harp_core::backend::global::{DeviceKind, get_default_device};
use harp_core::backend::pipeline::DispatchSizeConfig;
use harp_core::backend::traits::{Buffer, Compiler, Device, Kernel, KernelConfig, TypedBuffer};
use harp_core::backend::{BufferSignature, KernelSignature, Pipeline};
use harp_core::tensor::TensorInner;
use harp_core::tensor::forward::{ForwardError, collect_input_data_inner, register_realizer};
use harp_core::tensor::lowerer::lower_tensor_inner;
use harp_core::tensor::shape::Expr;
use harp_core::tensor::stringify::stringify_graph_inner;
use ocl::{Device as OclDevice, Platform};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

// ============================================================================
// Backend Registration
// ============================================================================

/// OpenCL backend registry implementation
struct OpenCLBackendRegistry;

impl BackendRegistry for OpenCLBackendRegistry {
    fn kind(&self) -> DeviceKind {
        DeviceKind::OpenCL
    }

    fn name(&self) -> &str {
        "OpenCL"
    }

    fn is_available(&self) -> bool {
        OpenCLDevice::is_available()
    }

    fn create_device(&self, index: usize) -> Result<Arc<dyn Any + Send + Sync>, DeviceError> {
        let device = OpenCLDevice::with_device(index).map_err(|e| {
            DeviceError::InitializationError(format!("Failed to create OpenCL device: {}", e))
        })?;
        Ok(Arc::new(device))
    }

    fn list_devices(&self) -> Vec<String> {
        let mut devices = Vec::new();
        for platform in Platform::list() {
            if let Ok(ocl_devices) = OclDevice::list_all(platform) {
                for ocl_device in ocl_devices {
                    if let Ok(name) = ocl_device.name() {
                        devices.push(name);
                    }
                }
            }
        }
        devices
    }
}

// ============================================================================
// Realizer Implementation
// ============================================================================

/// Realize a TensorInner on an OpenCL device
fn realize_opencl(inner: &TensorInner) -> Result<(), ForwardError> {
    // Get the OpenCL device
    let device: Arc<OpenCLDevice> = get_default_device::<OpenCLDevice>()
        .ok_or_else(|| ForwardError::DeviceUnavailable("OpenCL device not found".to_string()))?;

    // Generate cache key from graph structure (include device identity)
    let graph_repr = stringify_graph_inner(inner);
    let device_id = Arc::as_ptr(&device) as usize;
    let cache_key = KernelCacheKey::new(graph_repr, DeviceKind::OpenCL, device_id);

    // Check cache for compiled kernel (memory -> disk -> compile)
    let compiled: CacheEntry = if let Some(cached) = get_cached_kernel(&cache_key) {
        log::debug!("Kernel cache hit (memory): {}", cache_key.graph_repr());
        cached
    } else {
        // Compute disk cache hash
        let disk_hash = compute_cache_hash(&cache_key);
        let device_name = device.device_name();

        // Try loading from disk cache
        let from_disk = load_binary(&disk_hash, "opencl").and_then(|(binary, meta)| {
            // Check version and device compatibility
            if meta.harp_version != harp_version() {
                log::debug!("Disk cache version mismatch, recompiling");
                return None;
            }
            if meta.device_name != device_name {
                log::debug!("Disk cache device mismatch, recompiling");
                return None;
            }

            // Build kernel config from metadata
            let mut config =
                KernelConfig::new(&meta.entry_point).with_global_work_size(meta.grid_size);
            if let Some(ls) = meta.local_size {
                config = config.with_local_work_size(ls);
            }

            // Compile from binary
            let compiler = OpenCLCompiler::new();
            let kernel = compiler
                .compile_from_binary(device.as_ref(), &binary, config)
                .ok()?;

            log::debug!("Kernel loaded from disk cache: {}", disk_hash);

            // Build dispatch config (use constant sizes since we stored concrete values)
            let dispatch_config = DispatchSizeConfig::from_const(
                meta.grid_size,
                meta.local_size.unwrap_or([1, 1, 1]),
            );

            // We need to reconstruct the signature for the CacheEntry
            // For now, create a minimal signature since we already have the kernel
            let signature = KernelSignature::new(vec![], vec![]);

            Some(CacheEntry::new(
                Box::new(kernel) as Box<dyn Kernel>,
                signature,
                dispatch_config,
            ))
        });

        if let Some(cached) = from_disk {
            // Insert into memory cache
            insert_cached_kernel(cache_key, cached.clone());
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
                    let shape_expr: Vec<Expr> =
                        shape.iter().map(|&s| Expr::from(s as i64)).collect();
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
            let renderer = OpenCLRenderer::default();
            let compiler = OpenCLCompiler::new();
            let mut pipeline: Pipeline<OpenCLRenderer, OpenCLDevice, OpenCLCompiler> =
                Pipeline::new(renderer, compiler, device.as_ref().clone());

            let compiled = pipeline.compile_ast(ast, signature).map_err(|e| {
                ForwardError::CompilationError(format!("Failed to compile: {:?}", e))
            })?;

            // Save to disk cache
            if let Some(opencl_kernel) = compiled
                .kernel
                .as_any()
                .downcast_ref::<OpenCLKernel>()
                .and_then(|k| k.get_binary().ok().map(|b| (k, b)))
            {
                let (kernel, binary) = opencl_kernel;
                let shape_vars: HashMap<String, i64> = HashMap::new();
                let grid_size = compiled.dispatch_config.evaluate_grid_size(&shape_vars);
                let local_size = compiled.dispatch_config.evaluate_local_size(&shape_vars);

                let metadata = DiskCacheMetadata {
                    entry_point: kernel.config().entry_point.clone(),
                    grid_size,
                    local_size: Some(local_size),
                    device_name: device_name.clone(),
                    harp_version: harp_version().to_string(),
                };

                if let Err(e) = save_binary(&disk_hash, "opencl", &binary, &metadata) {
                    log::warn!("Failed to save kernel to disk cache: {}", e);
                }
            }

            // Insert into memory cache
            insert_cached_kernel(cache_key, compiled.clone());

            compiled
        }
    };

    // Collect input tensor data (needed for buffer creation)
    let input_data = collect_input_data_inner(inner);

    // Create input buffers from collected data
    let mut input_buffers: Vec<OpenCLBuffer> = Vec::new();
    for (data, shape) in &input_data {
        let buffer = OpenCLBuffer::from_vec(device.as_ref(), shape.clone(), DType::F32, data)
            .map_err(|e| {
                ForwardError::ExecutionError(format!("Failed to create input buffer: {}", e))
            })?;
        input_buffers.push(buffer);
    }

    // Allocate output buffer
    let output_shape = inner.shape().to_vec();
    let mut output_buffer = OpenCLBuffer::allocate(device.as_ref(), output_shape, DType::F32)
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

/// Initialize the OpenCL backend
///
/// This function registers the OpenCL backend with harp-core, making it
/// available for device selection via `HarpDevice::auto()` or `HarpDevice::opencl()`.
///
/// This should be called once at program startup. When using the `harp` facade
/// crate, this is done automatically via the `ctor` attribute.
pub fn init() {
    // Register the backend
    harp_core::backend::register_backend(Box::new(OpenCLBackendRegistry));

    // Register the realizer
    register_realizer(DeviceKind::OpenCL, realize_opencl);
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

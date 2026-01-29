//! Device abstraction for tensor execution.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::dtype::DType;
use crate::uop::UOp;

/// Error type for device operations.
#[derive(Debug)]
pub enum DeviceError {
    CompilationFailed(String),
    ExecutionFailed(String),
    BufferError(String),
    UnsupportedOperation(String),
}

impl std::fmt::Display for DeviceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceError::CompilationFailed(s) => write!(f, "Compilation failed: {}", s),
            DeviceError::ExecutionFailed(s) => write!(f, "Execution failed: {}", s),
            DeviceError::BufferError(s) => write!(f, "Buffer error: {}", s),
            DeviceError::UnsupportedOperation(s) => write!(f, "Unsupported operation: {}", s),
        }
    }
}

impl std::error::Error for DeviceError {}

pub type Result<T> = std::result::Result<T, DeviceError>;

/// A buffer that holds tensor data on a device.
pub trait Buffer: Send + Sync {
    /// Returns the size in bytes.
    fn size(&self) -> usize;

    /// Returns the data type.
    fn dtype(&self) -> DType;

    /// Copies data from host memory to device.
    fn copy_from_host(&mut self, data: &[u8]);

    /// Copies data from device to host memory.
    fn copy_to_host(&self) -> Vec<u8>;

    /// Returns self as Any for downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns self as mutable Any for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// A compiled kernel ready for execution.
pub trait CompiledKernel: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

/// A device that can execute tensor operations.
pub trait Device: Send + Sync {
    /// Returns the device name.
    fn name(&self) -> &str;

    /// Allocates a buffer on this device.
    fn alloc(&self, numel: usize, dtype: DType) -> Result<Box<dyn Buffer>>;

    /// Executes a UOp graph and returns the result buffer.
    fn realize(&self, uop: &UOp, buffers: &mut BufferMap) -> Result<Arc<dyn Buffer>>;
}

/// Thread-safe map from buffer IDs to buffers.
pub struct BufferMap {
    buffers: RwLock<HashMap<usize, Arc<dyn Buffer>>>,
    next_id: RwLock<usize>,
}

impl Default for BufferMap {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferMap {
    pub fn new() -> Self {
        BufferMap {
            buffers: RwLock::new(HashMap::new()),
            next_id: RwLock::new(0),
        }
    }

    /// Inserts a buffer and returns its ID.
    pub fn insert(&self, buffer: Arc<dyn Buffer>) -> usize {
        let mut next_id = self.next_id.write().unwrap();
        let id = *next_id;
        *next_id += 1;
        self.buffers.write().unwrap().insert(id, buffer);
        id
    }

    /// Gets a buffer by ID.
    pub fn get(&self, id: usize) -> Option<Arc<dyn Buffer>> {
        self.buffers.read().unwrap().get(&id).cloned()
    }

    /// Removes a buffer by ID.
    pub fn remove(&self, id: usize) -> Option<Arc<dyn Buffer>> {
        self.buffers.write().unwrap().remove(&id)
    }
}

/// Global device registry.
static DEVICES: std::sync::OnceLock<RwLock<HashMap<String, Arc<dyn Device>>>> =
    std::sync::OnceLock::new();

fn devices() -> &'static RwLock<HashMap<String, Arc<dyn Device>>> {
    DEVICES.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Registers a device.
pub fn register_device(device: Arc<dyn Device>) {
    let name = device.name().to_string();
    devices().write().unwrap().insert(name, device);
}

/// Gets a device by name.
pub fn get_device(name: &str) -> Option<Arc<dyn Device>> {
    devices().read().unwrap().get(name).cloned()
}

/// Gets the default device.
pub fn default_device() -> Arc<dyn Device> {
    get_device("CPU").expect("CPU device must be registered")
}

/// Lists all registered device names.
pub fn list_devices() -> Vec<String> {
    devices().read().unwrap().keys().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_map() {
        // Basic buffer map test - actual buffer tests will be in runtime
        let map = BufferMap::new();
        assert!(map.get(0).is_none());
    }
}

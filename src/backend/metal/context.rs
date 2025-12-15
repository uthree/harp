//! Metal native context

use crate::backend::traits::NativeContext;
use metal::{CommandQueue, Device};
use std::sync::Arc;

/// Error type for Metal native operations
#[derive(Debug, Clone)]
pub struct MetalNativeError(String);

impl std::fmt::Display for MetalNativeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metal error: {}", self.0)
    }
}

impl std::error::Error for MetalNativeError {}

impl From<String> for MetalNativeError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for MetalNativeError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Metal native context
///
/// Holds the Metal device and command queue.
#[derive(Clone)]
pub struct MetalNativeContext {
    pub(crate) device: Device,
    pub(crate) command_queue: Arc<CommandQueue>,
}

impl NativeContext for MetalNativeContext {
    type Error = MetalNativeError;

    fn new() -> Result<Self, Self::Error> {
        Self::with_device(0)
    }

    fn with_device(device_index: usize) -> Result<Self, Self::Error> {
        // Get all devices
        let devices = Device::all();
        if devices.is_empty() {
            return Err("No Metal devices found".into());
        }

        // Select the requested device
        let device = devices.get(device_index).cloned().ok_or_else(|| {
            MetalNativeError::from(format!(
                "Device index {} out of range (available: {})",
                device_index,
                devices.len()
            ))
        })?;

        // Create command queue
        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue: Arc::new(command_queue),
        })
    }

    fn is_available() -> bool {
        Device::system_default().is_some()
    }

    fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}

impl MetalNativeContext {
    /// Get the Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the Metal command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// List all available Metal devices
    pub fn list_devices() -> Vec<String> {
        Device::all().iter().map(|d| d.name().to_string()).collect()
    }

    /// Create a context using the system default device
    pub fn system_default() -> Result<Self, MetalNativeError> {
        let device = Device::system_default()
            .ok_or_else(|| MetalNativeError::from("No default Metal device found"))?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue: Arc::new(command_queue),
        })
    }
}

// Safety: Metal device and command queue are thread-safe
unsafe impl Send for MetalNativeContext {}
unsafe impl Sync for MetalNativeContext {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_is_available() {
        let available = MetalNativeContext::is_available();
        println!("Metal available: {}", available);
    }

    #[test]
    fn test_metal_context_creation() {
        if !MetalNativeContext::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalNativeContext::new();
        assert!(
            context.is_ok(),
            "Failed to create context: {:?}",
            context.err()
        );

        let context = context.unwrap();
        println!("Device name: {}", context.device_name());
    }

    #[test]
    fn test_metal_list_devices() {
        if !MetalNativeContext::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let devices = MetalNativeContext::list_devices();
        println!("Available Metal devices: {:?}", devices);
        assert!(!devices.is_empty());
    }
}

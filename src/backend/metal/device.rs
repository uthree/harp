//! Metal native device

use crate::backend::traits::Device;
use metal::{CommandQueue, Device as MtlDevice};
use std::sync::Arc;

/// Error type for Metal native operations
#[derive(Debug, Clone)]
pub struct MetalError(String);

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Metal error: {}", self.0)
    }
}

impl std::error::Error for MetalError {}

impl From<String> for MetalError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for MetalError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Metal native device
///
/// Holds the Metal device and command queue.
#[derive(Clone)]
pub struct MetalDevice {
    pub(crate) device: MtlDevice,
    pub(crate) command_queue: Arc<CommandQueue>,
}

impl Device for MetalDevice {
    type Error = MetalError;

    fn new() -> Result<Self, Self::Error> {
        Self::with_device(0)
    }

    fn with_device(device_index: usize) -> Result<Self, Self::Error> {
        // Get all devices
        let devices = MtlDevice::all();
        if devices.is_empty() {
            return Err("No Metal devices found".into());
        }

        // Select the requested device
        let device = devices.get(device_index).cloned().ok_or_else(|| {
            MetalError::from(format!(
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
        MtlDevice::system_default().is_some()
    }

    fn device_name(&self) -> String {
        self.device.name().to_string()
    }
}

impl MetalDevice {
    /// Get the Metal device
    pub fn device(&self) -> &MtlDevice {
        &self.device
    }

    /// Get the Metal command queue
    pub fn command_queue(&self) -> &CommandQueue {
        &self.command_queue
    }

    /// List all available Metal devices
    pub fn list_devices() -> Vec<String> {
        MtlDevice::all()
            .iter()
            .map(|d| d.name().to_string())
            .collect()
    }

    /// Create a device using the system default device
    pub fn system_default() -> Result<Self, MetalError> {
        let device = MtlDevice::system_default()
            .ok_or_else(|| MetalError::from("No default Metal device found"))?;

        let command_queue = device.new_command_queue();

        Ok(Self {
            device,
            command_queue: Arc::new(command_queue),
        })
    }
}

// Safety: Metal device and command queue are thread-safe
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_is_available() {
        let available = MetalDevice::is_available();
        println!("Metal available: {}", available);
    }

    #[test]
    fn test_metal_context_creation() {
        if !MetalDevice::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let context = MetalDevice::new();
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
        if !MetalDevice::is_available() {
            println!("Metal not available, skipping test");
            return;
        }

        let devices = MetalDevice::list_devices();
        println!("Available Metal devices: {:?}", devices);
        assert!(!devices.is_empty());
    }
}

//! Backend implementations for Harp
//!
//! This module contains GPU/CPU backend implementations that are enabled
//! via feature flags.

/// C code generation backend (feature: c)
#[cfg(feature = "c")]
pub mod c;

/// OpenCL GPU backend (feature: opencl)
#[cfg(feature = "opencl")]
pub mod opencl;

/// Metal GPU backend (feature: metal, macOS only)
#[cfg(all(feature = "metal", target_os = "macos"))]
pub mod metal;

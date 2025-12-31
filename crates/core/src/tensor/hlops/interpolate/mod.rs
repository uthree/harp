//! Interpolation operations (hlops)
//!
//! Provides interpolation methods for resizing tensors.
//! Composed from primops: arange, floor, cast, gather.
//!
//! ## Supported modes
//! - `nearest`: Nearest-neighbor interpolation (nearest1d, nearest2d, nearest3d)
//! - `linear`: Linear interpolation (linear1d, bilinear2d, trilinear3d)
//!
//! ## Supported dimensions
//! - 1D: `Dim3` (N, C, W) - e.g., audio signals
//! - 2D: `Dim4` (N, C, H, W) - e.g., images
//! - 3D: `Dim5` (N, C, D, H, W) - e.g., video/volumetric data

mod linear;
mod nearest;

// Methods are implemented directly on Tensor types,
// no re-exports needed

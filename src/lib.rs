//! Harp: High-level Array Processor
//!
//! このクレートは`harp-core`のre-exportです。
//! 実装の詳細は`harp-core`クレートを参照してください。
//!
//! # バックエンド
//!
//! バックエンドはfeature flagで有効化します:
//! - `opencl`: OpenCLバックエンド（`harp::backend::opencl`）
//! - `metal`: Metalバックエンド（`harp::backend::metal`、macOSのみ）
//!
//! ```toml
//! [dependencies]
//! harp = { version = "0.1", features = ["opencl"] }
//! ```

pub use harp_core::*;

/// バックエンドモジュール
///
/// 共通トレイトは常に利用可能です。
/// バックエンド固有の実装はfeature flagで有効化されます。
pub mod backend {
    pub use harp_core::backend::*;

    /// OpenCLバックエンド（`opencl` feature required）
    #[cfg(feature = "opencl")]
    pub mod opencl {
        pub use harp_backend_opencl::*;
    }

    /// Metalバックエンド（`metal` feature required、macOSのみ）
    #[cfg(feature = "metal")]
    pub mod metal {
        pub use harp_backend_metal::*;
    }
}

//! 畳み込み操作のためのモジュール
//!
//! conv/fold/unfold操作で共通して使用されるパラメータと操作を提供します。

mod ops;
mod params;

// ConvParamsは内部使用のみ
pub(crate) use params::ConvParams;
// IntoSpatialParamsはpublic API（unfold/fold/convで使用）
pub use params::IntoSpatialParams;

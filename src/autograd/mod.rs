//! 自動微分機能
//!
//! PyTorchライクなAPIで自動微分を提供します。
//! 計算グラフから勾配計算グラフを生成し、既存のコンパイルパイプラインで実行します。

mod backward;
mod grad_fn;
mod hlops; // 高レベル演算（他の演算の組み合わせで実現）
mod tensor;

pub use grad_fn::GradFn;
pub use tensor::Tensor;

#[cfg(test)]
mod tests;

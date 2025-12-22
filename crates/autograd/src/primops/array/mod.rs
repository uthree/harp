//! 配列演算 (Array Operations)
//!
//! 多次元配列の構造を変更する演算を定義します。
//! - 縮約・拡張演算（Sum, Prod, Max, Expand）
//! - 次元操作（Squeeze, Unsqueeze）
//! - 軸順序変更（Permute）
//! - 形状変更（Reshape）
//! - 線形代数演算（Matmul）
//! - 形状情報（Shape, Ndim）
//! - 初期化（Zeros, Ones）

pub mod dim;
pub mod initialization;
pub mod linalg;
pub mod permute;
pub mod reduce;
pub mod reshape;
pub mod shape;

// 縮約・拡張演算
pub use reduce::{Expand, ExpandBackward, Max, MaxBackward, Prod, ProdBackward, Sum, SumBackward};

// 次元操作
pub use dim::{Squeeze, SqueezeBackward, Unsqueeze, UnsqueezeBackward};

// 軸順序変更
pub use permute::{Permute, PermuteBackward, inverse_permutation};

// 形状変更
pub use reshape::{Reshape, ReshapeBackward};

// 線形代数
pub use linalg::{Matmul, MatmulBackward};

// 形状
pub use shape::{Ndim, Shape};

// 初期化
pub use initialization::{Ones, Zeros};

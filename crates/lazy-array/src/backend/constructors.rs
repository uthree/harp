//! 配列生成メソッド
//!
//! zeros, ones, full などの配列生成メソッドを提供します。
//! マクロにより f32/i32 の重複を排除しています。

use super::{ArrayElement, LazyArray};
use crate::device::Device;
use crate::dim::Dimension;
use crate::generators::IntoShape;
use harp_core::graph::GraphNode;

/// 数値型用の生成メソッド実装マクロ
///
/// 新しい数値型をサポートする際は、このマクロを1回呼び出すだけでOK。
macro_rules! impl_constructors {
    ($type:ty, $zero:expr, $one:expr) => {
        impl<D: Dimension> LazyArray<$type, D> {
            /// ゼロで初期化された配列を生成（遅延）
            pub fn zeros<S: IntoShape>(shape: S) -> Self {
                Self::zeros_on(shape, Device::default_device())
            }

            /// 指定デバイスでゼロ配列を生成（遅延）
            pub fn zeros_on<S: IntoShape>(shape: S, device: Device) -> Self {
                Self::full_on(shape, $zero, device)
            }

            /// 1で初期化された配列を生成（遅延）
            pub fn ones<S: IntoShape>(shape: S) -> Self {
                Self::ones_on(shape, Device::default_device())
            }

            /// 指定デバイスで1配列を生成（遅延）
            pub fn ones_on<S: IntoShape>(shape: S, device: Device) -> Self {
                Self::full_on(shape, $one, device)
            }

            /// 指定値で初期化された配列を生成（遅延）
            pub fn full<S: IntoShape>(shape: S, value: $type) -> Self {
                Self::full_on(shape, value, Device::default_device())
            }

            /// 指定デバイスで指定値配列を生成（遅延）
            ///
            /// constant(value) でスカラーを作成し、unsqueeze + broadcast_to + contiguous で
            /// 指定形状の配列に展開します。
            pub fn full_on<S: IntoShape>(shape: S, value: $type, device: Device) -> Self {
                use harp_core::graph::shape::Expr;

                let shape_vec = shape.into_shape();
                if shape_vec.is_empty() {
                    // スカラーの場合
                    let node = GraphNode::constant(value);
                    return Self::from_node(node, shape_vec, device);
                }

                // constant(value) でスカラーを作成
                let mut node = GraphNode::constant(value);

                // 目標の次元数分 unsqueeze して [1, 1, ..., 1] の形状にする
                for _ in 0..shape_vec.len() {
                    let new_view = node.view.clone().unsqueeze(0);
                    node = node.view(new_view);
                }

                // broadcast_to で目標形状に拡張
                let target_shape: Vec<Expr> =
                    shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
                node = node.broadcast_to(target_shape);

                // contiguous でメモリレイアウトを実体化
                node = node.contiguous();

                Self::from_node(node, shape_vec, device)
            }

            /// 連番配列を生成（遅延）
            pub fn arange(size: usize) -> Self {
                Self::arange_on(size, Device::default_device())
            }

            /// 指定デバイスで連番配列を生成（遅延）
            pub fn arange_on(size: usize, device: Device) -> Self {
                // TODO: arangeノードを実装
                let node = GraphNode::constant($zero);
                Self::from_node(node, vec![size], device)
            }

            /// 入力配列と同じ形状のゼロ配列を生成
            pub fn zeros_like<T2: ArrayElement>(other: &LazyArray<T2, D>) -> Self {
                Self::zeros_on(other.shape().to_vec(), other.device())
            }

            /// 入力配列と同じ形状の1配列を生成
            pub fn ones_like<T2: ArrayElement>(other: &LazyArray<T2, D>) -> Self {
                Self::ones_on(other.shape().to_vec(), other.device())
            }
        }
    };
}

// 各数値型の実装
impl_constructors!(f32, 0.0, 1.0);
impl_constructors!(i32, 0, 1);

// ============================================================================
// 乱数生成（f32のみ）
// ============================================================================

impl<D: Dimension> LazyArray<f32, D> {
    /// 一様乱数 [0, 1) で初期化された配列を生成（遅延）
    pub fn rand<S: IntoShape>(shape: S) -> Self {
        Self::rand_on(shape, Device::default_device())
    }

    /// 指定デバイスで一様乱数配列を生成（遅延）
    pub fn rand_on<S: IntoShape>(shape: S, device: Device) -> Self {
        use harp_core::graph::shape::Expr;

        let shape_vec = shape.into_shape();
        let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
        let node = GraphNode::rand(shape_exprs);
        Self::from_node(node, shape_vec, device)
    }

    /// 入力配列と同じ形状の一様乱数配列を生成
    pub fn rand_like<T2: ArrayElement>(other: &LazyArray<T2, D>) -> Self {
        Self::rand_on(other.shape().to_vec(), other.device())
    }

    /// 標準正規分布 N(0, 1) で初期化された配列を生成（遅延）
    pub fn randn<S: IntoShape>(shape: S) -> Self {
        Self::randn_on(shape, Device::default_device())
    }

    /// 指定デバイスで標準正規分布配列を生成（遅延）
    pub fn randn_on<S: IntoShape>(shape: S, device: Device) -> Self {
        use harp_core::graph::shape::Expr;

        let shape_vec = shape.into_shape();
        let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
        let node = GraphNode::randn(shape_exprs);
        Self::from_node(node, shape_vec, device)
    }

    /// 入力配列と同じ形状の標準正規分布配列を生成
    pub fn randn_like<T2: ArrayElement>(other: &LazyArray<T2, D>) -> Self {
        Self::randn_on(other.shape().to_vec(), other.device())
    }
}

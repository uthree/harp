//! 配列生成関数
//!
//! `zeros`, `ones`, `full`, `arange`などの配列生成関数を提供します。

use crate::array::{Array, ArrayElement};
use crate::context::ExecutionContext;
use crate::dim::Dimension;
use harp_core::backend::pipeline::KernelSourceRenderer;
use harp_core::backend::{Buffer, Compiler, Device, Kernel};
use harp_core::graph::shape::Expr;
use harp_core::graph::{DType, GraphNode};
use std::sync::Arc;

// ============================================================================
// IntoShape トレイト - 形状指定を柔軟に
// ============================================================================

/// 形状に変換可能な型
///
/// 配列やスライスから形状を生成できるようにします。
pub trait IntoShape {
    /// 形状のベクタに変換
    fn into_shape(self) -> Vec<usize>;

    /// 形状のExprベクタに変換
    fn into_shape_exprs(self) -> Vec<Expr>
    where
        Self: Sized,
    {
        self.into_shape()
            .into_iter()
            .map(|s| Expr::from(s as isize))
            .collect()
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

// 単一次元用
impl IntoShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

// タプル実装
impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

// ============================================================================
// 生成関数
// ============================================================================

/// ゼロで初期化された配列を生成
///
/// # 例
/// ```ignore
/// let arr: Array2<f32, ...> = zeros(ctx.clone(), [3, 4]);
/// ```
pub fn zeros<D, R, Dev, Comp, Buf, S>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    shape: S,
) -> Array<f32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
    S: IntoShape,
{
    full_f32(ctx, shape, 0.0)
}

/// 1で初期化された配列を生成
///
/// # 例
/// ```ignore
/// let arr: Array2<f32, ...> = ones(ctx.clone(), [3, 4]);
/// ```
pub fn ones<D, R, Dev, Comp, Buf, S>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    shape: S,
) -> Array<f32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
    S: IntoShape,
{
    full_f32(ctx, shape, 1.0)
}

/// 指定した値で初期化されたf32配列を生成
///
/// # 例
/// ```ignore
/// let arr: Array2<f32, ...> = full_f32(ctx.clone(), [3, 4], 3.14);
/// ```
pub fn full_f32<D, R, Dev, Comp, Buf, S>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    shape: S,
    value: f32,
) -> Array<f32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
    S: IntoShape,
{
    let shape_vec = shape.into_shape();
    let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
    let ndim = shape_vec.len();

    // スカラー定数を作成
    let scalar = GraphNode::constant(value);

    // 次元に合わせてreshape（スカラー→[1, 1, ...]）
    let ones_shape: Vec<Expr> = (0..ndim).map(|_| Expr::from(1)).collect();
    let reshaped = if ndim > 0 {
        scalar.reshape(ones_shape)
    } else {
        // スカラーのまま
        scalar
    };

    // broadcast_toで目標形状に拡張
    let node = if ndim > 0 {
        reshaped.broadcast_to(shape_exprs)
    } else {
        reshaped
    };

    Array::from_graph_node(ctx, node, shape_vec)
}

/// 指定した値で初期化されたi32配列を生成
///
/// # 例
/// ```ignore
/// let arr: Array2<i32, ...> = full_i32(ctx.clone(), [3, 4], 42);
/// ```
pub fn full_i32<D, R, Dev, Comp, Buf, S>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    shape: S,
    value: i32,
) -> Array<i32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
    S: IntoShape,
{
    let shape_vec = shape.into_shape();
    let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
    let ndim = shape_vec.len();

    // スカラー定数を作成（isizeとして、後でキャスト）
    let scalar = GraphNode::constant(value as isize);

    // 次元に合わせてreshape（スカラー→[1, 1, ...]）
    let ones_shape: Vec<Expr> = (0..ndim).map(|_| Expr::from(1)).collect();
    let reshaped = if ndim > 0 {
        scalar.reshape(ones_shape)
    } else {
        scalar
    };

    // broadcast_toで目標形状に拡張
    let node = if ndim > 0 {
        reshaped.broadcast_to(shape_exprs).cast(DType::I32)
    } else {
        reshaped.cast(DType::I32)
    };

    Array::from_graph_node(ctx, node, shape_vec)
}

/// zeros_i32: i32型のゼロ配列を生成
pub fn zeros_i32<D, R, Dev, Comp, Buf, S>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    shape: S,
) -> Array<i32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
    S: IntoShape,
{
    full_i32(ctx, shape, 0)
}

/// ones_i32: i32型の1配列を生成
pub fn ones_i32<D, R, Dev, Comp, Buf, S>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    shape: S,
) -> Array<i32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
    S: IntoShape,
{
    full_i32(ctx, shape, 1)
}

/// 連番配列 [0, 1, 2, ..., size-1] を生成（i32型）
///
/// # 例
/// ```ignore
/// let arr: Array1<i32, ...> = arange(ctx.clone(), 10);
/// ```
pub fn arange<D, R, Dev, Comp, Buf>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    size: usize,
) -> Array<i32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    let node = GraphNode::arange(size as isize);
    Array::from_graph_node(ctx, node, vec![size])
}

/// 連番配列 [0, 1, 2, ..., size-1] を生成（f32型）
///
/// # 例
/// ```ignore
/// let arr: Array1<f32, ...> = arange_f32(ctx.clone(), 10);
/// ```
pub fn arange_f32<D, R, Dev, Comp, Buf>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    size: usize,
) -> Array<f32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    let node = GraphNode::arange(size as isize).cast(DType::F32);
    Array::from_graph_node(ctx, node, vec![size])
}

/// 一様乱数配列 [0, 1) を生成（f32型）
///
/// # 例
/// ```ignore
/// let arr: Array2<f32, ...> = rand(ctx.clone(), [3, 4]);
/// ```
pub fn rand<D, R, Dev, Comp, Buf, S>(
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    shape: S,
) -> Array<f32, D, R, Dev, Comp, Buf>
where
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
    S: IntoShape,
{
    let shape_vec = shape.into_shape();
    let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
    let node = GraphNode::rand(shape_exprs);
    Array::from_graph_node(ctx, node, shape_vec)
}

// ============================================================================
// zeros_like, ones_like
// ============================================================================

/// 入力配列と同じ形状のゼロ配列を生成
pub fn zeros_like<T, D, R, Dev, Comp, Buf>(
    arr: &Array<T, D, R, Dev, Comp, Buf>,
) -> Array<f32, D, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    zeros(arr.context().clone(), arr.shape().to_vec())
}

/// 入力配列と同じ形状の1配列を生成
pub fn ones_like<T, D, R, Dev, Comp, Buf>(
    arr: &Array<T, D, R, Dev, Comp, Buf>,
) -> Array<f32, D, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    ones(arr.context().clone(), arr.shape().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_shape_array() {
        let shape: Vec<usize> = [3, 4].into_shape();
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    fn test_into_shape_vec() {
        let shape: Vec<usize> = vec![2, 3, 4].into_shape();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_into_shape_tuple() {
        let shape: Vec<usize> = (5, 6).into_shape();
        assert_eq!(shape, vec![5, 6]);

        let shape3: Vec<usize> = (2, 3, 4).into_shape();
        assert_eq!(shape3, vec![2, 3, 4]);
    }

    #[test]
    fn test_into_shape_single() {
        let shape: Vec<usize> = 10usize.into_shape();
        assert_eq!(shape, vec![10]);
    }

    #[test]
    fn test_into_shape_exprs() {
        let exprs = [3usize, 4].into_shape_exprs();
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0].as_const(), Some(3));
        assert_eq!(exprs[1].as_const(), Some(4));
    }
}

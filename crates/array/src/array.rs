//! Array構造体 - 遅延評価を透過的に扱う配列
//!
//! 単一の`Array<T, D, B>`型で、未評価（計算グラフのみ）と評価済み（バッファあり）の
//! 両方の状態を透過的に扱います。

use crate::backend::Backend;
use crate::context::ExecutionContext;
use crate::dim::{DimDyn, Dimension, IntoDyn};
use crate::generators::IntoShape;
use harp_core::ast::DType as AstDType;
use harp_core::backend::Buffer;
use harp_core::graph::shape::Expr;
use harp_core::graph::{DType, GraphNode};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// エラー型
// ============================================================================

/// 配列操作エラー
#[derive(Debug, Error)]
pub enum ArrayError {
    /// 形状不一致
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// 無効な軸
    #[error("invalid axis {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    /// 次元不一致
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// 未評価状態でのバッファアクセス
    #[error("array is not materialized yet")]
    NotMaterialized,

    /// コンテキストエラー
    #[error("context error: {0}")]
    Context(String),

    /// バッファエラー
    #[error("buffer error: {0}")]
    Buffer(String),
}

// ============================================================================
// ArrayElement - 配列要素トレイト
// ============================================================================

/// 配列要素として使用可能な型
pub trait ArrayElement: Clone + Send + Sync + 'static {
    /// 対応するDType
    fn dtype() -> DType;

    /// デフォルト値（ゼロ相当）
    fn zero() -> Self;

    /// 単位元（1相当）
    fn one() -> Self;
}

impl ArrayElement for f32 {
    fn dtype() -> DType {
        DType::F32
    }
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl ArrayElement for i32 {
    fn dtype() -> DType {
        DType::I32
    }
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

impl ArrayElement for bool {
    fn dtype() -> DType {
        DType::Bool
    }
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
}

// ============================================================================
// ArrayState - 配列の内部状態
// ============================================================================

/// 配列の内部状態（遅延 or 評価済み）
#[derive(Clone)]
pub enum ArrayState<Buf> {
    /// 未評価：計算グラフのみ保持
    Lazy {
        /// 計算グラフのルートノード
        graph_node: GraphNode,
    },
    /// 評価済み：バッファにデータが存在
    Materialized {
        /// データバッファ
        buffer: Arc<Buf>,
        /// 元のグラフノード（再計算用）
        graph_node: Option<GraphNode>,
    },
}

impl<Buf> fmt::Debug for ArrayState<Buf> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayState::Lazy { .. } => write!(f, "Lazy"),
            ArrayState::Materialized { .. } => write!(f, "Materialized"),
        }
    }
}

// ============================================================================
// Array - メイン構造体
// ============================================================================

/// 配列（遅延評価を透過的に扱う）
///
/// `Array<T, D, B>`は、計算グラフとバッファの両方を扱える統一的な配列型です。
/// 演算時はグラフを構築するだけで、実際の計算は必要になるまで遅延されます。
///
/// # 型パラメータ
/// - `T`: 要素の型（f32, i32, bool）
/// - `D`: 次元（Dim0, Dim1, Dim2, ..., DimDyn）
/// - `B`: バックエンド（Backendトレイトを実装）
///
/// # 例
/// ```ignore
/// use harp_array::prelude::*;
///
/// // 配列の作成（Lazy状態）
/// let a: Array<f32, Dim2, MyBackend> = Array::zeros([100, 100]);
/// let b: Array<f32, Dim2, MyBackend> = Array::ones([100, 100]);
///
/// // 演算（グラフ構築のみ、計算はまだ）
/// let c = &a + &b;
///
/// // データ取得時に計算が実行される
/// let data = c.to_vec()?;
/// ```
pub struct Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    /// 実行コンテキストへの参照
    ctx: Arc<ExecutionContext<B::Renderer, B::Device, B::Compiler, B::Buffer>>,
    /// 内部状態（遅延評価のため内部可変性を使用）
    state: RefCell<ArrayState<B::Buffer>>,
    /// 形状
    shape: Vec<usize>,
    /// 型マーカー
    _marker: PhantomData<(T, D)>,
}

impl<T, D, B> Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    /// 計算グラフノードから遅延配列を作成
    pub fn from_graph_node(graph_node: GraphNode, shape: Vec<usize>) -> Self {
        Self {
            ctx: B::global_context(),
            state: RefCell::new(ArrayState::Lazy { graph_node }),
            shape,
            _marker: PhantomData,
        }
    }

    /// 計算グラフノードから遅延配列を作成（コンテキスト指定）
    pub fn from_graph_node_with_ctx(
        ctx: Arc<ExecutionContext<B::Renderer, B::Device, B::Compiler, B::Buffer>>,
        graph_node: GraphNode,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            ctx,
            state: RefCell::new(ArrayState::Lazy { graph_node }),
            shape,
            _marker: PhantomData,
        }
    }

    /// バッファから評価済み配列を作成
    pub fn from_buffer(buffer: Arc<B::Buffer>, shape: Vec<usize>) -> Self {
        Self {
            ctx: B::global_context(),
            state: RefCell::new(ArrayState::Materialized {
                buffer,
                graph_node: None,
            }),
            shape,
            _marker: PhantomData,
        }
    }

    /// 形状を取得
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 次元数を取得
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// 要素数を取得
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// 配列が空かどうか
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// データ型を取得
    pub fn dtype(&self) -> DType {
        T::dtype()
    }

    /// 現在評価済みかどうか
    pub fn is_materialized(&self) -> bool {
        matches!(&*self.state.borrow(), ArrayState::Materialized { .. })
    }

    /// 現在遅延状態かどうか
    pub fn is_lazy(&self) -> bool {
        matches!(&*self.state.borrow(), ArrayState::Lazy { .. })
    }

    /// 計算グラフノードを取得（内部用）
    #[allow(dead_code)]
    pub(crate) fn graph_node(&self) -> GraphNode {
        match &*self.state.borrow() {
            ArrayState::Lazy { graph_node } => graph_node.clone(),
            ArrayState::Materialized { graph_node, .. } => graph_node
                .clone()
                .unwrap_or_else(|| panic!("graph_node not available for materialized array")),
        }
    }

    /// 実行コンテキストへの参照を取得
    pub fn context(
        &self,
    ) -> &Arc<ExecutionContext<B::Renderer, B::Device, B::Compiler, B::Buffer>> {
        &self.ctx
    }

    /// 明示的に評価を実行
    ///
    /// 計算グラフをコンパイル・実行し、結果をバッファに保存します。
    /// 既に評価済みの場合は何もしません。
    pub fn eval(&self) -> Result<(), ArrayError> {
        if self.is_materialized() {
            return Ok(());
        }

        let graph_node = match &*self.state.borrow() {
            ArrayState::Lazy { graph_node } => graph_node.clone(),
            ArrayState::Materialized { .. } => return Ok(()),
        };

        let compiled = self
            .ctx
            .compile(&graph_node)
            .map_err(|e| ArrayError::Context(e.to_string()))?;

        let ast_dtype = graph_dtype_to_ast(&T::dtype());

        // TODO: 入力バッファを持つグラフのサポート
        let output_buffer = self
            .ctx
            .execute(&compiled, HashMap::new(), self.shape.clone(), ast_dtype)
            .map_err(|e| ArrayError::Context(e.to_string()))?;

        *self.state.borrow_mut() = ArrayState::Materialized {
            buffer: Arc::new(output_buffer),
            graph_node: Some(graph_node),
        };

        Ok(())
    }

    /// データをホストメモリに読み出す
    ///
    /// 評価されていない場合は自動的に評価を実行します。
    pub fn to_vec(&self) -> Result<Vec<T>, ArrayError> {
        self.eval()?;

        let state = self.state.borrow();
        match &*state {
            ArrayState::Materialized { buffer, .. } => buffer
                .read_vec::<T>()
                .map_err(|e| ArrayError::Buffer(e.to_string())),
            ArrayState::Lazy { .. } => {
                unreachable!("Array should be materialized after eval()")
            }
        }
    }
}

// ============================================================================
// 生成メソッド（型ごとの実装）
// ============================================================================

impl<D, B> Array<f32, D, B>
where
    D: Dimension,
    B: Backend,
{
    /// ゼロで初期化された配列を生成
    ///
    /// # 例
    /// ```ignore
    /// let arr: Array<f32, Dim2, MyBackend> = Array::zeros([3, 4]);
    /// ```
    pub fn zeros<S: IntoShape>(shape: S) -> Self {
        Self::full(shape, 0.0)
    }

    /// 1で初期化された配列を生成
    ///
    /// # 例
    /// ```ignore
    /// let arr: Array<f32, Dim2, MyBackend> = Array::ones([3, 4]);
    /// ```
    pub fn ones<S: IntoShape>(shape: S) -> Self {
        Self::full(shape, 1.0)
    }

    /// 指定値で初期化された配列を生成
    pub fn full<S: IntoShape>(shape: S, value: f32) -> Self {
        let shape_vec = shape.into_shape();
        let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
        let ndim = shape_vec.len();

        let scalar = GraphNode::constant(value);

        let ones_shape: Vec<Expr> = (0..ndim).map(|_| Expr::from(1)).collect();
        let reshaped = if ndim > 0 {
            scalar.reshape(ones_shape)
        } else {
            scalar
        };

        let node = if ndim > 0 {
            reshaped.broadcast_to(shape_exprs)
        } else {
            reshaped
        };

        Self::from_graph_node(node, shape_vec)
    }

    /// 連番配列 [0.0, 1.0, 2.0, ...] を生成
    pub fn arange(size: usize) -> Self {
        let node = GraphNode::arange(size as isize).cast(DType::F32);
        Self::from_graph_node(node, vec![size])
    }

    /// 一様乱数配列 [0, 1) を生成
    pub fn rand<S: IntoShape>(shape: S) -> Self {
        let shape_vec = shape.into_shape();
        let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
        let node = GraphNode::rand(shape_exprs);
        Self::from_graph_node(node, shape_vec)
    }

    /// 入力配列と同じ形状のゼロ配列を生成
    pub fn zeros_like<T2: ArrayElement>(other: &Array<T2, D, B>) -> Self {
        Self::zeros(other.shape().to_vec())
    }

    /// 入力配列と同じ形状の1配列を生成
    pub fn ones_like<T2: ArrayElement>(other: &Array<T2, D, B>) -> Self {
        Self::ones(other.shape().to_vec())
    }
}

impl<D, B> Array<i32, D, B>
where
    D: Dimension,
    B: Backend,
{
    /// ゼロで初期化された配列を生成
    pub fn zeros<S: IntoShape>(shape: S) -> Self {
        Self::full(shape, 0)
    }

    /// 1で初期化された配列を生成
    pub fn ones<S: IntoShape>(shape: S) -> Self {
        Self::full(shape, 1)
    }

    /// 指定値で初期化された配列を生成
    pub fn full<S: IntoShape>(shape: S, value: i32) -> Self {
        let shape_vec = shape.into_shape();
        let shape_exprs: Vec<Expr> = shape_vec.iter().map(|&s| Expr::from(s as isize)).collect();
        let ndim = shape_vec.len();

        let scalar = GraphNode::constant(value as isize);

        let ones_shape: Vec<Expr> = (0..ndim).map(|_| Expr::from(1)).collect();
        let reshaped = if ndim > 0 {
            scalar.reshape(ones_shape)
        } else {
            scalar
        };

        let node = if ndim > 0 {
            reshaped.broadcast_to(shape_exprs).cast(DType::I32)
        } else {
            reshaped.cast(DType::I32)
        };

        Self::from_graph_node(node, shape_vec)
    }

    /// 連番配列 [0, 1, 2, ...] を生成
    pub fn arange(size: usize) -> Self {
        let node = GraphNode::arange(size as isize);
        Self::from_graph_node(node, vec![size])
    }

    /// 入力配列と同じ形状のゼロ配列を生成
    pub fn zeros_like<T2: ArrayElement>(other: &Array<T2, D, B>) -> Self {
        Self::zeros(other.shape().to_vec())
    }

    /// 入力配列と同じ形状の1配列を生成
    pub fn ones_like<T2: ArrayElement>(other: &Array<T2, D, B>) -> Self {
        Self::ones(other.shape().to_vec())
    }
}

// ============================================================================
// GraphのDType変換
// ============================================================================

/// GraphのDTypeをASTのDTypeに変換
fn graph_dtype_to_ast(dtype: &DType) -> AstDType {
    match dtype {
        DType::F32 => AstDType::F32,
        DType::I32 => AstDType::Int,
        DType::Bool => AstDType::Bool,
        DType::Unknown => AstDType::F32,
    }
}

// ============================================================================
// Clone, Debug, 次元変換
// ============================================================================

impl<T, D, B> Clone for Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    fn clone(&self) -> Self {
        Self {
            ctx: self.ctx.clone(),
            state: self.state.clone(),
            shape: self.shape.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T, D, B> fmt::Debug for Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension,
    B: Backend,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Array")
            .field("shape", &self.shape)
            .field("dtype", &T::dtype())
            .field("state", &*self.state.borrow())
            .finish()
    }
}

// ============================================================================
// 次元変換
// ============================================================================

impl<T, B> Array<T, DimDyn, B>
where
    T: ArrayElement,
    B: Backend,
{
    /// 動的次元から静的次元に変換
    pub fn into_dimensionality<D2: Dimension>(self) -> Result<Array<T, D2, B>, ArrayError> {
        if let Some(expected) = D2::NDIM
            && self.ndim() != expected
        {
            return Err(ArrayError::DimensionMismatch {
                expected,
                actual: self.ndim(),
            });
        }

        Ok(Array {
            ctx: self.ctx,
            state: self.state,
            shape: self.shape,
            _marker: PhantomData,
        })
    }
}

impl<T, D, B> Array<T, D, B>
where
    T: ArrayElement,
    D: Dimension + IntoDyn,
    B: Backend,
{
    /// 静的次元から動的次元に変換
    pub fn into_dyn(self) -> Array<T, DimDyn, B> {
        Array {
            ctx: self.ctx,
            state: self.state,
            shape: self.shape,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_element_dtype() {
        assert_eq!(f32::dtype(), DType::F32);
        assert_eq!(i32::dtype(), DType::I32);
        assert_eq!(bool::dtype(), DType::Bool);
    }

    #[test]
    fn test_array_element_zero_one() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(i32::zero(), 0);
        assert_eq!(i32::one(), 1);
        assert!(!bool::zero());
        assert!(bool::one());
    }
}

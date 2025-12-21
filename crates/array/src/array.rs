//! Array構造体 - 遅延評価を透過的に扱う配列
//!
//! 単一の`Array<T, D>`型で、未評価（計算グラフのみ）と評価済み（バッファあり）の
//! 両方の状態を透過的に扱います。

use crate::context::ExecutionContext;
use crate::dim::{DimDyn, Dimension, IntoDyn};
use harp_core::ast::DType as AstDType;
use harp_core::backend::pipeline::KernelSourceRenderer;
use harp_core::backend::{Buffer, Compiler, Device, Kernel};
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
}

impl ArrayElement for f32 {
    fn dtype() -> DType {
        DType::F32
    }
}

impl ArrayElement for i32 {
    fn dtype() -> DType {
        DType::I32
    }
}

impl ArrayElement for bool {
    fn dtype() -> DType {
        DType::Bool
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
/// `Array<T, D>`は、計算グラフとバッファの両方を扱える統一的な配列型です。
/// 演算時はグラフを構築するだけで、実際の計算は必要になるまで遅延されます。
///
/// # 型パラメータ
/// - `T`: 要素の型（f32, i32, bool）
/// - `D`: 次元（Dim0, Dim1, Dim2, ..., DimDyn）
/// - `R`, `Dev`, `Comp`, `Buf`: バックエンド関連の型パラメータ
///
/// # 例
/// ```ignore
/// // 配列の作成（Lazy状態）
/// let a: Array2<f32> = zeros([100, 100]);
/// let b: Array2<f32> = ones([100, 100]);
///
/// // 演算（グラフ構築のみ、計算はまだ）
/// let c = &a + &b;
///
/// // データ取得時に計算が実行される
/// let data = c.to_vec()?;
/// ```
pub struct Array<T, D, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    /// 実行コンテキストへの参照
    ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
    /// 内部状態（遅延評価のため内部可変性を使用）
    state: RefCell<ArrayState<Buf>>,
    /// 形状
    shape: Vec<usize>,
    /// 型マーカー
    _marker: PhantomData<(T, D)>,
}

impl<T, D, R, Dev, Comp, Buf> Array<T, D, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    /// 計算グラフノードから遅延配列を作成
    pub fn from_graph_node(
        ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
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
    pub fn from_buffer(
        ctx: Arc<ExecutionContext<R, Dev, Comp, Buf>>,
        buffer: Arc<Buf>,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            ctx,
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
            ArrayState::Materialized { graph_node, .. } => {
                graph_node.clone().unwrap_or_else(|| {
                    // バッファからの復元が必要な場合
                    // 通常はgraph_nodeが保持されているはず
                    panic!("graph_node not available for materialized array")
                })
            }
        }
    }

    /// 実行コンテキストへの参照を取得
    pub fn context(&self) -> &Arc<ExecutionContext<R, Dev, Comp, Buf>> {
        &self.ctx
    }

    /// 明示的に評価を実行
    ///
    /// 計算グラフをコンパイル・実行し、結果をバッファに保存します。
    /// 既に評価済みの場合は何もしません。
    ///
    /// # エラー
    ///
    /// コンパイルまたは実行に失敗した場合はエラーを返します。
    pub fn eval(&self) -> Result<(), ArrayError> {
        // 既に評価済みならスキップ
        if self.is_materialized() {
            return Ok(());
        }

        // グラフノードを取得
        let graph_node = match &*self.state.borrow() {
            ArrayState::Lazy { graph_node } => graph_node.clone(),
            ArrayState::Materialized { .. } => return Ok(()),
        };

        // コンパイル
        let compiled = self
            .ctx
            .compile(&graph_node)
            .map_err(|e| ArrayError::Context(e.to_string()))?;

        // グラフのDTypeをASTのDTypeに変換
        let ast_dtype = graph_dtype_to_ast(&T::dtype());

        // 実行（入力なし - 定数のみのグラフを想定）
        // TODO: 入力バッファを持つグラフのサポート
        let output_buffer = self
            .ctx
            .execute(&compiled, HashMap::new(), self.shape.clone(), ast_dtype)
            .map_err(|e| ArrayError::Context(e.to_string()))?;

        // 状態を更新
        *self.state.borrow_mut() = ArrayState::Materialized {
            buffer: Arc::new(output_buffer),
            graph_node: Some(graph_node),
        };

        Ok(())
    }

    /// データをホストメモリに読み出す
    ///
    /// 評価されていない場合は自動的に評価を実行します。
    ///
    /// # 例
    ///
    /// ```ignore
    /// let arr = zeros(ctx, [3, 4]);
    /// let data: Vec<f32> = arr.to_vec()?;
    /// assert_eq!(data.len(), 12);
    /// ```
    pub fn to_vec(&self) -> Result<Vec<T>, ArrayError> {
        // 必要なら評価
        self.eval()?;

        // バッファからデータを読み出し
        let state = self.state.borrow();
        match &*state {
            ArrayState::Materialized { buffer, .. } => buffer
                .read_vec::<T>()
                .map_err(|e| ArrayError::Buffer(e.to_string())),
            ArrayState::Lazy { .. } => {
                // evalが成功したはずなのにLazyのままは起こり得ない
                unreachable!("Array should be materialized after eval()")
            }
        }
    }
}

/// GraphのDTypeをASTのDTypeに変換
fn graph_dtype_to_ast(dtype: &DType) -> AstDType {
    match dtype {
        DType::F32 => AstDType::F32,
        DType::I32 => AstDType::Int,
        DType::Bool => AstDType::Bool,
        DType::Unknown => AstDType::F32, // フォールバック
    }
}

impl<T, D, R, Dev, Comp, Buf> Clone for Array<T, D, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
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

impl<T, D, R, Dev, Comp, Buf> fmt::Debug for Array<T, D, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    D: Dimension,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
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

impl<T, R, Dev, Comp, Buf> Array<T, DimDyn, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    /// 動的次元から静的次元に変換
    ///
    /// 次元数が一致しない場合はエラーを返します。
    pub fn into_dimensionality<D2: Dimension>(
        self,
    ) -> Result<Array<T, D2, R, Dev, Comp, Buf>, ArrayError> {
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

impl<T, D, R, Dev, Comp, Buf> Array<T, D, R, Dev, Comp, Buf>
where
    T: ArrayElement,
    D: Dimension + IntoDyn,
    R: KernelSourceRenderer + Clone,
    Dev: Device,
    Comp: Compiler<Dev = Dev>,
    Comp::Kernel: Kernel<Buffer = Buf> + Clone,
    Buf: Buffer<Dev = Dev>,
{
    /// 静的次元から動的次元に変換
    pub fn into_dyn(self) -> Array<T, DimDyn, R, Dev, Comp, Buf> {
        Array {
            ctx: self.ctx,
            state: self.state,
            shape: self.shape,
            _marker: PhantomData,
        }
    }
}

// ============================================================================
// 型エイリアス
// ============================================================================

// 注: 実際の使用時はバックエンド型を指定する必要があります
// 例: type Array2<T> = Array<T, Dim2, MetalRenderer, MetalDevice, MetalCompiler, MetalBuffer>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_element_dtype() {
        assert_eq!(f32::dtype(), DType::F32);
        assert_eq!(i32::dtype(), DType::I32);
        assert_eq!(bool::dtype(), DType::Bool);
    }
}

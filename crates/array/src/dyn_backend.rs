//! 配列型（遅延評価）
//!
//! 実行時にデバイスを選択できる配列型を提供します。
//! 演算は遅延評価され、`eval()`または`to_vec()`呼び出し時に実行されます。

use crate::device::Device;
use crate::dim::Dimension;
use harp_core::graph::{DType, Graph, GraphNode};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use thiserror::Error;

// ============================================================================
// ArrayElement - 配列要素トレイト
// ============================================================================

/// 配列要素として使用可能な型のトレイト
pub trait ArrayElement: Clone + Copy + Default + 'static {
    /// 型のゼロ値
    fn zero() -> Self;

    /// 型の1値
    fn one() -> Self;

    /// データ型名
    fn dtype_name() -> &'static str;

    /// GraphのDTypeに変換
    fn graph_dtype() -> DType;
}

impl ArrayElement for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn dtype_name() -> &'static str {
        "f32"
    }
    fn graph_dtype() -> DType {
        DType::F32
    }
}

impl ArrayElement for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn dtype_name() -> &'static str {
        "i32"
    }
    fn graph_dtype() -> DType {
        DType::I32
    }
}

impl ArrayElement for bool {
    fn zero() -> Self {
        false
    }
    fn one() -> Self {
        true
    }
    fn dtype_name() -> &'static str {
        "bool"
    }
    fn graph_dtype() -> DType {
        DType::Bool
    }
}

// ============================================================================
// ArrayError - 配列エラー型
// ============================================================================

/// 配列演算のエラー型
#[derive(Error, Debug)]
pub enum ArrayError {
    /// 形状の不一致
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// 無効な軸
    #[error("Invalid axis: {axis} for array with {ndim} dimensions")]
    InvalidAxis { axis: usize, ndim: usize },

    /// デバイスが利用不可
    #[error("Device not available: {0}")]
    DeviceNotAvailable(String),

    /// コンパイルエラー
    #[error("Compilation error: {0}")]
    Compilation(String),

    /// 実行エラー
    #[error("Execution error: {0}")]
    Execution(String),

    /// バックエンド未設定
    #[error("No backend available. Enable at least one backend feature: metal or opencl")]
    NoBackend,
}

// ============================================================================
// DynBuffer - 動的バッファ
// ============================================================================

/// 動的バッファ（デバイス非依存）
#[derive(Clone)]
pub enum DynBuffer {
    /// OpenCLバッファ
    #[cfg(feature = "opencl")]
    OpenCL(harp_backend_opencl::OpenCLBuffer),
    /// Metalバッファ
    #[cfg(feature = "metal")]
    Metal(harp_backend_metal::MetalBuffer),
}

impl DynBuffer {
    /// バッファの形状を取得
    pub fn shape(&self) -> &[usize] {
        match self {
            #[cfg(feature = "opencl")]
            DynBuffer::OpenCL(buf) => {
                use harp_core::backend::Buffer;
                buf.shape()
            }
            #[cfg(feature = "metal")]
            DynBuffer::Metal(buf) => {
                use harp_core::backend::Buffer;
                buf.shape()
            }
            #[allow(unreachable_patterns)]
            _ => &[],
        }
    }

    /// バッファからデータを読み出し
    pub fn read_vec<T: Clone + 'static>(&self) -> Result<Vec<T>, ArrayError> {
        match self {
            #[cfg(feature = "opencl")]
            DynBuffer::OpenCL(buf) => {
                use harp_core::backend::Buffer;
                buf.read_vec()
                    .map_err(|e| ArrayError::Execution(e.to_string()))
            }
            #[cfg(feature = "metal")]
            DynBuffer::Metal(buf) => {
                use harp_core::backend::Buffer;
                buf.read_vec()
                    .map_err(|e| ArrayError::Execution(e.to_string()))
            }
            #[allow(unreachable_patterns)]
            _ => Err(ArrayError::NoBackend),
        }
    }
}

// ============================================================================
// ArrayState - 配列の内部状態
// ============================================================================

/// 配列の内部状態
enum ArrayState {
    /// 遅延評価状態: GraphNodeのみ保持
    Lazy { node: GraphNode },
    /// 評価済み状態: バッファにデータが存在
    Materialized { node: GraphNode, buffer: DynBuffer },
}

// ============================================================================
// Array - 配列型
// ============================================================================

/// 配列型（遅延評価）
///
/// 演算はGraphNodeとして構築され、`eval()`または`to_vec()`呼び出し時に
/// 実際の計算が実行されます。
///
/// # 例
///
/// ```ignore
/// use harp_array::prelude::*;
///
/// // 配列の作成（遅延）
/// let a: Array2<f32> = Array2::zeros([3, 4]);
/// let b: Array2<f32> = Array2::ones([3, 4]);
///
/// // 演算（グラフ構築のみ）
/// let c = &a + &b;
///
/// // データ取得時に初めて計算が実行される
/// let data: Vec<f32> = c.to_vec()?;
/// ```
pub struct Array<T: ArrayElement, D: Dimension> {
    /// 内部状態（遅延/評価済み）
    state: Rc<RefCell<ArrayState>>,
    /// ターゲットデバイス
    device: Device,
    /// 形状
    shape: Vec<usize>,
    /// 型マーカー
    _phantom: PhantomData<(T, D)>,
}

impl<T: ArrayElement, D: Dimension> Array<T, D> {
    /// GraphNodeから配列を作成（内部用）
    pub(crate) fn from_node(node: GraphNode, shape: Vec<usize>, device: Device) -> Self {
        Self {
            state: Rc::new(RefCell::new(ArrayState::Lazy { node })),
            device,
            shape,
            _phantom: PhantomData,
        }
    }

    /// ホストデータから配列を作成
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::from_vec_on(data, shape, Device::default_device())
    }

    /// 指定デバイスでホストデータから配列を作成
    pub fn from_vec_on(data: Vec<T>, shape: Vec<usize>, device: Device) -> Self {
        // データをGraphNodeとして表現（Buffer定数として）
        let mut graph = Graph::new();
        let input_name = format!("const_{}", std::ptr::addr_of!(data) as usize);
        let shape_exprs: Vec<usize> = shape.clone();
        let node = graph.input(&input_name, T::graph_dtype(), shape_exprs);

        // TODO: 実際にはデータをバッファに書き込む必要がある
        // 現時点では簡易実装として、後で評価時にデータを使用

        Self {
            state: Rc::new(RefCell::new(ArrayState::Lazy { node })),
            device,
            shape,
            _phantom: PhantomData,
        }
    }

    /// 現在のデバイスを取得
    pub fn device(&self) -> Device {
        self.device
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

    /// 内部のGraphNodeを取得
    pub fn graph_node(&self) -> GraphNode {
        match &*self.state.borrow() {
            ArrayState::Lazy { node } => node.clone(),
            ArrayState::Materialized { node, .. } => node.clone(),
        }
    }

    /// 評価済みかどうか
    pub fn is_materialized(&self) -> bool {
        matches!(&*self.state.borrow(), ArrayState::Materialized { .. })
    }

    /// 別のデバイスに転送
    pub fn to(&self, device: Device) -> Result<Array<T, D>, ArrayError> {
        if !device.is_available() {
            return Err(ArrayError::DeviceNotAvailable(device.name().to_string()));
        }

        // 新しいデバイスで同じグラフノードを持つ配列を作成
        Ok(Array {
            state: Rc::new(RefCell::new(ArrayState::Lazy {
                node: self.graph_node(),
            })),
            device,
            shape: self.shape.clone(),
            _phantom: PhantomData,
        })
    }

    /// 明示的に評価を実行
    pub fn eval(&self) -> Result<(), ArrayError> {
        // 既に評価済みなら何もしない
        if self.is_materialized() {
            return Ok(());
        }

        // TODO: 実際の評価ロジックを実装
        // 1. GraphをPipelineでコンパイル
        // 2. バッファを確保
        // 3. カーネルを実行
        // 4. 結果をMaterialized状態に保存

        Err(ArrayError::Execution(
            "Evaluation not yet implemented. Use to_vec_eager() for eager evaluation.".to_string(),
        ))
    }

    /// データをベクタとして取得（遅延評価を実行）
    pub fn to_vec(&self) -> Result<Vec<T>, ArrayError> {
        self.eval()?;

        match &*self.state.borrow() {
            ArrayState::Materialized { buffer, .. } => buffer.read_vec(),
            ArrayState::Lazy { .. } => Err(ArrayError::Execution(
                "Array not materialized after eval".to_string(),
            )),
        }
    }
}

// ============================================================================
// 生成メソッド（f32）
// ============================================================================

use crate::generators::IntoShape;

impl<D: Dimension> Array<f32, D> {
    /// ゼロで初期化された配列を生成（遅延）
    pub fn zeros<S: IntoShape>(shape: S) -> Self {
        Self::zeros_on(shape, Device::default_device())
    }

    /// 指定デバイスでゼロ配列を生成（遅延）
    pub fn zeros_on<S: IntoShape>(shape: S, device: Device) -> Self {
        Self::full_on(shape, 0.0, device)
    }

    /// 1で初期化された配列を生成（遅延）
    pub fn ones<S: IntoShape>(shape: S) -> Self {
        Self::ones_on(shape, Device::default_device())
    }

    /// 指定デバイスで1配列を生成（遅延）
    pub fn ones_on<S: IntoShape>(shape: S, device: Device) -> Self {
        Self::full_on(shape, 1.0, device)
    }

    /// 指定値で初期化された配列を生成（遅延）
    pub fn full<S: IntoShape>(shape: S, value: f32) -> Self {
        Self::full_on(shape, value, Device::default_device())
    }

    /// 指定デバイスで指定値配列を生成（遅延）
    pub fn full_on<S: IntoShape>(shape: S, value: f32, device: Device) -> Self {
        let shape_vec = shape.into_shape();

        // 定数ノードを作成
        let node = GraphNode::constant(value);

        // TODO: 形状に合わせてブロードキャストするノードを作成
        // 現時点では簡易実装

        Self::from_node(node, shape_vec, device)
    }

    /// 連番配列を生成（遅延）
    pub fn arange(size: usize) -> Self {
        Self::arange_on(size, Device::default_device())
    }

    /// 指定デバイスで連番配列を生成（遅延）
    pub fn arange_on(size: usize, device: Device) -> Self {
        // TODO: arangeノードを実装
        let node = GraphNode::constant(0.0f32);
        Self::from_node(node, vec![size], device)
    }

    /// 入力配列と同じ形状のゼロ配列を生成
    pub fn zeros_like<T2: ArrayElement>(other: &Array<T2, D>) -> Self {
        Self::zeros_on(other.shape().to_vec(), other.device())
    }

    /// 入力配列と同じ形状の1配列を生成
    pub fn ones_like<T2: ArrayElement>(other: &Array<T2, D>) -> Self {
        Self::ones_on(other.shape().to_vec(), other.device())
    }
}

// ============================================================================
// 生成メソッド（i32）
// ============================================================================

impl<D: Dimension> Array<i32, D> {
    /// ゼロで初期化された配列を生成
    pub fn zeros<S: IntoShape>(shape: S) -> Self {
        Self::zeros_on(shape, Device::default_device())
    }

    /// 指定デバイスでゼロ配列を生成
    pub fn zeros_on<S: IntoShape>(shape: S, device: Device) -> Self {
        Self::full_on(shape, 0, device)
    }

    /// 1で初期化された配列を生成
    pub fn ones<S: IntoShape>(shape: S) -> Self {
        Self::ones_on(shape, Device::default_device())
    }

    /// 指定デバイスで1配列を生成
    pub fn ones_on<S: IntoShape>(shape: S, device: Device) -> Self {
        Self::full_on(shape, 1, device)
    }

    /// 指定値で初期化された配列を生成
    pub fn full<S: IntoShape>(shape: S, value: i32) -> Self {
        Self::full_on(shape, value, Device::default_device())
    }

    /// 指定デバイスで指定値配列を生成
    pub fn full_on<S: IntoShape>(shape: S, value: i32, device: Device) -> Self {
        let shape_vec = shape.into_shape();
        let node = GraphNode::constant(value as isize);
        Self::from_node(node, shape_vec, device)
    }

    /// 連番配列を生成
    pub fn arange(size: usize) -> Self {
        Self::arange_on(size, Device::default_device())
    }

    /// 指定デバイスで連番配列を生成
    pub fn arange_on(size: usize, device: Device) -> Self {
        let node = GraphNode::constant(0isize);
        Self::from_node(node, vec![size], device)
    }

    /// 入力配列と同じ形状のゼロ配列を生成
    pub fn zeros_like<T2: ArrayElement>(other: &Array<T2, D>) -> Self {
        Self::zeros_on(other.shape().to_vec(), other.device())
    }

    /// 入力配列と同じ形状の1配列を生成
    pub fn ones_like<T2: ArrayElement>(other: &Array<T2, D>) -> Self {
        Self::ones_on(other.shape().to_vec(), other.device())
    }
}

// ============================================================================
// Clone, Debug
// ============================================================================

impl<T: ArrayElement, D: Dimension> Clone for Array<T, D> {
    fn clone(&self) -> Self {
        Self {
            state: Rc::new(RefCell::new(ArrayState::Lazy {
                node: self.graph_node(),
            })),
            device: self.device,
            shape: self.shape.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T: ArrayElement, D: Dimension> std::fmt::Debug for Array<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("device", &self.device)
            .field("shape", &self.shape)
            .field("materialized", &self.is_materialized())
            .finish()
    }
}

// ============================================================================
// 型エイリアス
// ============================================================================

use crate::dim::{Dim0, Dim1, Dim2, Dim3, Dim4, Dim5, Dim6, DimDyn};

/// 0次元配列（スカラー）
pub type Array0<T> = Array<T, Dim0>;

/// 1次元配列（ベクトル）
pub type Array1<T> = Array<T, Dim1>;

/// 2次元配列（行列）
pub type Array2<T> = Array<T, Dim2>;

/// 3次元配列
pub type Array3<T> = Array<T, Dim3>;

/// 4次元配列
pub type Array4<T> = Array<T, Dim4>;

/// 5次元配列
pub type Array5<T> = Array<T, Dim5>;

/// 6次元配列
pub type Array6<T> = Array<T, Dim6>;

/// 動的次元配列
pub type ArrayD<T> = Array<T, DimDyn>;

// ============================================================================
// 演算子実装
// ============================================================================

use std::ops::{Add, Div, Mul, Neg, Sub};

/// 二項演算の結果形状を計算（ブロードキャスト）
fn broadcast_shapes(shape1: &[usize], shape2: &[usize]) -> Vec<usize> {
    if shape1.is_empty() {
        return shape2.to_vec();
    }
    if shape2.is_empty() {
        return shape1.to_vec();
    }
    if shape1 == shape2 {
        return shape1.to_vec();
    }
    panic!(
        "Shape mismatch for broadcast: {:?} and {:?}",
        shape1, shape2
    );
}

// Add: &Array + &Array
impl<T, D> Add for &Array<T, D>
where
    T: ArrayElement,
    D: Dimension,
{
    type Output = Array<T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = self.graph_node() + rhs.graph_node();
        Array::from_node(result_node, result_shape, self.device)
    }
}

// Sub: &Array - &Array
impl<T, D> Sub for &Array<T, D>
where
    T: ArrayElement,
    D: Dimension,
{
    type Output = Array<T, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = self.graph_node() - rhs.graph_node();
        Array::from_node(result_node, result_shape, self.device)
    }
}

// Mul: &Array * &Array
impl<T, D> Mul for &Array<T, D>
where
    T: ArrayElement,
    D: Dimension,
{
    type Output = Array<T, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = self.graph_node() * rhs.graph_node();
        Array::from_node(result_node, result_shape, self.device)
    }
}

// Div: &Array / &Array
impl<T, D> Div for &Array<T, D>
where
    T: ArrayElement,
    D: Dimension,
{
    type Output = Array<T, D>;

    fn div(self, rhs: Self) -> Self::Output {
        let result_shape = broadcast_shapes(self.shape(), rhs.shape());
        let result_node = self.graph_node() / rhs.graph_node();
        Array::from_node(result_node, result_shape, self.device)
    }
}

// Neg: -&Array
impl<T, D> Neg for &Array<T, D>
where
    T: ArrayElement,
    D: Dimension,
{
    type Output = Array<T, D>;

    fn neg(self) -> Self::Output {
        let result_node = -self.graph_node();
        Array::from_node(result_node, self.shape.clone(), self.device)
    }
}

// スカラー演算 (f32)
impl<D> Add<f32> for &Array<f32, D>
where
    D: Dimension,
{
    type Output = Array<f32, D>;

    fn add(self, rhs: f32) -> Self::Output {
        let result_node = self.graph_node() + GraphNode::constant(rhs);
        Array::from_node(result_node, self.shape.clone(), self.device)
    }
}

impl<D> Sub<f32> for &Array<f32, D>
where
    D: Dimension,
{
    type Output = Array<f32, D>;

    fn sub(self, rhs: f32) -> Self::Output {
        let result_node = self.graph_node() - GraphNode::constant(rhs);
        Array::from_node(result_node, self.shape.clone(), self.device)
    }
}

impl<D> Mul<f32> for &Array<f32, D>
where
    D: Dimension,
{
    type Output = Array<f32, D>;

    fn mul(self, rhs: f32) -> Self::Output {
        let result_node = self.graph_node() * GraphNode::constant(rhs);
        Array::from_node(result_node, self.shape.clone(), self.device)
    }
}

impl<D> Div<f32> for &Array<f32, D>
where
    D: Dimension,
{
    type Output = Array<f32, D>;

    fn div(self, rhs: f32) -> Self::Output {
        let result_node = self.graph_node() / GraphNode::constant(rhs);
        Array::from_node(result_node, self.shape.clone(), self.device)
    }
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::Dim2;

    #[test]
    fn test_array_creation() {
        let arr = <Array<f32, Dim2>>::zeros([3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 12);
        assert!(!arr.is_materialized());
    }

    #[test]
    fn test_array_ops_lazy() {
        let a = <Array<f32, Dim2>>::zeros([3, 4]);
        let b = <Array<f32, Dim2>>::ones([3, 4]);

        // 演算は遅延評価
        let c = &a + &b;
        assert!(!c.is_materialized());
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_array_scalar_ops() {
        let a = <Array<f32, Dim2>>::ones([2, 3]);
        let b = &a * 2.0f32;
        assert_eq!(b.shape(), &[2, 3]);
    }

    #[test]
    fn test_array_clone() {
        let a = <Array<f32, Dim2>>::zeros([2, 2]);
        let b = a.clone();
        assert_eq!(b.shape(), &[2, 2]);
    }
}

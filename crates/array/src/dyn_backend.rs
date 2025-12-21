//! 配列型
//!
//! 実行時にデバイスを選択できる配列型を提供します。
//! `.to(device)`メソッドでデバイス間転送が可能です。

use crate::array::{ArrayElement, ArrayError};
use crate::device::Device;
use crate::dim::Dimension;

// ============================================================================
// Array - 配列型
// ============================================================================

/// 配列型
///
/// 実行時にデバイスを選択できる配列型です。
/// 次元型`D`は保持されるため、`Array2`を転送しても`Array2`のままです。
///
/// # 例
///
/// ```ignore
/// use harp_array::prelude::*;
///
/// // 配列の作成
/// let arr: Array2<f32> = Array2::zeros([3, 4]);
///
/// // デバイスに転送
/// if Device::Metal.is_available() {
///     let metal_arr = arr.to(Device::Metal)?;
/// }
/// ```
pub struct Array<T: ArrayElement, D: Dimension> {
    /// 現在のデバイス
    device: Device,
    /// データ（ホストメモリに保持）
    data: Vec<T>,
    /// 形状
    shape: Vec<usize>,
    /// 次元型マーカー
    _dim: std::marker::PhantomData<D>,
}

impl<T: ArrayElement, D: Dimension> Array<T, D> {
    /// データとデバイスから配列を作成
    pub fn from_vec_with_device(data: Vec<T>, shape: Vec<usize>, device: Device) -> Self {
        Self {
            device,
            data,
            shape,
            _dim: std::marker::PhantomData,
        }
    }

    /// デフォルトデバイスでデータから配列を作成
    pub fn from_vec(data: Vec<T>, shape: Vec<usize>) -> Self {
        Self::from_vec_with_device(data, shape, Device::default_device())
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

    /// データをベクタとして取得
    pub fn to_vec(&self) -> Vec<T> {
        self.data.clone()
    }

    /// 別のデバイスに転送
    ///
    /// # 例
    ///
    /// ```ignore
    /// let arr = Array::from_vec_with_device(data, shape, Device::Metal);
    /// let cpu_arr = arr.to(Device::Cpu)?;
    /// ```
    pub fn to(&self, device: Device) -> Result<Array<T, D>, ArrayError> {
        if !device.is_available() {
            return Err(ArrayError::Context(format!(
                "Device {} is not available",
                device.name()
            )));
        }

        // 現在は単純にデータをコピー
        // 将来的には実際のデバイス間転送を実装
        Ok(Array {
            device,
            data: self.data.clone(),
            shape: self.shape.clone(),
            _dim: std::marker::PhantomData,
        })
    }
}

// ============================================================================
// 生成メソッド（f32）
// ============================================================================

use crate::generators::IntoShape;

impl<D: Dimension> Array<f32, D> {
    /// ゼロで初期化された配列を生成
    ///
    /// # 例
    ///
    /// ```ignore
    /// let arr: Array2<f32> = Array2::zeros([3, 4]);
    /// ```
    pub fn zeros<S: IntoShape>(shape: S) -> Self {
        Self::zeros_on(shape, Device::default_device())
    }

    /// 指定デバイスでゼロ配列を生成
    pub fn zeros_on<S: IntoShape>(shape: S, device: Device) -> Self {
        Self::full_on(shape, 0.0, device)
    }

    /// 1で初期化された配列を生成
    pub fn ones<S: IntoShape>(shape: S) -> Self {
        Self::ones_on(shape, Device::default_device())
    }

    /// 指定デバイスで1配列を生成
    pub fn ones_on<S: IntoShape>(shape: S, device: Device) -> Self {
        Self::full_on(shape, 1.0, device)
    }

    /// 指定値で初期化された配列を生成
    pub fn full<S: IntoShape>(shape: S, value: f32) -> Self {
        Self::full_on(shape, value, Device::default_device())
    }

    /// 指定デバイスで指定値配列を生成
    pub fn full_on<S: IntoShape>(shape: S, value: f32, device: Device) -> Self {
        let shape_vec = shape.into_shape();
        let len: usize = shape_vec.iter().product();
        let data = vec![value; len];
        Self::from_vec_with_device(data, shape_vec, device)
    }

    /// 連番配列 [0.0, 1.0, 2.0, ...] を生成
    pub fn arange(size: usize) -> Self {
        Self::arange_on(size, Device::default_device())
    }

    /// 指定デバイスで連番配列を生成
    pub fn arange_on(size: usize, device: Device) -> Self {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        Self::from_vec_with_device(data, vec![size], device)
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
        let len: usize = shape_vec.iter().product();
        let data = vec![value; len];
        Self::from_vec_with_device(data, shape_vec, device)
    }

    /// 連番配列 [0, 1, 2, ...] を生成
    pub fn arange(size: usize) -> Self {
        Self::arange_on(size, Device::default_device())
    }

    /// 指定デバイスで連番配列を生成
    pub fn arange_on(size: usize, device: Device) -> Self {
        let data: Vec<i32> = (0..size as i32).collect();
        Self::from_vec_with_device(data, vec![size], device)
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
            device: self.device,
            data: self.data.clone(),
            shape: self.shape.clone(),
            _dim: std::marker::PhantomData,
        }
    }
}

impl<T: ArrayElement, D: Dimension> std::fmt::Debug for Array<T, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Array")
            .field("device", &self.device)
            .field("shape", &self.shape)
            .field("len", &self.data.len())
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
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::Dim2;

    #[test]
    fn test_array_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let arr: Array<f32, Dim2> =
            Array::from_vec_with_device(data.clone(), shape.clone(), Device::Cpu);

        assert_eq!(arr.device(), Device::Cpu);
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.ndim(), 2);
        assert_eq!(arr.len(), 4);
        assert_eq!(arr.to_vec(), data);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_array_to_same_device() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let arr: Array<f32, Dim2> = Array::from_vec_with_device(data.clone(), shape, Device::Cpu);

        // CPUからCPUへの転送は常に成功
        let arr2 = arr.to(Device::Cpu).unwrap();
        assert_eq!(arr2.device(), Device::Cpu);
        assert_eq!(arr2.to_vec(), data);
    }

    #[test]
    #[cfg(not(feature = "cpu"))]
    fn test_array_to_unavailable_device() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let arr: Array<f32, Dim2> = Array::from_vec_with_device(data, vec![2, 2], Device::Cpu);

        // CPUが無効な場合はエラー
        assert!(arr.to(Device::Cpu).is_err());
    }

    #[test]
    fn test_array_type_aliases() {
        let _: Array0<f32> = Array::from_vec_with_device(vec![1.0], vec![], Device::Cpu);
        let _: Array1<f32> = Array::from_vec_with_device(vec![1.0, 2.0], vec![2], Device::Cpu);
        let _: Array2<f32> =
            Array::from_vec_with_device(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], Device::Cpu);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_array_zeros() {
        let arr = <Array<f32, Dim2>>::zeros([3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.len(), 12);
        assert!(arr.to_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_array_ones() {
        let arr = <Array<f32, Dim2>>::ones([2, 3]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert!(arr.to_vec().iter().all(|&x| x == 1.0));
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_array_full() {
        let arr = <Array<f32, Dim2>>::full([2, 2], 3.14);
        assert!(arr.to_vec().iter().all(|&x| (x - 3.14_f32).abs() < 1e-6));
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_array_arange() {
        let arr = <Array<f32, crate::dim::Dim1>>::arange(5);
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_array_zeros_like() {
        let original = <Array<f32, Dim2>>::ones([3, 4]);
        let zeros = <Array<f32, Dim2>>::zeros_like(&original);
        assert_eq!(zeros.shape(), original.shape());
        assert!(zeros.to_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn test_array_i32() {
        let arr = <Array<i32, Dim2>>::zeros([2, 3]);
        assert_eq!(arr.shape(), &[2, 3]);
        assert!(arr.to_vec().iter().all(|&x| x == 0));

        let ones = <Array<i32, Dim2>>::ones([2, 3]);
        assert!(ones.to_vec().iter().all(|&x| x == 1));

        let arange = <Array<i32, crate::dim::Dim1>>::arange(5);
        assert_eq!(arange.to_vec(), vec![0, 1, 2, 3, 4]);
    }
}

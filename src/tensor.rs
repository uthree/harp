//! Tensor API with lazy evaluation.

use std::sync::{Arc, Mutex};

use crate::device::{Buffer, BufferMap, default_device, get_device};
use crate::dtype::{DType, Scalar, ScalarValue};
use crate::shape::Shape;
use crate::uop::UOp;

/// Global buffer map for managing tensor data.
static BUFFER_MAP: std::sync::OnceLock<Mutex<BufferMap>> = std::sync::OnceLock::new();

fn buffer_map() -> &'static Mutex<BufferMap> {
    BUFFER_MAP.get_or_init(|| Mutex::new(BufferMap::new()))
}

/// Inner data for a Tensor.
struct TensorInner {
    uop: UOp,
    device: String,
    realized: Mutex<Option<Arc<dyn Buffer>>>,
}

/// A tensor with lazy evaluation.
///
/// Tensors are immutable and reference-counted. Operations on tensors
/// build a computation graph that is only evaluated when `realize()` is called.
#[derive(Clone)]
pub struct Tensor(Arc<TensorInner>);

impl Tensor {
    // ============ Constructors ============

    /// Creates a new tensor from nested arrays or slices.
    pub fn new<T: IntoTensorData>(data: T) -> Self {
        data.into_tensor()
    }

    /// Creates a tensor filled with zeros.
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        Self::full(shape, 0.0f32)
    }

    /// Creates a tensor filled with ones.
    pub fn ones(shape: impl Into<Shape>) -> Self {
        Self::full(shape, 1.0f32)
    }

    /// Creates a tensor filled with a constant value.
    pub fn full<T: Scalar>(shape: impl Into<Shape>, value: T) -> Self {
        let shape = shape.into();
        let uop = UOp::constant(ScalarValue::from(value), shape);
        Tensor::from_uop(uop, "CPU")
    }

    /// Creates a tensor from raw data.
    pub fn from_raw<T: Scalar>(data: &[T], shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        assert_eq!(data.len(), shape.numel(), "Data length must match shape");

        // Create a buffer and fill it
        let device = default_device();
        let mut buffer = device.alloc(shape.numel(), T::DTYPE).unwrap();

        // Convert data to bytes
        let mut bytes = Vec::with_capacity(data.len() * T::DTYPE.size_bytes());
        for v in data {
            bytes.extend_from_slice(&v.to_bytes());
        }
        buffer.copy_from_host(&bytes);

        // Store buffer and create load UOp
        let buffer_id = buffer_map().lock().unwrap().insert(Arc::from(buffer));
        let uop = UOp::load(buffer_id, T::DTYPE, shape);

        Tensor::from_uop(uop, "CPU")
    }

    /// Creates a tensor from a UOp.
    pub(crate) fn from_uop(uop: UOp, device: &str) -> Self {
        Tensor(Arc::new(TensorInner {
            uop,
            device: device.to_string(),
            realized: Mutex::new(None),
        }))
    }

    // ============ Accessors ============

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Shape {
        self.0.uop.shape()
    }

    /// Returns the data type of the tensor.
    pub fn dtype(&self) -> DType {
        self.0.uop.dtype()
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape().rank()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape().numel()
    }

    /// Returns the device name.
    pub fn device(&self) -> &str {
        &self.0.device
    }

    /// Returns the underlying UOp.
    pub fn uop(&self) -> &UOp {
        &self.0.uop
    }

    // ============ Realization ============

    /// Realizes (evaluates) the tensor computation and returns self.
    pub fn realize(&self) -> &Self {
        let mut realized = self.0.realized.lock().unwrap();
        if realized.is_none() {
            let device = get_device(&self.0.device).unwrap_or_else(default_device);
            let mut buffers = buffer_map().lock().unwrap();
            let buffer = device.realize(&self.0.uop, &mut buffers).unwrap();
            *realized = Some(buffer);
        }
        self
    }

    /// Converts the tensor to a Vec.
    pub fn to_vec<T: Scalar>(&self) -> Vec<T> {
        self.realize();
        let realized = self.0.realized.lock().unwrap();
        let buffer = realized.as_ref().unwrap();
        let bytes = buffer.copy_to_host();

        let numel = self.numel();
        let mut result = Vec::with_capacity(numel);
        for i in 0..numel {
            let offset = i * T::DTYPE.size_bytes();
            result.push(T::from_bytes(&bytes[offset..]));
        }
        result
    }

    /// Returns a single scalar value (for scalar tensors).
    pub fn item<T: Scalar>(&self) -> T {
        assert_eq!(self.numel(), 1, "item() requires a scalar tensor");
        self.to_vec::<T>()[0]
    }

    // ============ Unary Operations ============

    /// Negation.
    pub fn neg(&self) -> Tensor {
        Tensor::from_uop(self.0.uop.neg(), &self.0.device)
    }

    /// Exponential.
    pub fn exp(&self) -> Tensor {
        Tensor::from_uop(self.0.uop.exp(), &self.0.device)
    }

    /// Natural logarithm.
    pub fn log(&self) -> Tensor {
        Tensor::from_uop(self.0.uop.log(), &self.0.device)
    }

    /// Square root.
    pub fn sqrt(&self) -> Tensor {
        Tensor::from_uop(self.0.uop.sqrt(), &self.0.device)
    }

    /// Reciprocal (1/x).
    pub fn recip(&self) -> Tensor {
        Tensor::from_uop(self.0.uop.recip(), &self.0.device)
    }

    /// Sine.
    pub fn sin(&self) -> Tensor {
        Tensor::from_uop(self.0.uop.sin(), &self.0.device)
    }

    /// Cosine.
    pub fn cos(&self) -> Tensor {
        Tensor::from_uop(self.0.uop.cos(), &self.0.device)
    }

    // ============ Binary Operations ============

    /// Addition.
    pub fn add(&self, other: &Tensor) -> Tensor {
        Tensor::from_uop(self.0.uop.add(&other.0.uop), &self.0.device)
    }

    /// Subtraction.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        Tensor::from_uop(self.0.uop.sub(&other.0.uop), &self.0.device)
    }

    /// Multiplication.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        Tensor::from_uop(self.0.uop.mul(&other.0.uop), &self.0.device)
    }

    /// Division.
    pub fn div(&self, other: &Tensor) -> Tensor {
        Tensor::from_uop(self.0.uop.div(&other.0.uop), &self.0.device)
    }

    /// Element-wise maximum.
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        Tensor::from_uop(self.0.uop.maximum(&other.0.uop), &self.0.device)
    }

    /// Less than comparison.
    pub fn lt(&self, other: &Tensor) -> Tensor {
        Tensor::from_uop(self.0.uop.lt(&other.0.uop), &self.0.device)
    }

    /// Equality comparison.
    pub fn eq(&self, other: &Tensor) -> Tensor {
        Tensor::from_uop(self.0.uop.eq(&other.0.uop), &self.0.device)
    }

    /// Where operation: select from x or y based on self as condition.
    pub fn where_cond(&self, x: &Tensor, y: &Tensor) -> Tensor {
        Tensor::from_uop(
            UOp::where_op(&self.0.uop, &x.0.uop, &y.0.uop),
            &self.0.device,
        )
    }

    // ============ Reduce Operations ============

    /// Sum reduction.
    pub fn sum(&self, axes: Option<Vec<usize>>, keepdims: bool) -> Tensor {
        Tensor::from_uop(self.0.uop.sum(axes, keepdims), &self.0.device)
    }

    /// Max reduction.
    pub fn max(&self, axes: Option<Vec<usize>>, keepdims: bool) -> Tensor {
        Tensor::from_uop(self.0.uop.reduce_max(axes, keepdims), &self.0.device)
    }

    /// Mean reduction.
    pub fn mean(&self, axes: Option<Vec<usize>>, keepdims: bool) -> Tensor {
        let sum = self.sum(axes.clone(), keepdims);
        let count = if let Some(ref ax) = axes {
            ax.iter().map(|&i| self.shape().dim(i)).product::<usize>()
        } else {
            self.numel()
        };
        let count_tensor = Tensor::full(Shape::scalar(), count as f32);
        sum.div(&count_tensor)
    }

    // ============ Movement Operations ============

    /// Reshape to a new shape.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Tensor {
        Tensor::from_uop(self.0.uop.reshape(shape.into()), &self.0.device)
    }

    /// Expand (broadcast) to a new shape.
    pub fn expand(&self, shape: impl Into<Shape>) -> Tensor {
        Tensor::from_uop(self.0.uop.expand(shape.into()), &self.0.device)
    }

    /// Permute dimensions.
    pub fn permute(&self, axes: impl Into<Vec<usize>>) -> Tensor {
        Tensor::from_uop(self.0.uop.permute(axes.into()), &self.0.device)
    }

    /// Transpose (swap last two dimensions).
    pub fn transpose(&self) -> Tensor {
        let ndim = self.ndim();
        assert!(ndim >= 2, "Transpose requires at least 2 dimensions");
        let mut axes: Vec<_> = (0..ndim).collect();
        axes.swap(ndim - 2, ndim - 1);
        self.permute(axes)
    }

    /// Cast to a different dtype.
    pub fn cast(&self, dtype: DType) -> Tensor {
        Tensor::from_uop(self.0.uop.cast(dtype), &self.0.device)
    }

    /// Flatten to 1D.
    pub fn flatten(&self) -> Tensor {
        self.reshape([self.numel()])
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={}, dtype={}, device={})",
            self.shape(),
            self.dtype(),
            self.device()
        )
    }
}

// ============ Operator Overloads ============

impl std::ops::Neg for &Tensor {
    type Output = Tensor;
    fn neg(self) -> Tensor {
        self.neg()
    }
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Tensor {
        self.add(rhs)
    }
}

impl std::ops::Sub for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Tensor {
        self.sub(rhs)
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Tensor {
        self.mul(rhs)
    }
}

impl std::ops::Div for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Tensor {
        self.div(rhs)
    }
}

// ============ IntoTensorData Trait ============

/// Trait for types that can be converted into a Tensor.
pub trait IntoTensorData {
    fn into_tensor(self) -> Tensor;
}

// Scalar implementations
impl IntoTensorData for f32 {
    fn into_tensor(self) -> Tensor {
        Tensor::full(Shape::scalar(), self)
    }
}

impl IntoTensorData for f64 {
    fn into_tensor(self) -> Tensor {
        Tensor::full(Shape::scalar(), self)
    }
}

impl IntoTensorData for i32 {
    fn into_tensor(self) -> Tensor {
        Tensor::full(Shape::scalar(), self)
    }
}

impl IntoTensorData for i64 {
    fn into_tensor(self) -> Tensor {
        Tensor::full(Shape::scalar(), self)
    }
}

// 1D array implementations
impl<const N: usize> IntoTensorData for [f32; N] {
    fn into_tensor(self) -> Tensor {
        Tensor::from_raw(&self, [N])
    }
}

impl<const N: usize> IntoTensorData for [f64; N] {
    fn into_tensor(self) -> Tensor {
        Tensor::from_raw(&self, [N])
    }
}

impl<const N: usize> IntoTensorData for [i32; N] {
    fn into_tensor(self) -> Tensor {
        Tensor::from_raw(&self, [N])
    }
}

// 2D array implementations
impl<const M: usize, const N: usize> IntoTensorData for [[f32; N]; M] {
    fn into_tensor(self) -> Tensor {
        let flat: Vec<f32> = self.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_raw(&flat, [M, N])
    }
}

impl<const M: usize, const N: usize> IntoTensorData for [[f64; N]; M] {
    fn into_tensor(self) -> Tensor {
        let flat: Vec<f64> = self.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_raw(&flat, [M, N])
    }
}

impl<const M: usize, const N: usize> IntoTensorData for [[i32; N]; M] {
    fn into_tensor(self) -> Tensor {
        let flat: Vec<i32> = self.iter().flat_map(|row| row.iter().copied()).collect();
        Tensor::from_raw(&flat, [M, N])
    }
}

// Vec implementations
impl IntoTensorData for Vec<f32> {
    fn into_tensor(self) -> Tensor {
        let len = self.len();
        Tensor::from_raw(&self, [len])
    }
}

impl IntoTensorData for Vec<f64> {
    fn into_tensor(self) -> Tensor {
        let len = self.len();
        Tensor::from_raw(&self, [len])
    }
}

impl IntoTensorData for Vec<i32> {
    fn into_tensor(self) -> Tensor {
        let len = self.len();
        Tensor::from_raw(&self, [len])
    }
}

impl IntoTensorData for Vec<Vec<f32>> {
    fn into_tensor(self) -> Tensor {
        let rows = self.len();
        let cols = self.first().map(|r| r.len()).unwrap_or(0);
        let flat: Vec<f32> = self.into_iter().flatten().collect();
        Tensor::from_raw(&flat, [rows, cols])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::register_device;
    use crate::runtime::CpuDevice;

    fn setup() {
        register_device(Arc::new(CpuDevice::new()));
    }

    #[test]
    fn test_tensor_creation() {
        setup();
        let t = Tensor::zeros([2, 3]);
        assert_eq!(t.shape().dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::Float32);
    }

    #[test]
    fn test_tensor_from_array() {
        setup();
        let t = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
        assert_eq!(t.shape().dims(), &[2, 2]);

        let data = t.to_vec::<f32>();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_add() {
        setup();
        let a = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::new([[5.0f32, 6.0], [7.0, 8.0]]);
        let c = &a + &b;

        let data = c.to_vec::<f32>();
        assert_eq!(data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_tensor_sum() {
        setup();
        let t = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
        let s = t.sum(None, false);

        assert_eq!(s.item::<f32>(), 10.0);
    }

    #[test]
    fn test_tensor_reshape() {
        setup();
        let t = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
        let r = t.reshape([4]);

        assert_eq!(r.shape().dims(), &[4]);
        let data = r.to_vec::<f32>();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tensor_broadcast_add() {
        setup();
        let a = Tensor::new([[1.0f32, 2.0], [3.0, 4.0]]);
        let b = Tensor::new([10.0f32, 20.0]);
        let c = &a + &b;

        let data = c.to_vec::<f32>();
        assert_eq!(data, vec![11.0, 22.0, 13.0, 24.0]);
    }
}

pub mod autograd;
pub mod backend;
pub mod ops;

pub use autograd::{GradFn, TensorId, TensorMeta};
pub use backend::validate_backend;

use crate::ast::DType;
use crate::graph::{Graph, GraphNode};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Type-level representation of Rust types for tensors.
///
/// This trait maps Rust types to DType at compile time.
pub trait TensorType: 'static {
    /// The corresponding DType
    const DTYPE: DType;
}

impl TensorType for f32 {
    const DTYPE: DType = DType::F32;
}

impl TensorType for isize {
    const DTYPE: DType = DType::Isize;
}

impl TensorType for usize {
    const DTYPE: DType = DType::Usize;
}

/// Type-level representation of tensor dimensions.
///
/// This allows compile-time checking of tensor dimensions similar to ndarray.
pub trait Dimension: Clone {
    /// Number of dimensions (rank)
    const NDIM: Option<usize>;

    /// Check if a shape is valid for this dimension type
    fn check_shape(shape: &[usize]) -> bool;
}

/// Dynamic dimension - shape determined at runtime
#[derive(Clone)]
pub struct Dyn;

impl Dimension for Dyn {
    const NDIM: Option<usize> = None;

    fn check_shape(_shape: &[usize]) -> bool {
        true // Any shape is valid for dynamic dimensions
    }
}

/// 0-dimensional tensor (scalar)
#[derive(Clone)]
pub struct D0;

impl Dimension for D0 {
    const NDIM: Option<usize> = Some(0);

    fn check_shape(shape: &[usize]) -> bool {
        shape.is_empty() || (shape.len() == 1 && shape[0] == 1)
    }
}

/// 1-dimensional tensor (vector)
#[derive(Clone)]
pub struct D1;

impl Dimension for D1 {
    const NDIM: Option<usize> = Some(1);

    fn check_shape(shape: &[usize]) -> bool {
        shape.len() == 1
    }
}

/// 2-dimensional tensor (matrix)
#[derive(Clone)]
pub struct D2;

impl Dimension for D2 {
    const NDIM: Option<usize> = Some(2);

    fn check_shape(shape: &[usize]) -> bool {
        shape.len() == 2
    }
}

/// 3-dimensional tensor
#[derive(Clone)]
pub struct D3;

impl Dimension for D3 {
    const NDIM: Option<usize> = Some(3);

    fn check_shape(shape: &[usize]) -> bool {
        shape.len() == 3
    }
}

/// 4-dimensional tensor
#[derive(Clone)]
pub struct D4;

impl Dimension for D4 {
    const NDIM: Option<usize> = Some(4);

    fn check_shape(shape: &[usize]) -> bool {
        shape.len() == 4
    }
}

/// Raw data storage for tensors
#[derive(Clone)]
enum TensorData {
    F32(Vec<f32>),
    Isize(Vec<isize>),
    Usize(Vec<usize>),
}

static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Base tensor without dimension type checking.
/// This handles all core functionality.
#[derive(Clone)]
pub struct TensorBase {
    /// Unique ID for this tensor
    id: TensorId,
    /// Actual data (Some if evaluated, None if lazy)
    data: Option<TensorData>,
    /// Computation graph node (Some if part of computation graph)
    graph_node: Option<GraphNode>,
    /// Graph that this tensor is part of
    graph: Option<Graph>,
    /// Backend name for execution
    backend_name: String,
    /// Data type
    dtype: DType,
    /// Shape
    shape: Vec<usize>,
    /// Whether gradient computation is required
    requires_grad: bool,
    /// Gradient tensor (computed during backward pass)
    grad: Option<Box<TensorBase>>,
    /// Autograd metadata (for backward pass)
    autograd_meta: Option<TensorMeta>,
}

/// A tensor with optional compile-time type and dimension checking.
///
/// # Type Parameters
/// - `T`: Element type (f32, isize, usize)
/// - `D`: Dimension type (Dyn, D1, D2, etc.)
///
/// # Examples
///
/// ```
/// use harp::tensor::{Tensor, D1, D2, Dyn};
///
/// // 1D f32 tensor (compile-time checked)
/// let vec: Tensor<f32, D1> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c");
///
/// // 2D f32 tensor (compile-time checked)
/// let mat: Tensor<f32, D2> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2], "c");
///
/// // Dynamic dimension (runtime checked)
/// let dyn_tensor: Tensor<f32, Dyn> = Tensor::from_vec(vec![1.0], &[1], "c");
/// ```
pub struct Tensor<T: TensorType, D: Dimension = Dyn> {
    inner: TensorBase,
    _phantom: PhantomData<(T, D)>,
}

impl<T: TensorType, D: Dimension> Clone for Tensor<T, D> {
    fn clone(&self) -> Self {
        Tensor {
            inner: self.inner.clone(),
            _phantom: PhantomData,
        }
    }
}

impl TensorBase {
    /// Create a tensor from a Vec of data.
    pub fn from_vec<T: 'static>(data: Vec<T>, shape: &[usize], backend_name: String) -> Self {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            panic!(
                "Data length {} doesn't match shape {:?} (expected {} elements)",
                data.len(),
                shape,
                numel
            );
        }

        let dtype = Self::infer_dtype::<T>();
        let tensor_data = Self::vec_to_tensor_data(data, &dtype);

        TensorBase {
            id: TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            data: Some(tensor_data),
            graph_node: None,
            graph: None,
            backend_name,
            dtype,
            shape: shape.to_vec(),
            requires_grad: false,
            grad: None,
            autograd_meta: None,
        }
    }

    /// Create a tensor from a graph node.
    pub(crate) fn from_graph_node(
        graph_node: GraphNode,
        graph: Graph,
        backend_name: String,
    ) -> Self {
        let shape: Vec<usize> = graph_node
            .view
            .shape()
            .iter()
            .map(|expr| match expr {
                crate::graph::shape::Expr::Const(n) => *n as usize,
                _ => panic!("Dynamic shapes not yet supported in Tensor API"),
            })
            .collect();

        TensorBase {
            id: TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            data: None,
            graph_node: Some(graph_node.clone()),
            graph: Some(graph),
            backend_name,
            dtype: graph_node.dtype.clone(),
            shape,
            requires_grad: false,
            grad: None,
            autograd_meta: None,
        }
    }

    fn vec_to_tensor_data<T: 'static>(data: Vec<T>, dtype: &DType) -> TensorData {
        use std::any::TypeId;

        match dtype {
            DType::F32 if TypeId::of::<T>() == TypeId::of::<f32>() => {
                let ptr = data.as_ptr() as *const f32;
                let len = data.len();
                std::mem::forget(data);
                TensorData::F32(unsafe { Vec::from_raw_parts(ptr as *mut f32, len, len) })
            }
            DType::Isize if TypeId::of::<T>() == TypeId::of::<isize>() => {
                let ptr = data.as_ptr() as *const isize;
                let len = data.len();
                std::mem::forget(data);
                TensorData::Isize(unsafe { Vec::from_raw_parts(ptr as *mut isize, len, len) })
            }
            DType::Usize if TypeId::of::<T>() == TypeId::of::<usize>() => {
                let ptr = data.as_ptr() as *const usize;
                let len = data.len();
                std::mem::forget(data);
                TensorData::Usize(unsafe { Vec::from_raw_parts(ptr as *mut usize, len, len) })
            }
            _ => panic!("Type mismatch"),
        }
    }

    /// Get the unique ID of this tensor.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Enable gradient computation for this tensor.
    pub fn enable_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Disable gradient computation for this tensor.
    pub fn disable_grad(mut self) -> Self {
        self.requires_grad = false;
        self
    }

    /// Check if gradient computation is enabled.
    pub fn is_requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the gradient of this tensor.
    pub fn grad(&self) -> Option<&TensorBase> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Set the gradient of this tensor.
    pub fn set_grad(&mut self, grad: TensorBase) {
        self.grad = Some(Box::new(grad));
    }

    /// Zero the gradient of this tensor.
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Detach this tensor from the computation graph.
    ///
    /// Returns a new tensor that shares data with this tensor but is detached
    /// from the autograd graph. The returned tensor will never require gradients.
    ///
    /// This is useful when you want to use a tensor's value without tracking
    /// gradients through it.
    ///
    /// # Example
    /// ```ignore
    /// let x = tensor.enable_grad();
    /// let y = (x.clone() * 2.0).detach();  // y won't track gradients
    /// ```
    pub fn detach(&self) -> Self {
        let mut detached = self.clone();
        detached.requires_grad = false;
        detached.autograd_meta = None;
        detached.grad = None;
        detached
    }

    /// Perform backward pass to compute gradients.
    /// This should be called on the final output tensor (typically a scalar loss).
    pub fn backward(&mut self) {
        use std::collections::{HashMap, HashSet, VecDeque};

        // If this is a leaf tensor (no graph), just set gradient to ones
        let Some(grad_graph) = self.graph.as_ref() else {
            if self.requires_grad {
                // For leaf tensors, create a simple gradient tensor with ones
                let ones = vec![1.0f32; self.numel()];
                let grad = TensorBase::from_vec(ones, &self.shape, self.backend_name.clone());
                self.grad = Some(Box::new(grad));
            }
            return;
        };

        let grad_graph = grad_graph.clone();

        // Create gradient node with same shape as output
        let mut grad_graph_mut = grad_graph.clone();
        let shape_exprs: Vec<_> = self.shape.iter().map(|&s| (s as isize).into()).collect();
        let grad_output_node = grad_graph_mut.input(self.dtype.clone(), shape_exprs);

        // Map from tensor ID to gradient GraphNode
        let mut gradients: HashMap<TensorId, GraphNode> = HashMap::new();
        gradients.insert(self.id, grad_output_node.clone());

        // Collect all tensors in the computation graph via topological sort
        let mut visited = HashSet::new();
        let mut topo_order = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back((self.id, self.autograd_meta.clone()));

        // Build topological order (simple BFS-based approach)
        while let Some((tensor_id, meta_opt)) = queue.pop_front() {
            if visited.contains(&tensor_id) {
                continue;
            }
            visited.insert(tensor_id);

            if let Some(meta) = meta_opt {
                topo_order.push((tensor_id, meta.clone()));

                // Add inputs to queue
                for (input_id, _) in &meta.inputs {
                    if !visited.contains(input_id) {
                        // We don't have direct access to input tensors' metadata here
                        // This is a limitation we'll address in a future iteration
                        queue.push_back((*input_id, None));
                    }
                }
            }
        }

        // Reverse to get correct backprop order (outputs first, then inputs)
        topo_order.reverse();

        // Compute gradients in reverse topological order
        for (_tensor_id, meta) in topo_order {
            if let Some(grad_fn) = &meta.grad_fn {
                // Get gradient for this tensor's output
                let tensor_node = &meta.graph_node;

                // For simplicity in this initial implementation, we'll create a temporary gradient
                // In a full implementation, this would look up the actual accumulated gradient
                let mut temp_grad_graph = grad_graph_mut.clone();
                let out_shape_exprs: Vec<_> = tensor_node.view.shape().to_vec();
                let grad_out = temp_grad_graph.input(tensor_node.dtype.clone(), out_shape_exprs);

                // Compute gradients for inputs
                let input_nodes: Vec<GraphNode> =
                    meta.inputs.iter().map(|(_, node)| node.clone()).collect();
                let input_grads = grad_fn.backward(grad_out, &input_nodes);

                // Accumulate gradients for each input
                for (i, grad_opt) in input_grads.iter().enumerate() {
                    if let Some(grad) = grad_opt {
                        let input_id = meta.inputs[i].0;

                        if let Some(existing_grad) = gradients.get(&input_id) {
                            // Add to existing gradient
                            let new_grad = existing_grad.clone() + grad.clone();
                            gradients.insert(input_id, new_grad);
                        } else {
                            gradients.insert(input_id, grad.clone());
                        }
                    }
                }
            }
        }

        // For this tensor (the output), store the gradient
        // In a full implementation, we would store gradients for all tensors
        if let Some(grad_node) = gradients.get(&self.id) {
            // Create a tensor from the gradient graph node
            let grad_tensor = TensorBase::from_graph_node(
                grad_node.clone(),
                grad_graph_mut,
                self.backend_name.clone(),
            );
            self.grad = Some(Box::new(grad_tensor));
        }

        // NOTE: This is a simplified implementation. A complete implementation would:
        // 1. Track all tensors in a global computation graph
        // 2. Properly accumulate gradients across multiple paths
        // 3. Handle retain_graph and create_graph flags
        // 4. Support higher-order derivatives
    }

    fn infer_dtype<T: 'static>() -> DType {
        use std::any::TypeId;

        if TypeId::of::<T>() == TypeId::of::<f32>() {
            DType::F32
        } else if TypeId::of::<T>() == TypeId::of::<isize>() {
            DType::Isize
        } else if TypeId::of::<T>() == TypeId::of::<usize>() {
            DType::Usize
        } else {
            panic!("Unsupported data type");
        }
    }
}

impl<T: TensorType, D: Dimension> Tensor<T, D> {
    /// Create a tensor from a Vec of data.
    ///
    /// # Panics
    /// Panics if the shape doesn't match the dimension type D or if the data type doesn't match T.
    pub fn from_vec(data: Vec<T>, shape: &[usize], backend_name: &str) -> Self {
        if !D::check_shape(shape) {
            panic!(
                "Shape {:?} is not compatible with dimension type (expected ndim: {:?})",
                shape,
                D::NDIM
            );
        }

        validate_backend(backend_name);

        Tensor {
            inner: TensorBase::from_vec(data, shape, backend_name.to_string()),
            _phantom: PhantomData,
        }
    }

    /// Create a tensor from a graph node.
    pub(crate) fn from_graph_node(
        graph_node: GraphNode,
        graph: Graph,
        backend_name: String,
    ) -> Self {
        let shape: Vec<usize> = graph_node
            .view
            .shape()
            .iter()
            .map(|expr| match expr {
                crate::graph::shape::Expr::Const(n) => *n as usize,
                _ => panic!("Dynamic shapes not yet supported in Tensor API"),
            })
            .collect();

        if !D::check_shape(&shape) {
            panic!(
                "Graph node shape {:?} is not compatible with dimension type (expected ndim: {:?})",
                shape,
                D::NDIM
            );
        }

        // Check that the graph node's dtype matches T
        if graph_node.dtype != T::DTYPE {
            panic!(
                "Graph node dtype {:?} doesn't match tensor type (expected {:?})",
                graph_node.dtype,
                T::DTYPE
            );
        }

        Tensor {
            inner: TensorBase::from_graph_node(graph_node, graph, backend_name),
            _phantom: PhantomData,
        }
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Get the data type of the tensor.
    pub fn dtype(&self) -> &DType {
        self.inner.dtype()
    }

    /// Get the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Get the total number of elements.
    pub fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Enable gradient computation for this tensor.
    pub fn enable_grad(mut self) -> Self {
        self.inner = self.inner.enable_grad();
        self
    }

    /// Disable gradient computation for this tensor.
    pub fn disable_grad(mut self) -> Self {
        self.inner = self.inner.disable_grad();
        self
    }

    /// Check if gradient computation is enabled.
    pub fn is_requires_grad(&self) -> bool {
        self.inner.is_requires_grad()
    }

    /// Get the gradient of this tensor.
    pub fn grad(&self) -> Option<Tensor<T, D>> {
        self.inner.grad().map(|g| Tensor {
            inner: g.clone(),
            _phantom: PhantomData,
        })
    }

    /// Set the gradient of this tensor.
    pub fn set_grad(&mut self, grad: Tensor<T, D>) {
        self.inner.set_grad(grad.inner);
    }

    /// Zero the gradient of this tensor.
    pub fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }

    /// Perform backward pass to compute gradients.
    pub fn backward(&mut self) {
        self.inner.backward();
    }

    /// Detach this tensor from the computation graph.
    ///
    /// Returns a new tensor that shares data with this tensor but is detached
    /// from the autograd graph. The returned tensor will never require gradients.
    ///
    /// This is useful when you want to use a tensor's value without tracking
    /// gradients through it.
    ///
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3], "c").enable_grad();
    /// let y = &x + &x;
    /// let z = y.detach();  // z won't track gradients
    /// ```
    pub fn detach(&self) -> Self {
        Tensor {
            inner: self.inner.detach(),
            _phantom: PhantomData,
        }
    }
}

/// Type aliases for common f32 tensor types
pub type Tensor0<T = f32> = Tensor<T, D0>; // Scalar
pub type Tensor1<T = f32> = Tensor<T, D1>; // Vector
pub type Tensor2<T = f32> = Tensor<T, D2>; // Matrix
pub type Tensor3<T = f32> = Tensor<T, D3>; // 3D tensor
pub type Tensor4<T = f32> = Tensor<T, D4>; // 4D tensor
pub type TensorDyn<T = f32> = Tensor<T, Dyn>; // Dynamic dimension

#[cfg(all(test, feature = "backend-c"))]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor: Tensor1<f32> = Tensor::from_vec(data, &[4], "c");

        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.ndim(), 1);
        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.dtype(), &DType::F32);
    }

    #[test]
    fn test_tensor_2d_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let tensor: Tensor2<f32> = Tensor::from_vec(data, &[2, 2], "c");

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 4);
    }

    #[test]
    #[should_panic(expected = "Shape")]
    fn test_tensor_dimension_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        // Try to create 1D tensor with 2D shape - should panic
        let _tensor: Tensor1<f32> = Tensor::from_vec(data, &[2, 2], "c");
    }

    #[test]
    fn test_tensor_dynamic_dimension() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];

        // Dynamic dimension accepts any shape
        let tensor1: TensorDyn<f32> = Tensor::from_vec(data.clone(), &[4], "c");
        let tensor2: TensorDyn<f32> = Tensor::from_vec(data.clone(), &[2, 2], "c");
        let tensor3: TensorDyn<f32> = Tensor::from_vec(data, &[2, 1, 2], "c");

        assert_eq!(tensor1.ndim(), 1);
        assert_eq!(tensor2.ndim(), 2);
        assert_eq!(tensor3.ndim(), 3);
    }

    #[test]
    fn test_requires_grad() {
        let data = vec![1.0f32, 2.0, 3.0];
        let tensor = Tensor::<f32, D1>::from_vec(data, &[3], "c").enable_grad();

        assert!(tensor.is_requires_grad());
    }

    #[test]
    fn test_tensor_type_inference() {
        // Test with different types
        let f32_tensor: Tensor1<f32> = Tensor::from_vec(vec![1.0f32, 2.0], &[2], "c");
        let isize_tensor: Tensor1<isize> = Tensor::from_vec(vec![1isize, 2], &[2], "c");
        let usize_tensor: Tensor1<usize> = Tensor::from_vec(vec![1usize, 2], &[2], "c");

        assert_eq!(f32_tensor.dtype(), &DType::F32);
        assert_eq!(isize_tensor.dtype(), &DType::Isize);
        assert_eq!(usize_tensor.dtype(), &DType::Usize);
    }

    #[test]
    fn test_detach() {
        let x = Tensor::<f32, D1>::from_vec(vec![1.0, 2.0, 3.0], &[3], "c").enable_grad();
        assert!(x.is_requires_grad());

        // Detach should create a new tensor without gradient tracking
        let y = x.detach();
        assert!(!y.is_requires_grad());
        assert_eq!(y.shape(), x.shape());
    }

    #[test]
    fn test_detach_after_operation() {
        use crate::tensor::ops::*;

        let x = Tensor::<f32, D1>::from_vec(vec![1.0, 2.0, 3.0], &[3], "c").enable_grad();
        let y = Tensor::<f32, D1>::from_vec(vec![2.0, 3.0, 4.0], &[3], "c");

        // Perform an operation
        let z = &x + &y;
        assert!(z.is_requires_grad()); // z should track gradients

        // Detach z
        let w = z.detach();
        assert!(!w.is_requires_grad()); // w should not track gradients
        assert_eq!(w.shape(), z.shape());
    }
}

//! Learnable Parameter
//!
//! `Parameter` wraps a tensor and automatically enables gradient tracking.
//! It is designed for use with the `Module` trait and optimizers.

use eclat::graph::GraphNode;
use eclat::tensor::GradientParam;
use eclat::tensor::Tensor;
use eclat::tensor::dim::{Dimension, Dyn};
use std::cell::{Ref, RefCell};
use std::marker::PhantomData;
use std::rc::Rc;

/// Base trait for parameters that can be used by optimizers.
///
/// This trait allows optimizers to work with parameters of any dimension
/// through dynamic dispatch.
///
/// Extends `GradientParam` to enable backward pass gradient computation.
pub trait ParameterBase: GradientParam {
    /// Get the parameter name.
    fn name(&self) -> &str;

    /// Get the parameter shape.
    fn shape(&self) -> &[usize];

    /// Get the number of elements in the parameter.
    fn numel(&self) -> usize;

    /// Get the gradient as a dynamic tensor, if computed.
    fn grad_dyn(&self) -> Option<Tensor<Dyn, f32>>;

    /// Clear the gradient of this parameter.
    fn zero_grad(&self);

    /// Update the parameter data.
    fn update_data(&self, new_data: &[f32]) -> Result<(), ParameterError>;

    /// Get the parameter data as a vector.
    fn to_vec(&self) -> Result<Vec<f32>, ParameterError>;
}

/// Internal state of a parameter
struct ParameterInner<D: Dimension> {
    /// Parameter name (for debugging and state_dict)
    name: String,
    /// The underlying tensor with gradient tracking enabled
    tensor: RefCell<Tensor<D, f32>>,
    /// Shape of the parameter
    shape: Vec<usize>,
    /// Initial data (stored until device is available)
    initial_data: RefCell<Option<Vec<f32>>>,
    /// Phantom data for dimension type
    _dim: PhantomData<D>,
}

/// A learnable parameter for neural network modules.
///
/// `Parameter<D>` is a wrapper around `Tensor<D, f32>` that:
/// - Automatically enables gradient tracking
/// - Provides a named interface for parameter management
/// - Supports sharing via `Rc` for efficient cloning
///
/// # Type Parameter
/// - `D`: The dimension type (D1, D2, D3, D4, D5, or Dyn)
///
/// # Example
///
/// ```ignore
/// use eclat_nn::nn::Parameter;
/// use eclat::tensor::dim::D2;
///
/// // Create a 2D parameter with shape [10, 5]
/// let weight: Parameter<D2> = Parameter::new("weight", &[10, 5]);
///
/// // Create a 1D parameter with specific data
/// let bias: Parameter<D1> = Parameter::from_data("bias", &[0.0; 5], &[5]);
/// ```
#[derive(Clone)]
pub struct Parameter<D: Dimension> {
    inner: Rc<ParameterInner<D>>,
}

impl<D: Dimension> Parameter<D> {
    /// Create a new parameter with the given name and shape.
    ///
    /// The parameter is initialized with zeros and gradient tracking is enabled.
    pub fn new(name: &str, shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let data = vec![0.0f32; numel];
        Self::from_data(name, &data, shape)
    }

    /// Create a parameter from existing data.
    ///
    /// # Arguments
    /// * `name` - Parameter name
    /// * `data` - Initial data (flattened)
    /// * `shape` - Shape of the parameter
    ///
    /// # Panics
    /// Panics if `data.len()` doesn't match the product of `shape`.
    pub fn from_data(name: &str, data: &[f32], shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (numel={})",
            data.len(),
            shape,
            numel
        );

        // Create tensor with the specified dimension type
        let tensor: Tensor<D, f32> = Tensor::<Dyn, f32>::dyn_input(shape).into_static();
        tensor.requires_grad_(true);

        // Store initial data for lazy initialization (will be set when device is available)
        Self {
            inner: Rc::new(ParameterInner {
                name: name.to_string(),
                tensor: RefCell::new(tensor),
                shape: shape.to_vec(),
                initial_data: RefCell::new(Some(data.to_vec())),
                _dim: PhantomData,
            }),
        }
    }

    /// Get the parameter name.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Get the parameter shape.
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    /// Get the number of elements in the parameter.
    pub fn numel(&self) -> usize {
        self.inner.shape.iter().product()
    }

    /// Get a reference to the underlying tensor.
    ///
    /// This automatically ensures the parameter data is transferred to the device.
    pub fn tensor(&self) -> Ref<'_, Tensor<D, f32>> {
        // Ensure data is on device before returning tensor
        self.ensure_on_device()
            .expect("Failed to transfer parameter data to device");
        self.inner.tensor.borrow()
    }

    /// Get the gradient of this parameter, if computed.
    pub fn grad(&self) -> Option<Tensor<D, f32>> {
        self.inner.tensor.borrow().grad()
    }

    /// Get the gradient as a dynamic tensor, if computed.
    pub fn grad_dyn(&self) -> Option<Tensor<Dyn, f32>> {
        self.inner.tensor.borrow().grad().map(|g| g.into_dyn())
    }

    /// Clear the gradient of this parameter.
    pub fn zero_grad(&self) {
        self.inner.tensor.borrow().zero_grad();
    }

    /// Update the parameter data.
    ///
    /// This is typically called by optimizers after computing gradients.
    ///
    /// # Arguments
    /// * `new_data` - New data (flattened, must match parameter size)
    pub fn update_data(&self, new_data: &[f32]) -> Result<(), ParameterError> {
        use eclat::backend::has_default_device;

        if new_data.len() != self.numel() {
            return Err(ParameterError::ShapeMismatch {
                expected: self.numel(),
                got: new_data.len(),
            });
        }

        // Clear any cached initial data
        *self.inner.initial_data.borrow_mut() = None;

        // If device is available, transfer to device
        if has_default_device() {
            // Create a new tensor with the updated data
            let new_tensor: Tensor<D, f32> =
                Tensor::<Dyn, f32>::dyn_input(&self.inner.shape).into_static();
            new_tensor.requires_grad_(true);
            new_tensor
                .set_data(new_data)
                .map_err(|e| ParameterError::ExecutionError(e.to_string()))?;

            // Replace the old tensor
            *self.inner.tensor.borrow_mut() = new_tensor;
        } else {
            // No device: store as initial data for lazy transfer
            *self.inner.initial_data.borrow_mut() = Some(new_data.to_vec());
        }

        Ok(())
    }

    /// Get the parameter data as a vector.
    ///
    /// If a device is available and initial data was provided, this will
    /// first transfer the data to the device. If no device is available,
    /// this returns the initial data directly.
    pub fn to_vec(&self) -> Result<Vec<f32>, ParameterError> {
        // Try to ensure data is on device
        self.ensure_on_device()?;

        // If tensor is realized, read from device
        if self.inner.tensor.borrow().is_realized() {
            return self
                .inner
                .tensor
                .borrow()
                .to_vec()
                .map_err(|e| ParameterError::ExecutionError(e.to_string()));
        }

        // Otherwise return initial data if available
        if let Some(data) = self.inner.initial_data.borrow().as_ref() {
            return Ok(data.clone());
        }

        Err(ParameterError::ExecutionError(
            "Parameter has no data".to_string(),
        ))
    }

    /// Ensure the parameter data is on the device.
    ///
    /// This transfers the initial data to the device if available and not yet done.
    fn ensure_on_device(&self) -> Result<(), ParameterError> {
        use eclat::backend::has_default_device;

        // Check if already on device
        if self.inner.tensor.borrow().is_realized() {
            return Ok(());
        }

        // If no device, skip (will use initial_data)
        if !has_default_device() {
            return Ok(());
        }

        // Transfer initial data to device
        if let Some(data) = self.inner.initial_data.borrow_mut().take() {
            self.inner
                .tensor
                .borrow()
                .set_data(&data)
                .map_err(|e| ParameterError::ExecutionError(e.to_string()))?;
        }

        Ok(())
    }

    /// Check if the parameter has initial data set.
    pub fn has_data(&self) -> bool {
        self.inner.tensor.borrow().is_realized() || self.inner.initial_data.borrow().is_some()
    }

    /// Get a clone of the underlying GraphNode.
    pub fn graph_node(&self) -> GraphNode {
        self.inner.tensor.borrow().graph().clone()
    }

    /// Store the gradient from a computed GraphNode.
    ///
    /// This converts the GraphNode to a Tensor and stores it as the gradient.
    pub fn store_gradient(&self, grad_graph: GraphNode) {
        let grad_tensor: Tensor<D, f32> = Tensor::from_graph(grad_graph);
        self.inner.tensor.borrow().set_grad(grad_tensor);
    }
}

/// Implement ParameterBase for all Parameter<D> types
impl<D: Dimension> ParameterBase for Parameter<D> {
    fn name(&self) -> &str {
        Parameter::name(self)
    }

    fn shape(&self) -> &[usize] {
        Parameter::shape(self)
    }

    fn numel(&self) -> usize {
        Parameter::numel(self)
    }

    fn grad_dyn(&self) -> Option<Tensor<Dyn, f32>> {
        Parameter::grad_dyn(self)
    }

    fn zero_grad(&self) {
        Parameter::zero_grad(self)
    }

    fn update_data(&self, new_data: &[f32]) -> Result<(), ParameterError> {
        Parameter::update_data(self, new_data)
    }

    fn to_vec(&self) -> Result<Vec<f32>, ParameterError> {
        Parameter::to_vec(self)
    }
}

/// Implement GradientParam for all Parameter<D> types to enable backward pass.
impl<D: Dimension> GradientParam for Parameter<D> {
    fn graph_node(&self) -> GraphNode {
        Parameter::graph_node(self)
    }

    fn store_gradient(&self, grad_graph: GraphNode) {
        Parameter::store_gradient(self, grad_graph)
    }
}

impl<D: Dimension> std::fmt::Debug for Parameter<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parameter")
            .field("name", &self.inner.name)
            .field("shape", &self.inner.shape)
            .finish()
    }
}

/// Errors related to parameter operations
#[derive(Debug)]
pub enum ParameterError {
    /// Shape mismatch during update
    ShapeMismatch { expected: usize, got: usize },
    /// Execution error (e.g., during to_vec)
    ExecutionError(String),
}

impl std::fmt::Display for ParameterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "Shape mismatch: expected {} elements, got {}",
                    expected, got
                )
            }
            Self::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
        }
    }
}

impl std::error::Error for ParameterError {}

#[cfg(test)]
mod tests {
    use super::*;
    use eclat::tensor::dim::D2;

    #[test]
    fn test_parameter_new() {
        let p: Parameter<D2> = Parameter::new("test", &[3, 4]);
        assert_eq!(p.name(), "test");
        assert_eq!(p.shape(), &[3, 4]);
        assert_eq!(p.numel(), 12);
    }

    #[test]
    fn test_parameter_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let p: Parameter<D2> = Parameter::from_data("weight", &data, &[2, 3]);
        assert_eq!(p.name(), "weight");
        assert_eq!(p.shape(), &[2, 3]);
        assert_eq!(p.numel(), 6);
    }

    #[test]
    fn test_parameter_clone_shares_data() {
        let p1: Parameter<D2> = Parameter::new("shared", &[2, 2]);
        let p2 = p1.clone();

        // Both should have the same name and shape
        assert_eq!(p1.name(), p2.name());
        assert_eq!(p1.shape(), p2.shape());

        // Rc::ptr_eq to verify sharing
        assert!(Rc::ptr_eq(&p1.inner, &p2.inner));
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_parameter_from_data_wrong_size() {
        let data = vec![1.0, 2.0, 3.0];
        let _: Parameter<D2> = Parameter::from_data("bad", &data, &[2, 3]); // 6 != 3
    }

    #[test]
    fn test_parameter_base_trait() {
        let p: Parameter<D2> = Parameter::new("test", &[3, 4]);
        let base: &dyn ParameterBase = &p;

        assert_eq!(base.name(), "test");
        assert_eq!(base.shape(), &[3, 4]);
        assert_eq!(base.numel(), 12);
    }
}

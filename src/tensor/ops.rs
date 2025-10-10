use super::autograd::{
    AddBackward, CummaxBackward, CumprodBackward, CumsumBackward, GradFn, MaxBackward, MulBackward,
    NegBackward, ProductBackward, RecipBackward, SlidingWindowBackward, SumBackward, TensorMeta,
};
use super::{Dimension, Dyn, Tensor, TensorBase, TensorType};
use crate::graph::ops::{CumulativeOp, ReduceOp};
use crate::graph::{Graph, GraphNode, ReduceOps};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

/// Backward function for automatic differentiation
type BackwardFn = Box<dyn FnOnce(&mut GradientContext)>;

/// Context for managing gradients during backward pass
pub struct GradientContext {
    /// Map from tensor ID to accumulated gradient
    gradients: HashMap<usize, TensorBase>,
    /// Backward functions to execute
    backward_fns: Vec<BackwardFn>,
}

impl GradientContext {
    pub fn new() -> Self {
        GradientContext {
            gradients: HashMap::new(),
            backward_fns: Vec::new(),
        }
    }

    /// Accumulate gradient for a tensor
    pub fn accumulate_grad(&mut self, tensor_id: usize, grad: TensorBase) {
        if let Some(existing_grad) = self.gradients.get_mut(&tensor_id) {
            // Add to existing gradient
            let new_grad = TensorBase::from_binary_op(existing_grad.clone(), grad, |a, b| a + b);
            *existing_grad = new_grad;
        } else {
            self.gradients.insert(tensor_id, grad);
        }
    }

    /// Get gradient for a tensor
    pub fn get_grad(&self, tensor_id: usize) -> Option<&TensorBase> {
        self.gradients.get(&tensor_id)
    }

    /// Register a backward function
    pub fn register_backward(&mut self, backward_fn: BackwardFn) {
        self.backward_fns.push(backward_fn);
    }

    /// Execute all backward functions in reverse order
    pub fn backward(&mut self) {
        while let Some(backward_fn) = self.backward_fns.pop() {
            backward_fn(self);
        }
    }
}

impl Default for GradientContext {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorBase {
    /// Get or create a graph for this tensor.
    fn ensure_graph(&mut self) -> &mut Graph {
        if self.graph.is_none() {
            self.graph = Some(Graph::new());
        }
        self.graph.as_mut().unwrap()
    }

    /// Get the graph node for this tensor, creating one if necessary.
    fn ensure_graph_node(&mut self) -> GraphNode {
        if let Some(ref node) = self.graph_node {
            return node.clone();
        }

        // Create input node from this tensor's data
        let shape_exprs: Vec<_> = self.shape.iter().map(|&s| (s as isize).into()).collect();
        let dtype = self.dtype.clone();

        // Create a graph if needed
        let graph = self.ensure_graph();
        let node = graph.input(dtype, shape_exprs);

        self.graph_node = Some(node.clone());
        node
    }

    /// Create a tensor from a binary operation.
    fn from_binary_op<F>(mut lhs: Self, mut rhs: Self, op: F) -> Self
    where
        F: FnOnce(GraphNode, GraphNode) -> GraphNode,
    {
        assert_eq!(
            lhs.dtype, rhs.dtype,
            "Data types must match for binary operations"
        );
        assert_eq!(
            lhs.backend_name, rhs.backend_name,
            "Backend must match for binary operations"
        );

        let lhs_node = lhs.ensure_graph_node();
        let rhs_node = rhs.ensure_graph_node();

        let result_node = op(lhs_node.clone(), rhs_node.clone());
        let graph = lhs.graph.take().unwrap();
        let backend_name = lhs.backend_name.clone();

        TensorBase::from_graph_node(result_node, graph, backend_name)
    }

    /// Create a tensor from a binary operation with gradient tracking.
    fn from_binary_op_with_grad<F>(
        mut lhs: Self,
        mut rhs: Self,
        op: F,
        grad_fn: Rc<dyn GradFn>,
    ) -> Self
    where
        F: FnOnce(GraphNode, GraphNode) -> GraphNode,
    {
        assert_eq!(
            lhs.dtype, rhs.dtype,
            "Data types must match for binary operations"
        );
        assert_eq!(
            lhs.backend_name, rhs.backend_name,
            "Backend must match for binary operations"
        );

        let lhs_node = lhs.ensure_graph_node();
        let rhs_node = rhs.ensure_graph_node();
        let lhs_id = lhs.id;
        let rhs_id = rhs.id;

        let result_node = op(lhs_node.clone(), rhs_node.clone());
        let graph = lhs.graph.take().unwrap();
        let backend_name = lhs.backend_name.clone();

        let mut result = TensorBase::from_graph_node(result_node.clone(), graph, backend_name);

        // Create autograd metadata if either input requires grad
        if lhs.requires_grad || rhs.requires_grad {
            let inputs = vec![(lhs_id, lhs_node), (rhs_id, rhs_node)];
            result.autograd_meta = Some(TensorMeta::non_leaf(result_node, grad_fn, inputs));
            result.requires_grad = true;
        }

        result
    }

    /// Create a tensor from a unary operation.
    fn from_unary_op<F>(mut self, op: F) -> Self
    where
        F: FnOnce(GraphNode) -> GraphNode,
    {
        let input_node = self.ensure_graph_node();
        let result_node = op(input_node);
        let graph = self.graph.take().unwrap();
        let backend_name = self.backend_name.clone();

        TensorBase::from_graph_node(result_node, graph, backend_name)
    }

    /// Create a tensor from a unary operation with gradient tracking.
    fn from_unary_op_with_grad<F>(mut self, op: F, grad_fn: Rc<dyn GradFn>) -> Self
    where
        F: FnOnce(GraphNode) -> GraphNode,
    {
        let input_node = self.ensure_graph_node();
        let input_id = self.id;
        let requires_grad = self.requires_grad;

        let result_node = op(input_node.clone());
        let graph = self.graph.take().unwrap();
        let backend_name = self.backend_name.clone();

        let mut result = TensorBase::from_graph_node(result_node.clone(), graph, backend_name);

        // Create autograd metadata if input requires grad
        if requires_grad {
            let inputs = vec![(input_id, input_node)];
            result.autograd_meta = Some(TensorMeta::non_leaf(result_node, grad_fn, inputs));
            result.requires_grad = true;
        }

        result
    }
}

// Arithmetic operators for Tensor<T, D>

impl<T: TensorType, D: Dimension> Add for Tensor<T, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let result_inner = TensorBase::from_binary_op_with_grad(
            self.inner,
            rhs.inner,
            |a, b| a + b,
            Rc::new(AddBackward),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }
}

impl<T: TensorType, D: Dimension> Add for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs_clone = self.inner.clone();
        let rhs_clone = rhs.inner.clone();
        let result_inner = TensorBase::from_binary_op_with_grad(
            lhs_clone,
            rhs_clone,
            |a, b| a + b,
            Rc::new(AddBackward),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }
}

impl<T: TensorType, D: Dimension> Sub for Tensor<T, D> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        // Implement as add(a, neg(b))
        // This automatically creates the correct gradient graph: AddBackward + NegBackward
        self + (-rhs)
    }
}

impl<T: TensorType, D: Dimension> Sub for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        // Implement as add(a, neg(b))
        // This automatically creates the correct gradient graph: AddBackward + NegBackward
        self.clone() - rhs.clone()
    }
}

impl<T: TensorType, D: Dimension> Mul for Tensor<T, D> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let result_inner = TensorBase::from_binary_op_with_grad(
            self.inner,
            rhs.inner,
            |a, b| a * b,
            Rc::new(MulBackward),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }
}

impl<T: TensorType, D: Dimension> Mul for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_clone = self.inner.clone();
        let rhs_clone = rhs.inner.clone();
        let result_inner = TensorBase::from_binary_op_with_grad(
            lhs_clone,
            rhs_clone,
            |a, b| a * b,
            Rc::new(MulBackward),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }
}

impl<T: TensorType, D: Dimension> Div for Tensor<T, D> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        // Implement as mul(a, recip(b))
        // This automatically creates the correct gradient graph: MulBackward + RecipBackward
        self * rhs.recip()
    }
}

impl<T: TensorType, D: Dimension> Div for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn div(self, rhs: Self) -> Self::Output {
        // Implement as mul(a, recip(b))
        // This automatically creates the correct gradient graph: MulBackward + RecipBackward
        self.clone() / rhs.clone()
    }
}

impl<T: TensorType, D: Dimension> Neg for Tensor<T, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let result_inner =
            TensorBase::from_unary_op_with_grad(self.inner, |a| -a, Rc::new(NegBackward));
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }
}

impl<T: TensorType, D: Dimension> Neg for &Tensor<T, D> {
    type Output = Tensor<T, D>;

    fn neg(self) -> Self::Output {
        let clone = self.inner.clone();
        let result_inner = TensorBase::from_unary_op_with_grad(clone, |a| -a, Rc::new(NegBackward));
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }
}

// Additional mathematical operations
impl<T: TensorType, D: Dimension> Tensor<T, D> {
    /// Compute the reciprocal (1/x) of each element.
    pub fn recip(self) -> Self {
        let result_inner =
            TensorBase::from_unary_op_with_grad(self.inner, |a| a.recip(), Rc::new(RecipBackward));
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute the sine of each element.
    pub fn sin(self) -> Self {
        let result_inner = TensorBase::from_unary_op(self.inner, |a| a.sin());
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute the square root of each element.
    pub fn sqrt(self) -> Self {
        let result_inner = TensorBase::from_unary_op(self.inner, |a| a.sqrt());
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute the base-2 logarithm of each element.
    pub fn log2(self) -> Self {
        let result_inner = TensorBase::from_unary_op(self.inner, |a| a.log2());
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute 2 raised to the power of each element.
    pub fn exp2(self) -> Self {
        let result_inner = TensorBase::from_unary_op(self.inner, |a| a.exp2());
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute the natural logarithm of each element.
    /// Implemented as log2(x) / log2(e).
    pub fn ln(self) -> Self {
        // log2(e) ≈ 1.4426950408889634
        let log2_e_recip = 1.0f32 / 1.442_695_f32;
        let result_inner =
            TensorBase::from_unary_op(self.inner, move |a| a.log2() * GraphNode::f32(log2_e_recip));
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute e raised to the power of each element.
    /// Implemented as exp2(x * log2(e)).
    pub fn exp(self) -> Self {
        // log2(e) ≈ 1.4426950408889634
        let log2_e = 1.442_695_f32;
        let result_inner =
            TensorBase::from_unary_op(self.inner, move |a| (a * GraphNode::f32(log2_e)).exp2());
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute the maximum of two tensors element-wise.
    pub fn max(self, rhs: Self) -> Self {
        let result_inner = TensorBase::from_binary_op(self.inner, rhs.inner, |a, b| a.cmp_max(b));
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Compute x raised to the power of y for each element.
    /// Implemented as exp2(y * log2(x)).
    pub fn pow(self, exponent: Self) -> Self {
        let result_inner = TensorBase::from_binary_op(self.inner, exponent.inner, |base, exp| {
            (exp * base.log2()).exp2()
        });
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Sum all elements along the specified axis.
    /// Returns a tensor with dynamic dimensions since the shape changes.
    pub fn sum(self, axis: usize) -> Tensor<T, Dyn> {
        assert!(axis < self.inner.ndim(), "axis out of bounds");

        let result_inner = TensorBase::from_unary_op_with_grad(
            self.inner,
            move |a| a.sum(axis),
            Rc::new(SumBackward { axis }),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Product of all elements along the specified axis.
    /// Returns a tensor with dynamic dimensions since the shape changes.
    pub fn product(self, axis: usize) -> Tensor<T, Dyn> {
        assert!(axis < self.inner.ndim(), "axis out of bounds");

        let result_inner = TensorBase::from_unary_op_with_grad(
            self.inner,
            move |a| a.product(axis),
            Rc::new(ProductBackward { axis }),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Maximum element along the specified axis.
    /// Returns a tensor with dynamic dimensions since the shape changes.
    pub fn reduce_max(self, axis: usize) -> Tensor<T, Dyn> {
        assert!(axis < self.inner.ndim(), "axis out of bounds");

        let result_inner = TensorBase::from_unary_op_with_grad(
            self.inner,
            move |a| a.reduce(ReduceOp::Max, axis),
            Rc::new(MaxBackward { axis }),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Cumulative sum along the specified axis.
    /// Shape is preserved.
    pub fn cumsum(self, axis: usize) -> Self {
        assert!(axis < self.inner.ndim(), "axis out of bounds");

        let result_inner = TensorBase::from_unary_op_with_grad(
            self.inner,
            move |a| a.cumulative(CumulativeOp::Add, axis),
            Rc::new(CumsumBackward { axis }),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Cumulative product along the specified axis.
    /// Shape is preserved.
    pub fn cumprod(self, axis: usize) -> Self {
        assert!(axis < self.inner.ndim(), "axis out of bounds");

        let result_inner = TensorBase::from_unary_op_with_grad(
            self.inner,
            move |a| a.cumulative(CumulativeOp::Mul, axis),
            Rc::new(CumprodBackward { axis }),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Cumulative maximum along the specified axis.
    /// Shape is preserved.
    pub fn cummax(self, axis: usize) -> Self {
        assert!(axis < self.inner.ndim(), "axis out of bounds");

        let result_inner = TensorBase::from_unary_op_with_grad(
            self.inner,
            move |a| a.cumulative(CumulativeOp::Max, axis),
            Rc::new(CummaxBackward { axis }),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }

    /// Create a sliding window view for convolution operations.
    ///
    /// This operation adds a new dimension for sliding windows, transforming the shape.
    /// For example, [B, C, L] with window_size=K, stride=S becomes [B, C, L', K]
    /// where L' = (L - K) / S + 1.
    ///
    /// Returns a tensor with dynamic dimensions since the shape changes.
    ///
    /// # Arguments
    /// * `dim` - The dimension along which to create sliding windows
    /// * `window_size` - The size of each window
    /// * `stride` - The stride between windows
    ///
    /// # Example
    /// ```ignore
    /// // For Conv1d: input [B, C_in, L], kernel [C_out, C_in, K]
    /// let windowed = input.sliding_window(2, kernel_size, stride); // → [B, C_in, L', K]
    /// // Then combine with kernel for convolution
    /// ```
    pub fn sliding_window(self, dim: usize, window_size: usize, stride: usize) -> Tensor<T, Dyn> {
        assert!(dim < self.inner.ndim(), "dimension out of bounds");

        let result_inner = TensorBase::from_unary_op_with_grad(
            self.inner,
            move |a| a.sliding_window(dim, window_size, stride),
            Rc::new(SlidingWindowBackward {
                dim,
                window_size,
                stride,
            }),
        );
        Tensor {
            inner: result_inner,
            _phantom: PhantomData,
        }
    }
}

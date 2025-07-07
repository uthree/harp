use crate::{
    graph::Graph,
    node::Node,
    operator::{self, Operator},
    shape::tracker::ShapeTracker,
};
use ndarray::ArrayD;
use petgraph::graph::NodeIndex;
use std::{
    ops::{Add, Mul, Rem},
    sync::{Arc, Mutex},
};

/// Represents the actual data of a tensor.
///
/// This is a wrapper around `ndarray::ArrayD<f32>` to provide a distinct type
/// for tensor data within the computation graph.
#[derive(Clone, Debug)]
pub struct TensorData(pub ArrayD<f32>);

/// Represents a tensor in the computation graph.
///
/// A `Tensor` is a symbolic representation of a multi-dimensional array.
/// It holds a reference to the computation graph, its corresponding node index
/// within that graph, and its shape information.
#[derive(Clone)]
pub struct Tensor {
    /// A shared reference to the computation graph this tensor belongs to.
    pub graph: Arc<Mutex<Graph>>,
    /// The index of the node in the graph that produces this tensor.
    pub node_index: NodeIndex,
    /// The shape and stride information of this tensor.
    pub shape: ShapeTracker,
}

impl Tensor {
    /// Creates a new tensor and adds it as a node to the computation graph.
    ///
    /// This is a private helper function used by other tensor operations.
    ///
    /// # Arguments
    ///
    /// * `graph` - A shared reference to the graph.
    /// * `shape` - The `ShapeTracker` for the new tensor.
    /// * `op` - The `Operator` that produces this tensor.
    /// * `inputs` - A slice of references to input tensors for this operation.
    ///
    /// # Returns
    ///
    /// A new `Tensor` instance.
    fn new(
        graph: Arc<Mutex<Graph>>,
        shape: ShapeTracker,
        op: impl Operator + 'static,
        inputs: &[&Tensor],
    ) -> Self {
        let mut graph_mut = graph.lock().unwrap();
        let node = Node::new(op, shape.clone());
        let node_index = graph_mut.add_node(node);

        for (i, input) in inputs.iter().enumerate() {
            graph_mut.add_edge(input.node_index, node_index, i);
        }

        Self {
            graph: graph.clone(),
            node_index,
            shape,
        }
    }

    // --- Unary Ops ---

    /// Applies the base-2 exponential function (2^x) to the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the result of the operation.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use harp::graph::Graph;
    /// use harp::shape::tracker::ShapeTracker;
    ///
    /// let graph_arc = Arc::new(Mutex::new(Graph::new()));
    /// let input_shape = ShapeTracker::new(vec![2, 2]);
    /// let input_tensor = Graph::new_input(graph_arc.clone(), input_shape);
    /// let result_tensor = input_tensor.exp2();
    /// // The graph now contains nodes for input and exp2 operation.
    /// ```
    pub fn exp2(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Exp2,
            &[self],
        )
    }

    /// Applies the base-2 logarithm function (log2(x)) to the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the result of the operation.
    pub fn log2(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Log2,
            &[self],
        )
    }

    /// Applies the sine function (sin(x)) to the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the result of the operation.
    pub fn sin(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Sin,
            &[self],
        )
    }

    /// Applies the square root function (sqrt(x)) to the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the result of the operation.
    pub fn sqrt(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Sqrt,
            &[self],
        )
    }

    /// Applies the reciprocal function (1/x) to the tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the result of the operation.
    pub fn recip(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Recip,
            &[self],
        )
    }

    // --- Reduce Ops ---

    /// Performs a sum reduction along the specified dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to sum.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the result of the sum reduction.
    pub fn sum_reduce(&self, dim: usize) -> Self {
        // TODO: Calculate output shape properly
        let new_shape = self.shape.clone();
        Self::new(
            self.graph.clone(),
            new_shape,
            operator::SumReduce { dim },
            &[self],
        )
    }

    /// Performs a maximum reduction along the specified dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to find the maximum.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the result of the max reduction.
    pub fn max_reduce(&self, dim: usize) -> Self {
        // TODO: Calculate output shape properly
        let new_shape = self.shape.clone();
        Self::new(
            self.graph.clone(),
            new_shape,
            operator::MaxReduce { dim },
            &[self],
        )
    }
}

// --- Binary Ops ---

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    /// Overloads the `+` operator for tensor addition.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the sum of the two tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use harp::graph::Graph;
    /// use harp::shape::tracker::ShapeTracker;
    ///
    /// let graph_arc = Arc::new(Mutex::new(Graph::new()));
    /// let shape = ShapeTracker::new(vec![2, 2]);
    /// let a = Graph::new_input(graph_arc.clone(), shape.clone());
    /// let b = Graph::new_input(graph_arc.clone(), shape.clone());
    /// let c = &a + &b;
    /// // The graph now contains nodes for inputs and an addition operation.
    /// ```
    fn add(self, rhs: &'b Tensor) -> Self::Output {
        // TODO: Broadcasting and shape calculation
        let new_shape = self.shape.clone();
        Tensor::new(self.graph.clone(), new_shape, operator::Add, &[self, rhs])
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    /// Overloads the `*` operator for tensor multiplication.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the product of the two tensors.
    fn mul(self, rhs: &'b Tensor) -> Self::Output {
        // TODO: Broadcasting and shape calculation
        let new_shape = self.shape.clone();
        Tensor::new(self.graph.clone(), new_shape, operator::Mul, &[self, rhs])
    }
}

impl<'a, 'b> Rem<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    /// Overloads the `%` operator for tensor remainder.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` representing the remainder of the two tensors.
    fn rem(self, rhs: &'b Tensor) -> Self::Output {
        // TODO: Broadcasting and shape calculation
        let new_shape = self.shape.clone();
        Tensor::new(self.graph.clone(), new_shape, operator::Rem, &[self, rhs])
    }
}

// Note: LessThan does not have a direct std::ops trait, so it's a method.
impl Tensor {
    /// Performs an element-wise less than comparison.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The right-hand side tensor.
    ///
    /// # Returns
    ///
    /// A new `Tensor` with 1.0 where `self` is less than `rhs`, and 0.0 otherwise.
    pub fn less_than(&self, rhs: &Tensor) -> Tensor {
        // TODO: Broadcasting and shape calculation
        let new_shape = self.shape.clone();
        Tensor::new(
            self.graph.clone(),
            new_shape,
            operator::LessThan,
            &[self, rhs],
        )
    }
}

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

#[derive(Clone, Debug)]
pub struct TensorData(pub ArrayD<f32>);

#[derive(Clone)]
pub struct Tensor {
    pub graph: Arc<Mutex<Graph>>,
    pub node_index: NodeIndex,
    pub shape: ShapeTracker,
}

impl Tensor {
    /// Creates a new tensor and adds it to the graph.
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
    pub fn exp2(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Exp2,
            &[self],
        )
    }
    pub fn log2(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Log2,
            &[self],
        )
    }
    pub fn sin(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Sin,
            &[self],
        )
    }
    pub fn sqrt(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Sqrt,
            &[self],
        )
    }
    pub fn recip(&self) -> Self {
        Self::new(
            self.graph.clone(),
            self.shape.clone(),
            operator::Recip,
            &[self],
        )
    }

    // --- Reduce Ops ---
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
    fn add(self, rhs: &'b Tensor) -> Self::Output {
        // TODO: Broadcasting and shape calculation
        let new_shape = self.shape.clone();
        Tensor::new(self.graph.clone(), new_shape, operator::Add, &[self, rhs])
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &'b Tensor) -> Self::Output {
        // TODO: Broadcasting and shape calculation
        let new_shape = self.shape.clone();
        Tensor::new(self.graph.clone(), new_shape, operator::Mul, &[self, rhs])
    }
}

impl<'a, 'b> Rem<&'b Tensor> for &'a Tensor {
    type Output = Tensor;
    fn rem(self, rhs: &'b Tensor) -> Self::Output {
        // TODO: Broadcasting and shape calculation
        let new_shape = self.shape.clone();
        Tensor::new(self.graph.clone(), new_shape, operator::Rem, &[self, rhs])
    }
}

// Note: LessThan does not have a direct std::ops trait, so it's a method.
impl Tensor {
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

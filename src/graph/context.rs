use std::cell::RefCell;

use crate::{
    ast::{Const, DType},
    graph::{
        node::{NodeData, NodeId},
        op::GraphOp,
        shape::expr::Expr,
        view::NodeView,
    },
};

/// Owns all the nodes of a computation graph.
///
/// The `Graph` uses interior mutability (`RefCell`) to allow nodes to be added
/// dynamically while maintaining immutable references to the graph itself.
#[derive(Default, Debug)]
pub struct Graph {
    /// A vector holding the data for all nodes in the graph.
    pub nodes: RefCell<Vec<NodeData>>,
    /// A list of node IDs that are considered inputs to the graph.
    pub inputs: RefCell<Vec<NodeId>>,
    /// A list of node IDs that are considered outputs of the graph.
    pub outputs: RefCell<Vec<NodeId>>,
}

impl Graph {
    /// Creates a new, empty computation graph.
    pub fn new() -> Self {
        Graph {
            nodes: RefCell::new(Vec::new()),
            inputs: RefCell::new(Vec::new()),
            outputs: RefCell::new(Vec::new()),
        }
    }

    /// Adds a new node to the graph. This is an internal method.
    pub fn add_node(
        &self,
        op: GraphOp,
        src: Vec<NodeId>,
        dtype: DType,
        shape: Vec<Expr>,
    ) -> NodeId {
        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len();
        nodes.push(NodeData {
            op,
            src,
            dtype,
            shape,
        });
        NodeId(id)
    }

    /// Adds a new input node to the graph.
    ///
    /// # Arguments
    ///
    /// * `dtype` - The data type of the input tensor.
    /// * `shape` - The symbolic shape of the input tensor.
    pub fn input(&self, dtype: DType, shape: Vec<Expr>) -> NodeView<'_> {
        let id = self.add_node(GraphOp::Input, vec![], dtype, shape);
        self.inputs.borrow_mut().push(id);
        self.get_view(id)
    }

    /// Creates a new tensor filled with a constant value.
    pub fn full<T: Into<Const>>(&self, value: T, shape: Vec<Expr>) -> NodeView<'_> {
        let constant: Const = value.into();
        let dtype = constant.dtype();
        let id = self.add_node(GraphOp::Full(constant), vec![], dtype, shape);
        self.get_view(id)
    }

    /// Creates a new tensor with the given shape, filled with random values.
    pub fn rand(&self, dtype: DType, shape: Vec<Expr>) -> NodeView<'_> {
        let id = self.add_node(GraphOp::Rand, vec![], dtype, shape);
        self.get_view(id)
    }

    /// Gets a `NodeView` for a given `NodeId`.
    pub fn get_view(&self, id: NodeId) -> NodeView<'_> {
        NodeView { id, graph: self }
    }
}

impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        *self.nodes.borrow() == *other.nodes.borrow()
            && *self.inputs.borrow() == *other.inputs.borrow()
            && *self.outputs.borrow() == *other.outputs.borrow()
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Graph {
            nodes: RefCell::new(self.nodes.borrow().clone()),
            inputs: RefCell::new(self.inputs.borrow().clone()),
            outputs: RefCell::new(self.outputs.borrow().clone()),
        }
    }
}

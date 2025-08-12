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

    /// Builds a `Graph` from a final `Tensor`.
    pub fn from_tensor(tensor: &crate::tensor::Tensor) -> (Self, std::collections::HashMap<usize, NodeId>) {
        let graph = Self::new();
        let mut tensor_to_node = std::collections::HashMap::new();

        fn build_recursive(
            tensor: &crate::tensor::Tensor,
            graph: &Graph,
            tensor_to_node: &mut std::collections::HashMap<usize, NodeId>,
        ) -> NodeId {
            if let Some(&node_id) = tensor_to_node.get(&tensor.id()) {
                return node_id;
            }

            let t_data = tensor.0.borrow();
            let mut src_nodes = Vec::new();
            for src_tensor in &t_data.src {
                src_nodes.push(build_recursive(src_tensor, graph, tensor_to_node));
            }

            let shape_exprs: Vec<Expr> = t_data
                .shape
                .iter()
                .map(|&dim| Expr::from(dim))
                .collect();

            let op = crate::tensor::op_conversion::op_to_graph_op(
                t_data.op.clone(),
                graph,
                src_nodes,
                shape_exprs,
                t_data.dtype.clone(),
            );

            tensor_to_node.insert(tensor.id(), op.id);
            op.id
        }

        let final_node_id = build_recursive(tensor, &graph, &mut tensor_to_node);
        graph.get_view(final_node_id).as_output();

        (graph, tensor_to_node)
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

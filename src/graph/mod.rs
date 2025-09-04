use crate::ast::{ConstLiteral, DType};
pub mod ops;
pub mod shape;
use crate::graph::shape::Expr as ShapeExpr;

// --- New Arena-based Graph Structures ---

/// A handle to a node in the graph.
/// It's a lightweight, copyable identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub usize);

/// Represents a single operation or value in the computation graph.
/// Nodes are stored in the `Graph`'s arena.
#[derive(Debug, Clone)]
pub struct Node {
    pub op: GraphOp,
    pub inputs: Vec<NodeId>,
    pub dtype: DType,
    // We can add more metadata here later, e.g., shape, name.
}

/// The main computation graph structure.
/// It owns all the nodes and manages the graph structure.
#[derive(Default, Debug, Clone)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub signature: GraphSignature,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
}

impl Graph {
    /// Creates a new, empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new node to the graph and returns its ID.
    pub fn add_node(&mut self, op: GraphOp, inputs: Vec<NodeId>, dtype: DType) -> NodeId {
        let id = NodeId(self.nodes.len());
        self.nodes.push(Node { op, inputs, dtype });
        id
    }

    /// Retrieves a reference to a node by its ID.
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.0]
    }

    /// Retrieves a mutable reference to a node by its ID.
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.0]
    }

    /// Creates a new input node and adds it to the graph's input list.
    pub fn input(&mut self, shape: Vec<ShapeExpr>, dtype: DType) -> NodeId {
        // Store the signature for this input tensor.
        let tensor_signature = TensorSignature {
            dtype: dtype.clone(),
            shape,
        };
        self.signature.inputs.push(tensor_signature);

        // Create the input node in the graph arena.
        let input_node_id = self.add_node(GraphOp::Input, vec![], dtype);
        self.inputs.push(input_node_id);
        input_node_id
    }
}

// --- Graph-related Definitions ---

#[derive(Default, Debug, Clone, PartialEq)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>,
    pub inputs: Vec<TensorSignature>,
    pub outputs: Vec<TensorSignature>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    pub name: String,
    pub default: isize,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct TensorSignature {
    pub dtype: DType,
    pub shape: Vec<ShapeExpr>,
}

#[derive(Debug, Clone)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Max,
    Neg,
    Recip,
    Sqrt,
    Sin,
    Log2,
    Exp2,
    Rem,
}

#[derive(Debug, Clone)]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

#[derive(Debug, Clone)]
pub enum GraphOp {
    Input,
    Const(ConstLiteral, Vec<ShapeExpr>),
    Cast,
    Rand(Vec<ShapeExpr>),
    Arange(usize),
    Reshape(Vec<ShapeExpr>),
    Reduce(ReduceOp, Vec<usize>),
    Cumulative(ReduceOp, usize),
    Contiguous,
}

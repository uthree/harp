use crate::{
    ast::DType,
    graph::{op::GraphOp, shape::expr::Expr},
};

/// A unique identifier for a node within a `Graph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

/// The data associated with a single node in the computation graph.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeData {
    /// The operation performed by this node.
    pub op: GraphOp,
    /// The `NodeId`s of the input nodes to this operation.
    pub src: Vec<NodeId>,
    /// The data type of the tensor produced by this node.
    pub dtype: DType,
    /// The symbolic shape of the tensor.
    pub shape: Vec<Expr>,
}

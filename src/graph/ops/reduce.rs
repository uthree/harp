use crate::{
    ast::AstOp,
    graph::{context::Graph, node::NodeId, op::GraphOp},
};

pub trait ReduceOps {
    fn _reduce(&self, op: AstOp, src: NodeId, axis: usize) -> NodeId;
    fn sum(&self, src: NodeId, axis: usize) -> NodeId;
    fn max(&self, src: NodeId, axis: usize) -> NodeId;
    fn prod(&self, src: NodeId, axis: usize) -> NodeId;
    fn cumsum(&self, src: NodeId, axis: usize) -> NodeId;
}

impl ReduceOps for Graph {
    /// Internal helper to create a reduction node.
    ///
    /// # Arguments
    ///
    /// * `op` - The reduction operation (e.g., `AstOp::Add`, `AstOp::Max`).
    /// * `src` - The `NodeId` of the input tensor.
    /// * `axis` - The axis along which to perform the reduction.
    ///
    /// # Panics
    ///
    /// Panics if the `axis` is out of bounds for the input tensor's shape.
    fn _reduce(&self, op: AstOp, src: NodeId, axis: usize) -> NodeId {
        let (dtype, mut shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        assert!(axis < shape.len(), "Reduction axis out of bounds");
        shape.remove(axis);
        self.add_node(GraphOp::Reduce(op, axis), vec![src], dtype, shape)
    }

    fn sum(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Add, src, axis)
    }

    fn max(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Max, src, axis)
    }

    fn prod(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Mul, src, axis)
    }

    fn cumsum(&self, src: NodeId, axis: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        assert!(axis < shape.len(), "Cumulative axis out of bounds");
        self.add_node(
            GraphOp::Cumulative(AstOp::Add, axis),
            vec![src],
            dtype,
            shape,
        )
    }
}

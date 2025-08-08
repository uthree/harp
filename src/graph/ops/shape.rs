use crate::{
    graph::{
        context::Graph,
        node::NodeId,
        op::GraphOp,
        shape::{expr::Expr, tracker::ShapeTracker},
    },
};

pub trait ShapeOps {
    fn permute(&self, src: NodeId, axes: Vec<usize>) -> NodeId;
    fn contiguous(&self, src: NodeId) -> NodeId;
    fn squeeze(&self, src: NodeId, axis: usize) -> NodeId;
    fn unsqueeze(&self, src: NodeId, axis: usize) -> NodeId;
    fn expand(&self, src: NodeId, new_shape: Vec<Expr>) -> NodeId;
    fn slice(&self, src: NodeId, args: Vec<(Expr, Expr)>) -> NodeId;
    fn reshape(&self, src: NodeId, new_shape: Vec<Expr>) -> NodeId;
}

impl ShapeOps for Graph {
    fn permute(&self, src: NodeId, axes: Vec<usize>) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.permute(axes.clone()).shape().to_vec();
        self.add_node(GraphOp::Permute(axes), vec![src], dtype, new_shape)
    }

    fn contiguous(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Contiguous, vec![src], dtype, shape)
    }

    fn squeeze(&self, src: NodeId, axis: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.squeeze(axis).shape().to_vec();
        self.add_node(GraphOp::Squeeze(axis), vec![src], dtype, new_shape)
    }

    fn unsqueeze(&self, src: NodeId, axis: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.unsqueeze(axis).shape().to_vec();
        self.add_node(GraphOp::Unsqueeze(axis), vec![src], dtype, new_shape)
    }

    fn expand(&self, src: NodeId, new_shape: Vec<Expr>) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        // This just validates the expand operation. The final shape is `new_shape`.
        let _ = tracker.expand(new_shape.clone());
        self.add_node(
            GraphOp::Expand(new_shape.clone()),
            vec![src],
            dtype,
            new_shape,
        )
    }

    fn slice(&self, src: NodeId, args: Vec<(Expr, Expr)>) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.slice(&args).shape().to_vec();
        self.add_node(GraphOp::Slice(args), vec![src], dtype, new_shape)
    }

    fn reshape(&self, src: NodeId, new_shape: Vec<Expr>) -> NodeId {
        let dtype = self.nodes.borrow()[src.0].dtype.clone();
        // Reshape requires the memory to be contiguous. We conservatively insert a
        // contiguous call here. The lowerer for `contiguous` will handle the
        // case where the tensor is already contiguous as a no-op.
        let contiguous_src = self.contiguous(src);
        self.add_node(
            GraphOp::Reshape(new_shape.clone()),
            vec![contiguous_src],
            dtype,
            new_shape,
        )
    }
}
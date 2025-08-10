use crate::{
    ast::{AstNode, AstOp, DType},
    graph::{context::Graph, node::NodeId, op::GraphOp},
};

pub trait ElementwiseOps {
    fn add(&self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn sub(&self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn mul(&self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn div(&self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn rem(&self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn lt(&self, lhs: NodeId, rhs: NodeId) -> NodeId;
    fn neg(&self, src: NodeId) -> NodeId;
    fn sin(&self, src: NodeId) -> NodeId;
    fn cos(&self, src: NodeId) -> NodeId;
    fn sqrt(&self, src: NodeId) -> NodeId;
    fn log2(&self, src: NodeId) -> NodeId;
    fn exp2(&self, src: NodeId) -> NodeId;
    fn recip(&self, src: NodeId) -> NodeId;
}

impl ElementwiseOps for Graph {
    fn add(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, lhs_shape, rhs_shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
                rhs_node.shape.clone(),
            )
        };
        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in add: {lhs_shape:?} vs {rhs_shape:?}");
        }
        let ast_node = AstNode::capture(0, lhs_dtype) + AstNode::capture(1, rhs_dtype);
        self.add_node(
            GraphOp::Elementwise(AstOp::Add),
            vec![lhs, rhs],
            ast_node.dtype,
            lhs_shape,
        )
    }

    fn sub(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, lhs_shape, rhs_shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
                rhs_node.shape.clone(),
            )
        };
        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in sub: {lhs_shape:?} vs {rhs_shape:?}");
        }
        let ast_node = AstNode::capture(0, lhs_dtype) - AstNode::capture(1, rhs_dtype);
        self.add_node(
            GraphOp::Elementwise(AstOp::Sub),
            vec![lhs, rhs],
            ast_node.dtype,
            lhs_shape,
        )
    }

    fn mul(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, lhs_shape, rhs_shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
                rhs_node.shape.clone(),
            )
        };
        if lhs_shape != rhs_shape {
            panic!("Shape mismatch in mul: {lhs_shape:?} vs {rhs_shape:?}");
        }
        let ast_node = AstNode::capture(0, lhs_dtype) * AstNode::capture(1, rhs_dtype);
        self.add_node(
            GraphOp::Elementwise(AstOp::Mul),
            vec![lhs, rhs],
            ast_node.dtype,
            lhs_shape,
        )
    }

    fn rem(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            )
        };
        let ast_node = AstNode::capture(0, lhs_dtype) % AstNode::capture(1, rhs_dtype);
        self.add_node(
            GraphOp::Elementwise(AstOp::Rem),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    fn lt(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            )
        };
        // The result of a comparison is usually a boolean, but we'll represent it
        // as the same float type (0.0 or 1.0) for simplicity in the backend.
        let ast_node = AstNode::new(
            AstOp::LessThan,
            vec![
                AstNode::capture(0, lhs_dtype),
                AstNode::capture(1, rhs_dtype),
            ],
            DType::F32, // FIXME: This should probably be a boolean type
        );
        self.add_node(
            GraphOp::Elementwise(AstOp::LessThan),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    fn div(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let recip_rhs = self.recip(rhs);
        self.mul(lhs, recip_rhs)
    }

    fn neg(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let ast_node = -AstNode::capture(0, dtype);
        self.add_node(
            GraphOp::Elementwise(AstOp::Neg),
            vec![src],
            ast_node.dtype,
            shape,
        )
    }

    fn sin(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Sin), vec![src], dtype, shape)
    }

    fn cos(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Cos), vec![src], dtype, shape)
    }

    fn sqrt(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Sqrt), vec![src], dtype, shape)
    }

    fn log2(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Log2), vec![src], dtype, shape)
    }

    fn exp2(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Exp2), vec![src], dtype, shape)
    }

    fn recip(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Recip), vec![src], dtype, shape)
    }
}

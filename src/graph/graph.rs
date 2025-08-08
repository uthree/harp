use std::cell::RefCell;

use crate::{
    ast::{AstNode, AstOp, Const, DType},
    graph::{
        node::{NodeData, NodeId},
        op::GraphOp,
        shape::{expr::Expr, tracker::ShapeTracker},
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

    // --- Internal methods for creating operation nodes ---

    pub fn add(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
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

    pub fn sub(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let neg_rhs = self.neg(rhs);
        self.add(lhs, neg_rhs)
    }

    pub fn mul(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
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

    pub fn rem(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
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

    pub fn lt(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
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

    pub fn div(&self, lhs: NodeId, rhs: NodeId) -> NodeId {
        let recip_rhs = self.recip(rhs);
        self.mul(lhs, recip_rhs)
    }

    pub fn neg(&self, src: NodeId) -> NodeId {
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

    pub fn sin(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Sin), vec![src], dtype, shape)
    }

    pub fn sqrt(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Sqrt), vec![src], dtype, shape)
    }

    pub fn log2(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Log2), vec![src], dtype, shape)
    }

    pub fn exp2(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Exp2), vec![src], dtype, shape)
    }

    pub fn recip(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Elementwise(AstOp::Recip), vec![src], dtype, shape)
    }

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

    pub fn sum(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Add, src, axis)
    }

    pub fn max(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Max, src, axis)
    }

    pub fn prod(&self, src: NodeId, axis: usize) -> NodeId {
        self._reduce(AstOp::Mul, src, axis)
    }

    pub fn cumsum(&self, src: NodeId, axis: usize) -> NodeId {
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

    pub fn permute(&self, src: NodeId, axes: Vec<usize>) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.permute(axes.clone()).shape().to_vec();
        self.add_node(GraphOp::Permute(axes), vec![src], dtype, new_shape)
    }

    pub fn contiguous(&self, src: NodeId) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        self.add_node(GraphOp::Contiguous, vec![src], dtype, shape)
    }

    pub fn squeeze(&self, src: NodeId, axis: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.squeeze(axis).shape().to_vec();
        self.add_node(GraphOp::Squeeze(axis), vec![src], dtype, new_shape)
    }

    pub fn unsqueeze(&self, src: NodeId, axis: usize) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.unsqueeze(axis).shape().to_vec();
        self.add_node(GraphOp::Unsqueeze(axis), vec![src], dtype, new_shape)
    }

    pub fn expand(&self, src: NodeId, new_shape: Vec<Expr>) -> NodeId {
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

    pub fn slice(&self, src: NodeId, args: Vec<(Expr, Expr)>) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.slice(&args).shape().to_vec();
        self.add_node(GraphOp::Slice(args), vec![src], dtype, new_shape)
    }

    pub fn unfold(
        &self,
        src: NodeId,
        dim: usize,
        kernel_size: usize,
        stride: usize,
    ) -> NodeId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let tracker = ShapeTracker::new(shape);
        let new_shape = tracker.unfold(dim, kernel_size, stride).shape().to_vec();
        self.add_node(
            GraphOp::Unfold {
                dim,
                kernel_size,
                stride,
            },
            vec![src],
            dtype,
            new_shape,
        )
    }

    pub fn conv1d(
        &self,
        input: NodeId,
        weight: NodeId,
        kernel_size: usize,
        stride: usize,
        groups: usize,
    ) -> NodeId {
        assert_eq!(groups, 1, "Grouped convolution not yet supported");

        // 1. Unfold input
        let x_unfolded = self.unfold(input, 2, kernel_size, stride);

        // 2. Reshape for broadcast
        // x_unfolded: [N, C_in, L_out, K] -> [N, 1, C_in, L_out, K]
        let x_reshaped = self.unsqueeze(x_unfolded, 1);

        // weight: [C_out, C_in, K] -> [1, C_out, C_in, 1, K]
        let w_reshaped_1 = self.unsqueeze(weight, 0);
        let w_reshaped_2 = self.unsqueeze(w_reshaped_1, 3);

        // 3. Expand
        // Get shapes for expand
        let (x_shape, w_shape) = {
            let nodes = self.nodes.borrow();
            let x_node = &nodes[x_reshaped.0];
            let w_node = &nodes[w_reshaped_2.0];
            (x_node.shape.clone(), w_node.shape.clone())
        };
        // x_shape: [N, 1, C_in, L_out, K]
        // w_shape: [1, C_out, C_in, 1, K]
        let n = x_shape[0].clone();
        let c_out = w_shape[1].clone();
        let c_in = x_shape[2].clone();
        let l_out = x_shape[3].clone();
        let k = x_shape[4].clone();

        let broadcast_shape = vec![n, c_out, c_in, l_out, k];
        let x_expanded = self.expand(x_reshaped, broadcast_shape.clone());
        let w_expanded = self.expand(w_reshaped_2, broadcast_shape);

        // 4. Multiply
        let mul_result = self.mul(x_expanded, w_expanded);

        // 5. Sum reduction
        // sum over K (axis=4) -> [N, C_out, C_in, L_out]
        let sum_k = self.sum(mul_result, 4);
        // sum over C_in (axis=2) -> [N, C_out, L_out]
        let output = self.sum(sum_k, 2);

        output
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

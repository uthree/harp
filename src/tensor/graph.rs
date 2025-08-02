use crate::ast::{AstNode, DType, Op as AstOp};
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Neg, Sub};

// Graph structure that owns all the nodes using interior mutability
#[derive(Default, Debug)]
pub struct Graph {
    pub nodes: RefCell<Vec<TensorData>>,
}

// A handle to a node in the graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

// A temporary view of a tensor in the graph
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub id: TensorId,
    pub graph: &'a Graph,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorData {
    pub op: TensorOp,
    pub src: Vec<TensorId>,
    pub dtype: DType,
    pub shape: Vec<AstNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TensorOp {
    Elementwise(AstOp),
    Reduce(AstOp, Vec<usize>),
    Contiguous,
    Leaf,
    MergedElementwise(AstNode),
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: RefCell::new(Vec::new()),
        }
    }

    fn add_node(
        &self,
        op: TensorOp,
        src: Vec<TensorId>,
        dtype: DType,
        shape: Vec<AstNode>,
    ) -> TensorId {
        let mut nodes = self.nodes.borrow_mut();
        let id = nodes.len();
        nodes.push(TensorData {
            op,
            src,
            dtype,
            shape,
        });
        TensorId(id)
    }

    pub fn new_leaf(&self, dtype: DType, shape: Vec<AstNode>) -> TensorView {
        let id = self.add_node(TensorOp::Leaf, vec![], dtype, shape);
        self.get_view(id)
    }

    pub fn get_view(&self, id: TensorId) -> TensorView {
        TensorView { id, graph: self }
    }

    pub fn add(&self, lhs: TensorId, rhs: TensorId) -> TensorId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            ) // TODO: Proper shape calculation
        };
        let ast_node = AstNode::capture(0, lhs_dtype) + AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Add),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    pub fn sub(&self, lhs: TensorId, rhs: TensorId) -> TensorId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            ) // TODO: Proper shape calculation
        };
        let ast_node = AstNode::capture(0, lhs_dtype) - AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Add),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    pub fn mul(&self, lhs: TensorId, rhs: TensorId) -> TensorId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            ) // TODO: Proper shape calculation
        };
        let ast_node = AstNode::capture(0, lhs_dtype) * AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Mul),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    pub fn div(&self, lhs: TensorId, rhs: TensorId) -> TensorId {
        let (lhs_dtype, rhs_dtype, shape) = {
            let nodes = self.nodes.borrow();
            let lhs_node = &nodes[lhs.0];
            let rhs_node = &nodes[rhs.0];
            (
                lhs_node.dtype.clone(),
                rhs_node.dtype.clone(),
                lhs_node.shape.clone(),
            ) // TODO: Proper shape calculation
        };
        let ast_node = AstNode::capture(0, lhs_dtype) / AstNode::capture(1, rhs_dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Mul),
            vec![lhs, rhs],
            ast_node.dtype,
            shape,
        )
    }

    pub fn neg(&self, src: TensorId) -> TensorId {
        let (dtype, shape) = {
            let nodes = self.nodes.borrow();
            let src_node = &nodes[src.0];
            (src_node.dtype.clone(), src_node.shape.clone())
        };
        let ast_node = -AstNode::capture(0, dtype);
        self.add_node(
            TensorOp::Elementwise(AstOp::Neg),
            vec![src],
            ast_node.dtype,
            shape,
        )
    }
}

impl<'a> TensorView<'a> {
    pub fn op(&self) -> TensorOp {
        self.graph.nodes.borrow()[self.id.0].op.clone()
    }
    pub fn src(&self) -> Vec<TensorId> {
        self.graph.nodes.borrow()[self.id.0].src.clone()
    }
    pub fn dtype(&self) -> DType {
        self.graph.nodes.borrow()[self.id.0].dtype.clone()
    }
    pub fn shape(&self) -> Vec<AstNode> {
        self.graph.nodes.borrow()[self.id.0].shape.clone()
    }
}

// --- Operator Overloads for TensorView ---

impl<'a> Add for TensorView<'a> {
    type Output = TensorView<'a>;
    fn add(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.add(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Sub for TensorView<'a> {
    type Output = TensorView<'a>;
    fn sub(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.sub(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Mul for TensorView<'a> {
    type Output = TensorView<'a>;
    fn mul(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.mul(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Div for TensorView<'a> {
    type Output = TensorView<'a>;
    fn div(self, rhs: Self) -> Self::Output {
        let new_id = self.graph.div(self.id, rhs.id);
        self.graph.get_view(new_id)
    }
}

impl<'a> Neg for TensorView<'a> {
    type Output = TensorView<'a>;
    fn neg(self) -> Self::Output {
        let new_id = self.graph.neg(self.id);
        self.graph.get_view(new_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, Op as AstOp};

    #[test]
    fn test_graph_creation_and_view() {
        let graph = Graph::new();
        let a = graph.new_leaf(DType::F32, vec![]);
        let b = graph.new_leaf(DType::F32, vec![]);
        assert_eq!(a.id.0, 0);
        assert_eq!(b.id.0, 1);
        assert_eq!(graph.nodes.borrow().len(), 2);
        assert_eq!(a.dtype(), DType::F32);
    }

    #[test]
    fn test_view_add() {
        let graph = Graph::new();
        let a = graph.new_leaf(DType::F32, vec![]);
        let b = graph.new_leaf(DType::F32, vec![]);
        let c = a + b;

        assert_eq!(c.id.0, 2);
        assert_eq!(c.op(), TensorOp::Elementwise(AstOp::Add));
        assert_eq!(c.src(), vec![a.id, b.id]);
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_view_neg() {
        let graph = Graph::new();
        let a = graph.new_leaf(DType::F32, vec![]);
        let b = -a;

        assert_eq!(b.id.0, 1);
        assert_eq!(b.op(), TensorOp::Elementwise(AstOp::Neg));
        assert_eq!(b.src(), vec![a.id]);
        assert_eq!(b.dtype(), DType::F32);
    }

    #[test]
    fn test_view_implicit_cast() {
        let graph = Graph::new();
        let a = graph.new_leaf(DType::I32, vec![]);
        let b = graph.new_leaf(DType::F32, vec![]);
        let c = a + b;

        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_complex_expression_with_views() {
        let graph = Graph::new();
        let a = graph.new_leaf(DType::F32, vec![]);
        let b = graph.new_leaf(DType::F32, vec![]);
        let c = graph.new_leaf(DType::F32, vec![]);
        // d = a * b + c
        let d = a * b + c;

        assert_eq!(d.op(), TensorOp::Elementwise(AstOp::Add));
        let mul_node = d.graph.get_view(d.src()[0]);
        assert_eq!(mul_node.op(), TensorOp::Elementwise(AstOp::Mul));
        assert_eq!(mul_node.src(), vec![a.id, b.id]);
        assert_eq!(d.src()[1], c.id);
    }
}

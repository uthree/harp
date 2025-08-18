pub mod pattern;
pub mod shape;

use crate::ast::{AstNode, AstOp, Const, DType};
use crate::graph::shape::Expr as ShapeExpr;
use crate::graph::shape::View;
use std::hash::{Hash, Hasher};
use std::ops::{
    Add, AddAssign, Deref, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::rc::Rc;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>, // Shapeを決定するための変数。
    pub inputs: Vec<TensorSignature>,                 // 入力の型
    pub outputs: Vec<TensorSignature>,                // 出力の型
}

impl GraphSignature {
    pub fn new() -> Self {
        Self::default()
    }
}

// Shapeを決定するのに使う変数（整数）のシグチャ。これを導入することにより、異なるサイズのテンソルであっても、同じカーネルや計算グラフを流用できる。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ShapeVariableSignature {
    pub name: String,         // 変数名
    pub condition: ShapeExpr, // その値が利用可能かどうか判定するための式
    pub default: isize,       // デフォルト値, ベンチマークや最適化のために使用する。
}

// 入出力テンソルの型を表現する構造体。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorSignature {
    pub dtype: DType, // データ型
    pub shape: Vec<ShapeExpr>, // 形状
                      // ちなみにViewに関しては、入出力の時点では常にContiguousであるとする。
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GraphOp {
    Input { dtype: DType },
    Full(Const),
    Contiguous,
    Elementwise(AstOp),
    Reduce(AstOp, usize),
    Permute(Vec<usize>),
    Cumulative(AstOp, usize),
    Capture(usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphNodeData {
    pub op: GraphOp,
    pub src: Vec<GraphNode>,
    pub dtype: DType,
    pub view: View,
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

// Manual implementation of Hash and Eq for GraphNode to use it in HashMap
impl Hash for GraphNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl PartialEq for GraphNode {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for GraphNode {}

impl GraphNode {
    pub fn full(value: Const, shape: Vec<ShapeExpr>) -> Self {
        let dtype = value.dtype();
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Full(value),
            src: vec![],
            dtype,
            view: View::new_contiguous(shape),
        }))
    }

    /// Returns the logical shape of the node.
    pub fn shape(&self) -> &[ShapeExpr] {
        self.view.shape()
    }

    pub fn permute(self, axes: Vec<usize>) -> Self {
        let new_view = self.view.clone().permute(axes.clone());
        let dtype = self.dtype.clone();
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Permute(axes),
            src: vec![self],
            dtype,
            view: new_view,
        }))
    }

    pub fn cumulative(self, op: AstOp, axis: usize) -> Self {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Cumulative(op, axis),
            src: vec![self],
            dtype,
            view,
        }))
    }

    pub fn reduce(self, op: AstOp, axis: usize) -> Self {
        let dtype = self.dtype.clone();
        let mut shape = self.shape().to_vec();
        shape.remove(axis);
        let view = View::new_contiguous(shape);
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Reduce(op, axis),
            src: vec![self],
            dtype,
            view,
        }))
    }

    pub fn capture(pos: usize) -> Self {
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Capture(pos),
            src: vec![],
            dtype: DType::Any,
            view: View::new_contiguous(Vec::<ShapeExpr>::new()),
        }))
    }

    /// Matches the node against a pattern.
    ///
    /// If the node matches the pattern, it returns a vector of captured nodes.
    /// Otherwise, it returns `None`.
    #[must_use]
    pub fn matches(&self, pattern: &GraphNode) -> Option<Vec<GraphNode>> {
        let mut captures = Vec::new();
        if !self.matches_inner(pattern, &mut captures) {
            return None;
        }

        // The `astpat!` macro generates capture indices starting from 0.
        // We can rely on this to correctly size the results vector.
        // Find the maximum capture index to determine the size of the vector.
        let num_captures = captures.iter().map(|(i, _)| i).max().map_or(0, |m| m + 1);
        let mut result = vec![GraphNode::capture(0); num_captures];
        for (i, node) in captures {
            if i < num_captures {
                result[i] = node;
            }
        }
        Some(result)
    }

    pub fn permute_logical_indices(&self, indices: &[AstNode]) -> Vec<AstNode> {
        if let GraphOp::Permute(axes) = &self.op {
            let mut permuted_indices = vec![AstNode::from(0isize); indices.len()];
            for (i, &axis) in axes.iter().enumerate() {
                permuted_indices[axis] = indices[i].clone();
            }
            permuted_indices
        } else {
            panic!("permute_logical_indices called on non-permute node");
        }
    }

    fn matches_inner(&self, pattern: &GraphNode, captures: &mut Vec<(usize, GraphNode)>) -> bool {
        if let GraphOp::Capture(pos) = &pattern.op {
            captures.push((*pos, self.clone()));
            return true;
        }

        if pattern.dtype != DType::Any && self.dtype != pattern.dtype {
            return false;
        }

        if self.op != pattern.op || self.src.len() != pattern.src.len() {
            return false;
        }

        for (arg, pattern_arg) in self.src.iter().zip(&pattern.src) {
            if !arg.matches_inner(pattern_arg, captures) {
                return false;
            }
        }

        true
    }
}

// ... (Operator implementations for GraphNode remain the same) ...
macro_rules! impl_graphnode_binary_op {
    ($trait:ident, $fname:ident, $op:ident) => {
        impl<T: Into<GraphNode>> $trait<T> for GraphNode {
            type Output = GraphNode;
            fn $fname(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                assert_eq!(
                    self.dtype, rhs.dtype,
                    "Mismatched dtypes in binary operation"
                );
                assert_eq!(
                    self.shape(),
                    rhs.shape(),
                    "Mismatched shapes in binary operation"
                );
                let dtype = self.dtype.clone();
                let view = self.view.clone();
                GraphNode(Rc::new(GraphNodeData {
                    op: GraphOp::Elementwise(AstOp::$op),
                    src: vec![self, rhs],
                    dtype,
                    view,
                }))
            }
        }
    };
}

impl_graphnode_binary_op!(Add, add, Add);
impl_graphnode_binary_op!(Sub, sub, Sub);
impl_graphnode_binary_op!(Mul, mul, Mul);
impl_graphnode_binary_op!(Div, div, Div);
impl_graphnode_binary_op!(Rem, rem, Rem);

macro_rules! impl_graphnode_binary_assign_op {
    ($trait:ident, $fname:ident, $op:tt) => {
        impl<T: Into<GraphNode>> $trait<T> for GraphNode {
            fn $fname(&mut self, rhs: T) {
                *self = self.clone() $op rhs.into();
            }
        }
    };
}

impl_graphnode_binary_assign_op!(AddAssign, add_assign, +);
impl_graphnode_binary_assign_op!(SubAssign, sub_assign, -);
impl_graphnode_binary_assign_op!(MulAssign, mul_assign, *);
impl_graphnode_binary_assign_op!(DivAssign, div_assign, /);
impl_graphnode_binary_assign_op!(RemAssign, rem_assign, %);

impl Neg for GraphNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Elementwise(AstOp::Neg),
            src: vec![self],
            dtype,
            view,
        }))
    }
}

#[derive(Debug, Clone, Default)]
pub struct Graph {
    pub signature: GraphSignature,
    pub inputs: Vec<GraphNode>,
    pub outputs: Vec<GraphNode>,
}

impl Hash for Graph {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.signature.hash(state);
        for node in &self.inputs {
            node.hash(state);
        }
        for node in &self.outputs {
            node.hash(state);
        }
    }
}

impl PartialEq for Graph {
    fn eq(&self, other: &Self) -> bool {
        self.signature == other.signature
            && self.inputs == other.inputs
            && self.outputs == other.outputs
    }
}
impl Eq for Graph {}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_input(&mut self, shape: Vec<ShapeExpr>, dtype: &DType) -> GraphNode {
        let tensor_signature = TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        };
        self.signature.inputs.push(tensor_signature);
        let input_node = GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Input {
                dtype: dtype.clone(),
            },
            src: vec![],
            dtype: dtype.clone(),
            view: View::new_contiguous(shape),
        }));
        self.inputs.push(input_node.clone());
        input_node
    }
}

// ... (Tests for Graph remain the same) ...
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::expr::Expr as ShapeExpr;
    use rstest::rstest;

    #[test]
    fn test_add_input() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(1), ShapeExpr::from(2)];
        let dtype = DType::F32;

        let input_node = graph.add_input(shape.clone(), &dtype);

        // Check if the input node is added to the graph's inputs
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0], input_node);

        // Check if the signature is updated correctly
        assert_eq!(graph.signature.inputs.len(), 1);
        assert_eq!(graph.signature.inputs[0].shape, shape);
        assert_eq!(graph.signature.inputs[0].dtype, dtype);

        // Check the properties of the returned GraphNode
        if let GraphOp::Input { dtype: node_dtype } = &input_node.op {
            assert_eq!(input_node.shape(), shape.as_slice());
            assert_eq!(*node_dtype, dtype);
        } else {
            panic!("Expected GraphOp::Input");
        }
    }

    #[test]
    fn test_node_shape() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let dtype = DType::Isize;

        let input_node = graph.add_input(shape.clone(), &dtype);

        // The shape of the input node should be the one it was created with.
        assert_eq!(input_node.shape(), shape);
    }

    #[rstest]
    #[case(AstOp::Add, |a, b| a + b)]
    #[case(AstOp::Sub, |a, b| a - b)]
    #[case(AstOp::Mul, |a, b| a * b)]
    #[case(AstOp::Div, |a, b| a / b)]
    #[case(AstOp::Rem, |a, b| a % b)]
    fn test_graph_node_binary_ops(
        #[case] op: AstOp,
        #[case] op_fn: fn(GraphNode, GraphNode) -> GraphNode,
    ) {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);

        let result_node = op_fn(a.clone(), b.clone());

        // Check the operation type
        assert_eq!(result_node.op, GraphOp::Elementwise(op));

        // Check the source nodes
        assert_eq!(result_node.src.len(), 2);
        assert_eq!(result_node.src[0], a);
        assert_eq!(result_node.src[1], b);

        // Check the dtype
        assert_eq!(result_node.dtype, dtype);

        // Check the shape
        assert_eq!(result_node.shape(), shape);
    }

    #[rstest]
    #[case(AstOp::Add, |mut a, b| { a += b; a })]
    #[case(AstOp::Sub, |mut a, b| { a -= b; a })]
    #[case(AstOp::Mul, |mut a, b| { a *= b; a })]
    #[case(AstOp::Div, |mut a, b| { a /= b; a })]
    #[case(AstOp::Rem, |mut a, b| { a %= b; a })]
    fn test_graph_node_binary_assign_ops(
        #[case] op: AstOp,
        #[case] op_fn: fn(GraphNode, GraphNode) -> GraphNode,
    ) {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a_orig = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);

        let a = a_orig.clone();
        let result_node = op_fn(a, b.clone());

        // Check the operation type
        assert_eq!(result_node.op, GraphOp::Elementwise(op));

        // Check the source nodes
        assert_eq!(result_node.src.len(), 2);
        assert_eq!(result_node.src[0], a_orig);
        assert_eq!(result_node.src[1], b);

        // Check the dtype
        assert_eq!(result_node.dtype, dtype);

        // Check the shape
        assert_eq!(result_node.shape(), shape);
    }

    #[test]
    fn test_graph_node_neg_op() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let result_node = -a.clone();

        // Check the operation type
        assert_eq!(result_node.op, GraphOp::Elementwise(AstOp::Neg));

        // Check the source nodes
        assert_eq!(result_node.src.len(), 1);
        assert_eq!(result_node.src[0], a);

        // Check the dtype
        assert_eq!(result_node.dtype, dtype);

        // Check the shape
        assert_eq!(result_node.shape(), shape);
    }

    #[test]
    fn test_cumulative_op() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let op = AstOp::Add;
        let axis = 0;
        let result_node = a.clone().cumulative(op.clone(), axis);

        // Check the operation type
        assert_eq!(result_node.op, GraphOp::Cumulative(op, axis));

        // Check the source nodes
        assert_eq!(result_node.src.len(), 1);
        assert_eq!(result_node.src[0], a);

        // Check the dtype
        assert_eq!(result_node.dtype, dtype);

        // Check the shape
        assert_eq!(result_node.shape(), shape);
    }

    #[test]
    fn test_reduce_op() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let op = AstOp::Add;
        let axis = 1;
        let result_node = a.clone().reduce(op.clone(), axis);

        // Check the operation type
        assert_eq!(result_node.op, GraphOp::Reduce(op, axis));

        // Check the source nodes
        assert_eq!(result_node.src.len(), 1);
        assert_eq!(result_node.src[0], a);

        // Check the dtype
        assert_eq!(result_node.dtype, dtype);

        // Check the shape
        let expected_shape = vec![ShapeExpr::from(10)];
        assert_eq!(result_node.shape(), expected_shape);
    }
}

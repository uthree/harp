use crate::ast::{ConstLiteral, DType};
use crate::graph::shape::view::View;
use crate::graph::{GraphNode, GraphOp};

impl GraphNode {
    pub fn constant(value: ConstLiteral) -> Self {
        let dtype = match &value {
            ConstLiteral::F32(_) => DType::F32,
            ConstLiteral::Usize(_) => DType::Usize,
            ConstLiteral::Isize(_) => DType::Isize,
        };

        // shape=[] (スカラー) のviewを作成
        let view = View::new_contiguous(Vec::<i32>::new());

        GraphNode::new(GraphOp::Const(value), dtype, view)
    }

    pub fn f32(value: f32) -> Self {
        Self::constant(ConstLiteral::F32(value))
    }

    pub fn usize(value: usize) -> Self {
        Self::constant(ConstLiteral::Usize(value))
    }

    pub fn isize(value: isize) -> Self {
        Self::constant(ConstLiteral::Isize(value))
    }
}

impl From<f32> for GraphNode {
    fn from(value: f32) -> Self {
        GraphNode::f32(value)
    }
}

impl From<usize> for GraphNode {
    fn from(value: usize) -> Self {
        GraphNode::usize(value)
    }
}

impl From<isize> for GraphNode {
    fn from(value: isize) -> Self {
        GraphNode::isize(value)
    }
}

impl From<ConstLiteral> for GraphNode {
    fn from(value: ConstLiteral) -> Self {
        GraphNode::constant(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_const_operations() {
        // Test f32 constant
        let f32_const = GraphNode::f32(3.14);
        assert_eq!(f32_const.dtype, DType::F32);
        assert_eq!(f32_const.view.shape().len(), 0); // scalar
        if let GraphOp::Const(ConstLiteral::F32(val)) = &f32_const.op {
            assert_eq!(*val, 3.14);
        } else {
            panic!("Expected Const op with F32 literal");
        }

        // Test usize constant
        let usize_const = GraphNode::usize(42);
        assert_eq!(usize_const.dtype, DType::Usize);
        assert_eq!(usize_const.view.shape().len(), 0); // scalar
        if let GraphOp::Const(ConstLiteral::Usize(val)) = &usize_const.op {
            assert_eq!(*val, 42);
        } else {
            panic!("Expected Const op with Usize literal");
        }

        // Test isize constant
        let isize_const = GraphNode::isize(-10);
        assert_eq!(isize_const.dtype, DType::Isize);
        assert_eq!(isize_const.view.shape().len(), 0); // scalar
        if let GraphOp::Const(ConstLiteral::Isize(val)) = &isize_const.op {
            assert_eq!(*val, -10);
        } else {
            panic!("Expected Const op with Isize literal");
        }

        // Test using constant method directly
        let direct_const = GraphNode::constant(ConstLiteral::F32(2.71));
        assert_eq!(direct_const.dtype, DType::F32);
        assert_eq!(direct_const.view.shape().len(), 0);
    }

    #[test]
    fn test_from_traits() {
        // Test From trait implementations
        let node1: GraphNode = 3.14f32.into();
        assert_eq!(node1.dtype, DType::F32);

        let node2: GraphNode = 42usize.into();
        assert_eq!(node2.dtype, DType::Usize);

        let node3: GraphNode = (-10isize).into();
        assert_eq!(node3.dtype, DType::Isize);

        let node4: GraphNode = ConstLiteral::F32(1.0).into();
        assert_eq!(node4.dtype, DType::F32);

        // Test that these can be used in arithmetic operations
        let input_node = GraphNode::f32(1.0);
        let constant_node: GraphNode = 2.0f32.into();
        let result = input_node + constant_node;
        assert_eq!(result.dtype, DType::F32);
    }
}
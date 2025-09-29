use crate::graph::shape::view::View;
use crate::graph::{GraphNode, GraphOp};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

pub trait ReduceOps {
    fn sum(self, axis: usize) -> Self;
    fn product(self, axis: usize) -> Self;
    fn max(self, axis: usize) -> Self;
}

impl ReduceOps for GraphNode {
    fn sum(self, axis: usize) -> Self {
        self.reduce(ReduceOp::Add, axis)
    }

    fn product(self, axis: usize) -> Self {
        self.reduce(ReduceOp::Mul, axis)
    }

    fn max(self, axis: usize) -> Self {
        self.reduce(ReduceOp::Max, axis)
    }
}

impl GraphNode {
    pub fn reduce(self, op: ReduceOp, axis: usize) -> Self {
        assert!(axis < self.view.ndim(), "axis out of bounds");

        // 指定された軸を縮約してviewを作成
        let result_view = self.create_reduced_view(axis);

        GraphNode::new(
            GraphOp::Reduce(op.clone(), axis, self.clone()),
            self.dtype.clone(),
            result_view,
        )
    }

    fn create_reduced_view(&self, axis: usize) -> View {
        let current_shape = self.view.shape().to_vec();
        let mut new_shape = current_shape;
        new_shape.remove(axis);
        View::new_contiguous(new_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::Graph;

    #[test]
    fn test_reduce_operations() {
        let mut graph = Graph::new();

        // Create an input node with shape [2, 3, 4]
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);

        // Test sum reduction along different axes
        let sum_axis0 = input_node.clone().sum(0);
        assert_eq!(sum_axis0.view.shape(), &[3.into(), 4.into()]);

        let sum_axis1 = input_node.clone().sum(1);
        assert_eq!(sum_axis1.view.shape(), &[2.into(), 4.into()]);

        let sum_axis2 = input_node.clone().sum(2);
        assert_eq!(sum_axis2.view.shape(), &[2.into(), 3.into()]);

        // Test product reduction
        let product_axis0 = input_node.clone().product(0);
        assert_eq!(product_axis0.view.shape(), &[3.into(), 4.into()]);

        // Test max reduction
        let max_axis1 = input_node.max(1);
        assert_eq!(max_axis1.view.shape(), &[2.into(), 4.into()]);
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn test_reduce_axis_out_of_bounds() {
        let mut graph = Graph::new();
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into()]);

        // This should panic because axis 2 doesn't exist
        input_node.sum(2);
    }
}

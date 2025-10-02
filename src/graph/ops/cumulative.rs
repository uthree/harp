use crate::graph::{GraphNode, GraphOp};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CumulativeOp {
    Add,
    Mul,
    Max,
}

pub trait CumulativeOps {
    fn cumsum(self, axis: usize) -> Self;
    fn cumprod(self, axis: usize) -> Self;
    fn cummax(self, axis: usize) -> Self;
}

impl CumulativeOps for GraphNode {
    fn cumsum(self, axis: usize) -> Self {
        self.cumulative(CumulativeOp::Add, axis)
    }

    fn cumprod(self, axis: usize) -> Self {
        self.cumulative(CumulativeOp::Mul, axis)
    }

    fn cummax(self, axis: usize) -> Self {
        self.cumulative(CumulativeOp::Max, axis)
    }
}

impl GraphNode {
    pub fn cumulative(self, op: CumulativeOp, axis: usize) -> Self {
        assert!(axis < self.view.ndim(), "axis out of bounds");

        // 累積演算は形状を変更しない
        let result_view = self.view.clone();

        GraphNode::new(
            GraphOp::Cumulative(op.clone(), axis, self.clone()),
            self.dtype.clone(),
            result_view,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::Graph;

    #[test]
    fn test_cumulative_operations() {
        let mut graph = Graph::new();

        // Create an input node with shape [2, 3, 4]
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into(), 4.into()]);

        // Test cumulative sum along different axes
        let cumsum_axis0 = input_node.clone().cumsum(0);
        assert_eq!(cumsum_axis0.view.shape(), &[2.into(), 3.into(), 4.into()]);

        let cumsum_axis1 = input_node.clone().cumsum(1);
        assert_eq!(cumsum_axis1.view.shape(), &[2.into(), 3.into(), 4.into()]);

        let cumsum_axis2 = input_node.clone().cumsum(2);
        assert_eq!(cumsum_axis2.view.shape(), &[2.into(), 3.into(), 4.into()]);

        // Test cumulative product
        let cumprod_axis0 = input_node.clone().cumprod(0);
        assert_eq!(cumprod_axis0.view.shape(), &[2.into(), 3.into(), 4.into()]);

        // Test cumulative max
        let cummax_axis1 = input_node.cummax(1);
        assert_eq!(cummax_axis1.view.shape(), &[2.into(), 3.into(), 4.into()]);
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn test_cumulative_axis_out_of_bounds() {
        let mut graph = Graph::new();
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into()]);

        // This should panic because axis 2 doesn't exist
        input_node.cumsum(2);
    }
}

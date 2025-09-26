use crate::graph::{GraphNode, GraphOp};
#[derive(Debug)]
pub enum ElementwiseOp {
    Add(GraphNode, GraphNode),
}


impl std::ops::Add for GraphNode {
    type Output = GraphNode;
    fn add(self, rhs: Self) -> Self::Output {
        // dtypeチェック
        assert_eq!(self.dtype, rhs.dtype, "dtypes must match for element-wise operations");

        // 結果のviewを決定
        let result_view = self.view.elementwise_result_view(&rhs.view);

        // 新しいGraphNodeを作成
        GraphNode::new(
            GraphOp::Elementwise(ElementwiseOp::Add(self.clone(), rhs.clone())),
            self.dtype.clone(),
            result_view,
        )
    }
}

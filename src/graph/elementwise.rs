use crate::graph::{GraphNode, GraphOp};
#[derive(Debug)]
pub enum ElementwiseOp {
    Add(GraphNode, GraphNode),
    Max(GraphNode, GraphNode),
    Mod(GraphNode, GraphNode),
}


macro_rules! impl_elementwise_op {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl std::ops::$trait for GraphNode {
            type Output = GraphNode;
            fn $method(self, rhs: Self) -> Self::Output {
                // dtypeチェック
                assert_eq!(self.dtype, rhs.dtype, "dtypes must match for element-wise operations");

                // 結果のviewを決定
                let result_view = self.view.elementwise_result_view(&rhs.view);

                // 新しいGraphNodeを作成
                GraphNode::new(
                    GraphOp::Elementwise(ElementwiseOp::$variant(self.clone(), rhs.clone())),
                    self.dtype.clone(),
                    result_view,
                )
            }
        }
    };
}

impl_elementwise_op!(Add, add, Add);
impl_elementwise_op!(Rem, rem, Mod);

impl GraphNode {
    pub fn max(self, rhs: Self) -> Self {
        // dtypeチェック
        assert_eq!(self.dtype, rhs.dtype, "dtypes must match for element-wise operations");

        // 結果のviewを決定
        let result_view = self.view.elementwise_result_view(&rhs.view);

        // 新しいGraphNodeを作成
        GraphNode::new(
            GraphOp::Elementwise(ElementwiseOp::Max(self.clone(), rhs.clone())),
            self.dtype.clone(),
            result_view,
        )
    }
}

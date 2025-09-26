use crate::graph::GraphNode;

impl std::ops::Sub for GraphNode {
    type Output = GraphNode;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl std::ops::Div for GraphNode {
    type Output = GraphNode;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}
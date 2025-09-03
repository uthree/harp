use crate::graph::GraphNode;

impl<T> std::ops::Add<T> for GraphNode
where
    T: Into<GraphNode>,
{
    type Output = GraphNode;
    fn add(self, rhs: T) -> Self::Output {
        todo!()
    }
}

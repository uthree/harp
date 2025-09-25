use crate::graph::GraphNode;
#[derive(Debug)]
pub enum ElementwiseOp {
    Add(GraphNode, GraphNode),
}

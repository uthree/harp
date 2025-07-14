//! Defines the Intermediate Representation (IR) for the computation graph.

use harp_graph::Graph;
use std::fmt;

/// Represents the different operations that can be in the computation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operator {
    /// Load data from a source (e.g., a tensor).
    Load { name: String },
    /// Element-wise addition.
    Add,
    /// Element-wise subtraction.
    Sub,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise division.
    Div,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Load { name } => write!(f, "Load({})", name),
            Operator::Add => write!(f, "Add"),
            Operator::Sub => write!(f, "Sub"),
            Operator::Mul => write!(f, "Mul"),
            Operator::Div => write!(f, "Div"),
        }
    }
}

/// A type alias for our computation graph.
///
/// - The node data `T` is an `Operator`.
/// - The edge data `E` is a `usize` representing the operand index.
pub type ComputationGraph = Graph<Operator, usize>;

#[cfg(test)]
mod tests {
    use super::*;
    use harp_graph::NodeId;

    #[test]
    fn test_build_computation_graph() {
        let mut graph = ComputationGraph::new();

        // Build a graph for: (a + b) * c
        let a = graph.add_node(Operator::Load {
            name: "a".to_string(),
        });
        let b = graph.add_node(Operator::Load {
            name: "b".to_string(),
        });
        let c = graph.add_node(Operator::Load {
            name: "c".to_string(),
        });

        let add_op = graph.add_node(Operator::Add);
        graph.add_edge(add_op, a, 0);
        graph.add_edge(add_op, b, 1);

        let mul_op = graph.add_node(Operator::Mul);
        graph.add_edge(mul_op, add_op, 0);
        graph.add_edge(mul_op, c, 1);

        assert_eq!(graph.len(), 5);
        let mul_node = graph.get(mul_op).unwrap();
        assert_eq!(mul_node.children.len(), 2);
    }

    #[test]
    fn test_computation_graph_to_dot() {
        let mut graph = ComputationGraph::new();
        let a = graph.add_node(Operator::Load {
            name: "a".to_string(),
        });
        let b = graph.add_node(Operator::Load {
            name: "b".to_string(),
        });
        let add_op = graph.add_node(Operator::Add);
        graph.add_edge(add_op, a, 0);
        graph.add_edge(add_op, b, 1);

        let dot = graph.to_dot();
        let expected = r#"digraph G {
  node [shape=box];
  n0 [label="Load(a)"];
  n1 [label="Load(b)"];
  n2 [label="Add"];
  n2 -> n0 [label="0"];
  n2 -> n1 [label="1"];
}
"#;
        assert_eq!(dot, expected);
    }
}

//! Defines the Intermediate Representation (IR) for the computation graph.

// Re-export the core graph types from the foundational crate.
pub use harp_graph::{Graph, NodeId};

use std::fmt;

// --- Shape and Dimension ---

/// Represents a single dimension, which can be fixed or symbolic.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dim {
    Fixed(usize),
    Symbolic(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Fixed(val) => write!(f, "{}", val),
            Dim::Symbolic(name) => write!(f, "{}", name),
        }
    }
}

/// Represents the shape of a tensor.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<Dim>,
}

impl Shape {
    pub fn new(dims: Vec<Dim>) -> Self {
        Self { dims }
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}


// --- Operators and Graph ---

/// Represents the different operations that can be in the computation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operator {
    /// Load data from a source with a specific shape.
    Load { name: String, shape: Shape },
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
            Operator::Load { name, shape } => write!(f, "Load({}, {})", name, shape),
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

        let shape = Shape::new(vec![Dim::Fixed(10), Dim::Symbolic("N".to_string())]);
        let a = graph.add_node(Operator::Load { name: "a".to_string(), shape: shape.clone() });
        let b = graph.add_node(Operator::Load { name: "b".to_string(), shape: shape.clone() });
        let c = graph.add_node(Operator::Load { name: "c".to_string(), shape: shape.clone() });

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
        let shape = Shape::new(vec![Dim::Fixed(10)]);
        let a = graph.add_node(Operator::Load { name: "a".to_string(), shape: shape.clone() });
        let b = graph.add_node(Operator::Load { name: "b".to_string(), shape: shape.clone() });
        let add_op = graph.add_node(Operator::Add);
        graph.add_edge(add_op, a, 0);
        graph.add_edge(add_op, b, 1);

        let dot = graph.to_dot();
        let expected = r#"digraph G {
  node [shape=box];
  n0 [label="Load(a, [10])"];
  n1 [label="Load(b, [10])"];
  n2 [label="Add"];
  n2 -> n0 [label="0"];
  n2 -> n1 [label="1"];
}
"#;
        assert_eq!(dot, expected);
    }
}
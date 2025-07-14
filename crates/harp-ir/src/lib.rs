//! Defines the Intermediate Representation (IR) for the computation graph.

pub mod shapetracker;

// Re-export the core graph types from the foundational crate.
pub use harp_graph::{Graph, NodeId};
pub use shapetracker::ShapeTracker;

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
    pub dims: Vec<Dim>,
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
/// Every operator knows its output shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operator {
    Load { name: String, shape: Shape },
    Add { shape: Shape },
    Sub { shape: Shape },
    Mul { shape: Shape },
    Div { shape: Shape },
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Load { name, shape } => write!(f, "Load({}, {})", name, shape),
            Operator::Add { shape } => write!(f, "Add({})", shape),
            Operator::Sub { shape } => write!(f, "Sub({})", shape),
            Operator::Mul { shape } => write!(f, "Mul({})", shape),
            Operator::Div { shape } => write!(f, "Div({})", shape),
        }
    }
}

/// A type alias for our computation graph.
pub type ComputationGraph = Graph<Operator, usize>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_computation_graph_to_dot() {
        let mut graph = ComputationGraph::new();
        let shape = Shape::new(vec![Dim::Fixed(10)]);
        let a = graph.add_node(Operator::Load { name: "a".to_string(), shape: shape.clone() });
        let b = graph.add_node(Operator::Load { name: "b".to_string(), shape: shape.clone() });
        let add_op = graph.add_node(Operator::Add { shape: shape.clone() });
        graph.add_edge(add_op, a, 0);
        graph.add_edge(add_op, b, 1);

        let dot = graph.to_dot();
        let expected = r#"digraph G {
  node [shape=box];
  n0 [label="Load(a, [10])"];
  n1 [label="Load(b, [10])"];
  n2 [label="Add([10])"];
  n2 -> n0 [label="0"];
  n2 -> n1 [label="1"];
}
"#;
        assert_eq!(dot, expected);
    }
}

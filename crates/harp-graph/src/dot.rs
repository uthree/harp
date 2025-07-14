//! Functions for rendering the graph to DOT format.

use crate::Graph;
use std::fmt::{Display, Write};

impl<T, E> Graph<T, E>
where
    T: Display,
    E: Display,
{
    /// Renders the graph to a string in DOT format.
    ///
    /// This string can be used with tools like Graphviz to visualize the graph.
    ///
    /// # Example
    ///
    /// ```
    /// // Assuming `dot -Tpng graph.dot -o graph.png`
    /// ```
    pub fn to_dot(&self) -> String {
        let mut dot = String::new();
        writeln!(dot, "digraph G {{").unwrap();
        writeln!(dot, "  node [shape=box];").unwrap();

        // Define nodes
        for (i, node) in self.nodes.iter().enumerate() {
            writeln!(dot, "  n{} [label=\"{}\"];", i, node.data).unwrap();
        }

        // Define edges
        for (parent_idx, node) in self.nodes.iter().enumerate() {
            for (edge_data, child_id) in &node.children {
                writeln!(
                    dot,
                    "  n{} -> n{} [label=\"{}\"];",
                    parent_idx,
                    child_id.0,
                    edge_data
                )
                .unwrap();
            }
        }

        writeln!(dot, "}}").unwrap();
        dot
    }
}

#[cfg(test)]
mod tests {
    use crate::Graph;

    #[test]
    fn test_to_dot() {
        // A(Sub) -> [ (0, B), (1, C) ]
        let mut graph: Graph<String, usize> = Graph::new();
        let a = graph.add_node("Sub".to_string());
        let b = graph.add_node("B".to_string());
        let c = graph.add_node("C".to_string());
        graph.add_edge(a, b, 0);
        graph.add_edge(a, c, 1);

        let dot_output = graph.to_dot();

        let expected = r#"digraph G {
  node [shape=box];
  n0 [label="Sub"];
  n1 [label="B"];
  n2 [label="C"];
  n0 -> n1 [label="0"];
  n0 -> n2 [label="1"];
}
"#;
        assert_eq!(dot_output, expected);
    }
}
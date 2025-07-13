//! Provides the `ToDot` trait for generating DOT graph visualizations.

/// A trait for types that can be converted into a DOT graph format string.
pub trait ToDot {
    /// Converts the object to a string in DOT format for visualization.
    ///
    /// The resulting string can be used with tools like Graphviz to generate
    /// a visual representation of the graph.
    fn to_dot(&self) -> String;
}

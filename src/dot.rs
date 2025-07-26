//! A trait for representing a graph in DOT format.
pub trait ToDot {
    /// Returns the graph representation in DOT format as a string.
    fn to_dot(&self) -> String;
}

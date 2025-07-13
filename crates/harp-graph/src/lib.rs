//! A generic, arena-based graph data structure.

/// A unique identifier for a node in the graph.
/// It's a wrapper around usize to provide type safety.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(usize);

impl From<usize> for NodeId {
    fn from(id: usize) -> Self {
        NodeId(id)
    }
}

/// Represents a node in the graph.
/// The `T` is the data payload of the node.
pub struct Node<T> {
    /// The data held by the node.
    pub data: T,
    /// Indices of the children nodes in the graph's node list.
    pub children: Vec<NodeId>,
}

/// Represents an entire graph of nodes.
/// It owns all the nodes in an arena-style vector.
pub struct Graph<T> {
    /// Arena of nodes.
    nodes: Vec<Node<T>>,
}

impl<T> Graph<T> {
    /// Creates a new, empty graph.
    pub fn new() -> Self {
        Graph { nodes: Vec::new() }
    }

    /// Adds a new node to the graph with the given data and returns its ID.
    pub fn add_node(&mut self, data: T) -> NodeId {
        let id = self.nodes.len();
        self.nodes.push(Node {
            data,
            children: Vec::new(),
        });
        NodeId(id)
    }

    /// Adds a directed edge from a parent node to a child node.
    ///
    /// # Panics
    /// Panics if the parent ID is out of bounds.
    pub fn add_edge(&mut self, parent: NodeId, child: NodeId) {
        self.nodes[parent.0].children.push(child);
    }

    /// Gets a reference to a node by its ID.
    pub fn get(&self, id: NodeId) -> Option<&Node<T>> {
        self.nodes.get(id.0)
    }

    /// Gets a mutable reference to a node by its ID.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut Node<T>> {
        self.nodes.get_mut(id.0)
    }

    /// Returns the total number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if the graph contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl<T> Default for Graph<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node() {
        let mut graph = Graph::new();
        let n1 = graph.add_node("A");
        let n2 = graph.add_node("B");
        assert_eq!(n1, NodeId(0));
        assert_eq!(n2, NodeId(1));
        assert_eq!(graph.len(), 2);
    }

    #[test]
    fn test_add_edge_and_get_node() {
        let mut graph = Graph::new();
        let n1 = graph.add_node("A");
        let n2 = graph.add_node("B");
        let n3 = graph.add_node("C");

        graph.add_edge(n1, n2);
        graph.add_edge(n1, n3);

        let node1 = graph.get(n1).unwrap();
        assert_eq!(node1.data, "A");
        assert_eq!(node1.children, vec![n2, n3]);

        let node2 = graph.get(n2).unwrap();
        assert_eq!(node2.data, "B");
        assert!(node2.children.is_empty());
    }
}
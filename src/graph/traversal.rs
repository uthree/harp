//! Graph traversal and manipulation utilities

use std::collections::{HashMap, HashSet, VecDeque};

use super::node::GraphNode;

/// Graph traversal order
pub enum TraversalOrder {
    /// Breadth-first (BFS)
    BreadthFirst,
    /// Depth-first post-order
    DepthFirstPost,
    /// Topological sort (dependency order)
    Topological,
}

/// Collect nodes from graph in specified order
pub fn collect_nodes(roots: &[GraphNode], order: TraversalOrder) -> Vec<GraphNode> {
    match order {
        TraversalOrder::BreadthFirst => bfs(roots),
        TraversalOrder::DepthFirstPost => dfs_post(roots),
        TraversalOrder::Topological => topological_sort(roots),
    }
}

/// Breadth-first search
fn bfs(roots: &[GraphNode]) -> Vec<GraphNode> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    for root in roots {
        if visited.insert(root.clone()) {
            queue.push_back(root.clone());
        }
    }

    while let Some(node) = queue.pop_front() {
        result.push(node.clone());
        for src in node.sources() {
            if visited.insert(src.clone()) {
                queue.push_back(src.clone());
            }
        }
    }

    result
}

/// Depth-first search (post-order)
fn dfs_post(roots: &[GraphNode]) -> Vec<GraphNode> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();

    fn dfs_visit(node: &GraphNode, visited: &mut HashSet<GraphNode>, result: &mut Vec<GraphNode>) {
        if !visited.insert(node.clone()) {
            return;
        }
        for src in node.sources() {
            dfs_visit(src, visited, result);
        }
        result.push(node.clone());
    }

    for root in roots {
        dfs_visit(root, &mut visited, &mut result);
    }

    result
}

/// Topological sort (inputs before outputs)
///
/// Returns nodes in order such that all dependencies come before dependents.
pub fn topological_sort(outputs: &[GraphNode]) -> Vec<GraphNode> {
    // DFS post-order naturally gives us inputs-first order
    // because we visit sources before adding the node itself
    dfs_post(outputs)
}

/// Collect input nodes (external buffer references)
pub fn collect_inputs(roots: &[GraphNode]) -> Vec<GraphNode> {
    collect_nodes(roots, TraversalOrder::Topological)
        .into_iter()
        .filter(|n| n.is_external())
        .collect()
}

/// Convert graph to string representation (for debugging)
pub fn graph_to_string(roots: &[GraphNode]) -> String {
    let nodes = collect_nodes(roots, TraversalOrder::Topological);
    let mut result = String::new();

    for (i, node) in nodes.iter().enumerate() {
        result.push_str(&format!(
            "[{}] {:?} shape={:?} dtype={:?} sources={}\n",
            i,
            node.name().unwrap_or("unnamed"),
            node.shape(),
            node.dtype(),
            node.sources().len()
        ));
    }

    result
}

/// Find common subexpressions (nodes referenced multiple times)
///
/// Returns nodes that are referenced by more than one parent node.
pub fn find_common_subexpressions(roots: &[GraphNode]) -> Vec<(GraphNode, usize)> {
    let mut ref_count: HashMap<GraphNode, usize> = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    // Count references from roots
    for root in roots {
        *ref_count.entry(root.clone()).or_insert(0) += 1;
        if visited.insert(root.clone()) {
            queue.push_back(root.clone());
        }
    }

    // BFS to count all references
    while let Some(node) = queue.pop_front() {
        for src in node.sources() {
            *ref_count.entry(src.clone()).or_insert(0) += 1;
            if visited.insert(src.clone()) {
                queue.push_back(src.clone());
            }
        }
    }

    ref_count.into_iter().filter(|(_, c)| *c > 1).collect()
}

/// Count total nodes in graph
pub fn count_nodes(roots: &[GraphNode]) -> usize {
    fn count_recursive(node: &GraphNode, visited: &mut HashSet<GraphNode>) {
        if !visited.insert(node.clone()) {
            return;
        }
        for src in node.sources() {
            count_recursive(src, visited);
        }
    }

    let mut visited = HashSet::new();

    for root in roots {
        count_recursive(root, &mut visited);
    }

    visited.len()
}

/// Check if graph contains cycles (should not happen with proper construction)
pub fn has_cycle(roots: &[GraphNode]) -> bool {
    fn check_cycle(
        node: &GraphNode,
        visited: &mut HashSet<GraphNode>,
        in_stack: &mut HashSet<GraphNode>,
    ) -> bool {
        if in_stack.contains(node) {
            return true; // Cycle detected
        }
        if visited.contains(node) {
            return false; // Already processed
        }

        visited.insert(node.clone());
        in_stack.insert(node.clone());

        for src in node.sources() {
            if check_cycle(src, visited, in_stack) {
                return true;
            }
        }

        in_stack.remove(node);
        false
    }

    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();

    for root in roots {
        if check_cycle(root, &mut visited, &mut in_stack) {
            return true;
        }
    }

    false
}

/// Graph transformation trait
pub trait GraphTransform {
    /// Transform a single node (return None to keep original)
    fn transform(&self, node: &GraphNode) -> Option<GraphNode>;

    /// Apply transformation to entire graph
    fn apply(&self, roots: &[GraphNode]) -> Vec<GraphNode> {
        roots
            .iter()
            .map(|r| self.transform_recursive(r, &mut HashMap::new()))
            .collect()
    }

    /// Recursively transform with memoization
    fn transform_recursive(
        &self,
        node: &GraphNode,
        cache: &mut HashMap<GraphNode, GraphNode>,
    ) -> GraphNode {
        // Check cache first
        if let Some(cached) = cache.get(node) {
            return cached.clone();
        }

        // Transform children first
        let new_sources: Vec<GraphNode> = node
            .sources()
            .iter()
            .map(|s| self.transform_recursive(s, cache))
            .collect();

        // Check if sources changed
        let sources_changed = new_sources.iter().zip(node.sources()).any(|(a, b)| a != b);

        // Create node with potentially new sources
        let node_with_new_sources = if sources_changed {
            node.with_new_sources(new_sources)
        } else {
            node.clone()
        };

        // Apply transformation
        let result = self
            .transform(&node_with_new_sources)
            .unwrap_or(node_with_new_sources);

        cache.insert(node.clone(), result.clone());
        result
    }
}

impl<F> GraphTransform for F
where
    F: Fn(&GraphNode) -> Option<GraphNode>,
{
    fn transform(&self, node: &GraphNode) -> Option<GraphNode> {
        self(node)
    }
}

/// Simple node replacement transform
pub struct NodeReplacer {
    replacements: HashMap<GraphNode, GraphNode>,
}

impl NodeReplacer {
    pub fn new() -> Self {
        Self {
            replacements: HashMap::new(),
        }
    }

    pub fn add_replacement(&mut self, from: GraphNode, to: GraphNode) {
        self.replacements.insert(from, to);
    }
}

impl Default for NodeReplacer {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphTransform for NodeReplacer {
    fn transform(&self, node: &GraphNode) -> Option<GraphNode> {
        self.replacements.get(node).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::builder::input;
    use crate::graph::shape::Expr;

    fn make_test_graph() -> (GraphNode, GraphNode, GraphNode) {
        let a = input(vec![Expr::Const(32)], DType::F32).with_name("a");
        let b = input(vec![Expr::Const(32)], DType::F32).with_name("b");
        let c = (&a + &b).with_name("c");
        (a, b, c)
    }

    #[test]
    fn test_topological_sort() {
        let (a, b, c) = make_test_graph();
        let sorted = topological_sort(&[c.clone()]);

        // a and b should come before c
        let a_idx = sorted.iter().position(|n| n == &a).unwrap();
        let b_idx = sorted.iter().position(|n| n == &b).unwrap();
        let c_idx = sorted.iter().position(|n| n == &c).unwrap();

        assert!(a_idx < c_idx);
        assert!(b_idx < c_idx);
    }

    #[test]
    fn test_collect_inputs() {
        let (a, b, c) = make_test_graph();
        let inputs = collect_inputs(&[c]);

        assert_eq!(inputs.len(), 2);
        assert!(inputs.contains(&a));
        assert!(inputs.contains(&b));
    }

    #[test]
    fn test_count_nodes() {
        let (_, _, c) = make_test_graph();
        let count = count_nodes(&[c]);

        assert_eq!(count, 3); // a, b, c
    }

    #[test]
    fn test_no_cycle() {
        let (_, _, c) = make_test_graph();
        assert!(!has_cycle(&[c]));
    }

    #[test]
    fn test_common_subexpressions() {
        let a = input(vec![Expr::Const(32)], DType::F32);
        // Use 'a' twice
        let b = &a + &a;

        let common = find_common_subexpressions(&[b]);
        // 'a' is referenced twice
        assert!(common.iter().any(|(n, count)| n == &a && *count == 2));
    }

    #[test]
    fn test_node_replacer() {
        let a = input(vec![Expr::Const(32)], DType::F32).with_name("a");
        let b = input(vec![Expr::Const(32)], DType::F32).with_name("b");
        let c = (&a + &b).with_name("c");

        let new_a = input(vec![Expr::Const(32)], DType::F32).with_name("new_a");

        let mut replacer = NodeReplacer::new();
        replacer.add_replacement(a.clone(), new_a.clone());

        let transformed = replacer.apply(&[c]);
        let new_c = &transformed[0];

        // The sources of new_c should include new_a instead of a
        let inputs = collect_inputs(&[new_c.clone()]);
        assert!(inputs.iter().any(|n| n.name() == Some("new_a")));
    }
}

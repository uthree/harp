//! Pattern-matching-based graph optimizer.

use crate::ast::{AstNode, AstOp};
use crate::tensor::graph::{Graph, NodeId, NodeView, TensorOp};
use log::{debug, info, trace};
use rustc_hash::FxHashMap;

// --- Pattern Definition ---

/// A pattern to match against a node in the graph.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// Matches any node.
    Wildcard,
    /// Captures a node, assigning it a unique capture ID.
    Capture(usize),
    /// Matches a specific operation with a list of patterns for its inputs.
    Op(TensorOpPattern, Vec<Pattern>),
}

/// A pattern for matching against a `TensorOp`.
/// This is a simplified version of `TensorOp` for pattern matching.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorOpPattern {
    Elementwise(crate::ast::AstOp),
    // Add other ops as needed for patterns
}

// --- Pattern Helper Functions ---

/// Creates a capture pattern.
pub fn p_cap(id: usize) -> Pattern {
    Pattern::Capture(id)
}

/// Creates an elementwise operation pattern.
pub fn p_op(op: AstOp, args: Vec<Pattern>) -> Pattern {
    Pattern::Op(TensorOpPattern::Elementwise(op), args)
}

// --- Rule and Rewriter ---

/// A rewrite rule, consisting of a pattern to search for and a replacement.
pub struct GraphRule {
    pub name: &'static str,
    pub pattern: Pattern,
    /// The replacement is a function that takes the graph and captures
    /// and returns the `NodeId` of the new, rewritten node.
    pub replacer: Box<
        dyn Fn(
            &GraphRewriter,
            &Graph,
            &Graph,
            &mut FxHashMap<NodeId, NodeId>,
            &FxHashMap<usize, NodeId>,
            NodeId, // The ID of the node that matched the root of the pattern
        ) -> NodeId,
    >,
}

/// Applies a set of rules to optimize a computation graph.
pub struct GraphRewriter {
    rules: Vec<GraphRule>,
}

impl GraphRewriter {
    /// Creates a new rewriter with a given set of rules.
    pub fn new(rules: Vec<GraphRule>) -> Self {
        Self { rules }
    }

    /// Optimizes a graph by applying the rewriter's rules.
    ///
    /// Returns a new, optimized graph.
    pub fn apply(&self, graph: &Graph) -> Graph {
        info!("Starting graph rewrite process...");
        let new_graph = Graph::new();
        let mut memo = FxHashMap::default(); // memoization table for rewritten nodes

        // Rewrite all output nodes
        for &output_id in graph.outputs.borrow().iter() {
            let new_output_id = self.rewrite_node(&new_graph, graph, output_id, &mut memo);
            new_graph.outputs.borrow_mut().push(new_output_id);
        }

        // Re-register inputs in the new graph, preserving the original order.
        let new_inputs: Vec<_> = graph
            .inputs
            .borrow()
            .iter()
            .filter_map(|input_id| memo.get(input_id).copied())
            .collect();
        *new_graph.inputs.borrow_mut() = new_inputs;

        info!(
            "Graph rewrite complete. Original size: {} nodes, New size: {} nodes.",
            graph.nodes.borrow().len(),
            new_graph.nodes.borrow().len()
        );
        new_graph
    }

    /// Recursively rewrites a node and its dependencies.
    pub fn rewrite_node(
        &self,
        new_graph: &Graph,
        old_graph: &Graph,
        node_id: NodeId,
        memo: &mut FxHashMap<NodeId, NodeId>,
    ) -> NodeId {
        // If we've already rewritten this node, return the memoized ID
        if let Some(&new_id) = memo.get(&node_id) {
            trace!("Memo hit for node {node_id:?}");
            return new_id;
        }

        let view = old_graph.get_view(node_id);
        trace!("Rewriting node {:?}: {:?}", view.id, view.op());

        // First, try to apply a rule to the current node
        for rule in &self.rules {
            let mut captures = FxHashMap::default();
            trace!("Attempting to apply rule '{}'", rule.name);
            if rule.pattern.match_node(view, &mut captures) {
                debug!(
                    "Rule '{}' matched node {:?}. Applying rewrite.",
                    rule.name, view.id
                );
                // If a rule matches, we use the replacer to generate the new node.
                // The replacer is responsible for recursively rewriting the captured nodes.
                let new_id = (rule.replacer)(self, new_graph, old_graph, memo, &captures, node_id);
                memo.insert(node_id, new_id);
                trace!(
                    "Node {:?} rewritten to {:?} by rule '{}'",
                    node_id, new_id, rule.name
                );
                return new_id;
            }
        }

        // If no rule matches, rewrite the node's inputs and then create a copy
        // of the current node in the new graph.
        trace!("No rule matched for node {:?}. Rewriting inputs.", view.id);
        let old_node = &old_graph.nodes.borrow()[node_id.0];
        let new_src = old_node
            .src
            .iter()
            .map(|&src_id| self.rewrite_node(new_graph, old_graph, src_id, memo))
            .collect();

        let new_id = new_graph.add_node(
            old_node.op.clone(),
            new_src,
            old_node.dtype.clone(),
            old_node.shape.clone(),
        );
        memo.insert(node_id, new_id);
        trace!("Node {node_id:?} copied to new node {new_id:?}");
        new_id
    }
}

impl Pattern {
    /// Attempts to match this pattern against a given `NodeView`.
    pub fn match_node(&self, view: NodeView, captures: &mut FxHashMap<usize, NodeId>) -> bool {
        trace!("Matching pattern {:?} against node {:?}", self, view.id);
        match self {
            Pattern::Wildcard => {
                trace!("Wildcard matched node {:?}", view.id);
                true
            }
            Pattern::Capture(id) => {
                trace!("Captured node {:?} as @{}", view.id, id);
                captures.insert(*id, view.id);
                true
            }
            Pattern::Op(op_pattern, arg_patterns) => {
                let node_op = view.op();
                let node_src = view.src();

                // Check if the operation and number of arguments match
                if !op_pattern.matches(&node_op) || arg_patterns.len() != node_src.len() {
                    trace!(
                        "Op pattern mismatch or wrong arg count for node {:?}",
                        view.id
                    );
                    return false;
                }

                // Recursively match arguments
                for (arg_pattern, arg_node_id) in arg_patterns.iter().zip(node_src) {
                    let arg_view = view.graph.get_view(arg_node_id);
                    if !arg_pattern.match_node(arg_view, captures) {
                        return false;
                    }
                }
                trace!("Op pattern fully matched node {:?}", view.id);
                true
            }
        }
    }
}

impl TensorOpPattern {
    /// Checks if this pattern matches a given `TensorOp`.
    pub fn matches(&self, op: &TensorOp) -> bool {
        match (self, op) {
            (TensorOpPattern::Elementwise(p_op), TensorOp::Elementwise(o_op)) => p_op == o_op,
            // Other op types can be added here
            _ => false,
        }
    }
}

// --- Rule Macro ---

#[macro_export]
macro_rules! graph_rule {
    ($name:expr, $pattern:expr => $replacer:expr) => {
        $crate::opt::graph::GraphRule {
            name: $name,
            pattern: $pattern,
            replacer: Box::new(
                |rewriter, new_graph, old_graph, memo, captures, top_node_id| {
                    // Boilerplate: recursively rewrite all captured nodes.
                    let mut rewritten_captures: Vec<_> = captures
                        .iter()
                        .map(|(cap_id, node_id)| {
                            (
                                *cap_id,
                                rewriter.rewrite_node(new_graph, old_graph, *node_id, memo),
                            )
                        })
                        .collect();
                    // Sort by capture ID to provide them to the user replacer in a consistent order.
                    rewritten_captures.sort_by_key(|&(id, _)| id);
                    let rewritten_ids: Vec<_> =
                        rewritten_captures.into_iter().map(|(_, id)| id).collect();

                    // Call the user-provided replacer logic.
                    let user_replacer = $replacer;
                    user_replacer(new_graph, old_graph, captures, top_node_id, rewritten_ids)
                },
            ),
        }
    };
}

/// Returns a list of graph optimization rules.
pub fn get_fusion_rules() -> Vec<GraphRule> {
    vec![graph_rule!("fuse_elementwise_mul_add",
        // Pattern: (a * b) + c
        p_op(AstOp::Add, vec![
            p_op(AstOp::Mul, vec![p_cap(0), p_cap(1)]),
            p_cap(2)
        ])
        =>
        // Replacer
        |new_graph: &Graph,
         old_graph: &Graph,
         captures: &FxHashMap<usize, NodeId>,
         top_node_id: NodeId,
         rewritten_ids: Vec<NodeId>| {
            // rewritten_ids contains the new node IDs for captures [0, 1, 2] in order.
            let new_a_id = rewritten_ids[0];
            let new_b_id = rewritten_ids[1];
            let new_c_id = rewritten_ids[2];

            // We still need the original captures to get metadata from the old graph.
            let a_id = captures[&0];
            let b_id = captures[&1];
            let c_id = captures[&2];

            // The new fused node will have inputs [a, b, c]
            let new_src = vec![new_a_id, new_b_id, new_c_id];

            // Build the AST for the fused operation.
            let a_ast = AstNode::capture(0, old_graph.get_view(a_id).dtype());
            let b_ast = AstNode::capture(1, old_graph.get_view(b_id).dtype());
            let c_ast = AstNode::capture(2, old_graph.get_view(c_id).dtype());
            let fused_ast = a_ast * b_ast + c_ast;

            // The shape and dtype of the fused op is the same as the original top-level node.
            let top_node_view = old_graph.get_view(top_node_id);

            new_graph.add_node(
                TensorOp::FusedElementwise(fused_ast),
                new_src,
                top_node_view.dtype(),
                top_node_view.shape(),
            )
        }
    )]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{AstOp, DType};
    use crate::tensor::graph::Graph;

    fn setup_logger() {
        // Initialize the logger for tests, ignoring errors if it's already set up
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_elementwise_fusion() {
        setup_logger();
        // 1. Create the original graph for: a * b + c
        let graph = Graph::new();
        let a = graph.input(DType::F32, vec![]);
        let b = graph.input(DType::F32, vec![]);
        let c = graph.input(DType::F32, vec![]);
        let d = (a * b) + c;
        d.as_output();

        assert_eq!(graph.nodes.borrow().len(), 5); // a, b, c, mul, add

        // 2. Create a rewriter and apply the rules
        let rules = get_fusion_rules();
        let rewriter = GraphRewriter::new(rules);
        let new_graph = rewriter.apply(&graph);

        // 3. Verify the new graph
        assert_eq!(new_graph.nodes.borrow().len(), 4); // a, b, c, fused_add_mul
        let output_node_id = new_graph.outputs.borrow()[0];
        let output_node = &new_graph.nodes.borrow()[output_node_id.0];

        // Check that the output node is a FusedElementwise op
        match &output_node.op {
            TensorOp::FusedElementwise(ast) => {
                // Check the structure of the internal AST
                // Expected: Add(Mul(Capture(0), Capture(1)), Capture(2))
                assert_eq!(ast.op, AstOp::Add);
                assert_eq!(ast.src.len(), 2);
                let mul_node = &ast.src[0];
                let cap2_node = &ast.src[1];
                assert_eq!(mul_node.op, AstOp::Mul);
                assert_eq!(cap2_node.op, AstOp::Capture(2, DType::F32));
            }
            _ => panic!("Expected FusedElementwise op, found {:?}", output_node.op),
        }

        // Check that the inputs to the fused node are correct.
        // The sources of the fused node should be the new inputs, in the same order.
        let inputs = new_graph.inputs.borrow();
        assert_eq!(inputs.len(), 3);
        assert_eq!(output_node.src, *inputs);
    }
}

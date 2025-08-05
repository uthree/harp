//! Pattern-matching-based graph optimizer.

use crate::ast::{AstNode, Op as AstOp};
use crate::tensor::graph::{Graph, NodeId, NodeView, TensorOp};
use log::{debug, info, trace};
use std::collections::HashMap;

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
    Elementwise(crate::ast::Op),
    // Add other ops as needed for patterns
}

/// A rewrite rule, consisting of a pattern to search for and a replacement.
pub struct Rule {
    pub name: &'static str,
    pub pattern: Pattern,
    /// The replacement is a function that takes the graph and captures
    /// and returns the `NodeId` of the new, rewritten node.
    pub replacer: Box<
        dyn Fn(
            &Rewriter,
            &Graph,
            &Graph,
            &mut HashMap<NodeId, NodeId>,
            &HashMap<usize, NodeId>,
            NodeId, // The ID of the node that matched the root of the pattern
        ) -> NodeId,
    >,
}

/// Applies a set of rules to optimize a computation graph.
pub struct Rewriter {
    rules: Vec<Rule>,
}

impl Rewriter {
    /// Creates a new rewriter with a given set of rules.
    pub fn new(rules: Vec<Rule>) -> Self {
        Self { rules }
    }

    /// Optimizes a graph by applying the rewriter's rules.
    ///
    /// Returns a new, optimized graph.
    pub fn rewrite(&self, graph: &Graph) -> Graph {
        info!("Starting graph rewrite process...");
        let new_graph = Graph::new();
        let mut memo = HashMap::new(); // memoization table for rewritten nodes

        // Rewrite all output nodes
        for &output_id in graph.outputs.borrow().iter() {
            let new_output_id = self.rewrite_node(&new_graph, graph, output_id, &mut memo);
            new_graph.outputs.borrow_mut().push(new_output_id);
        }

        // Re-register inputs in the new graph
        let mut new_inputs = Vec::new();
        for &input_id in graph.inputs.borrow().iter() {
            if let Some(&new_input_id) = memo.get(&input_id) {
                new_inputs.push(new_input_id);
            }
        }
        // Sort by id to maintain order
        new_inputs.sort_by_key(|a| a.0);
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
        memo: &mut HashMap<NodeId, NodeId>,
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
            let mut captures = HashMap::new();
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
    pub fn match_node(&self, view: NodeView, captures: &mut HashMap<usize, NodeId>) -> bool {
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

/// Returns a list of graph optimization rules.
pub fn get_fusion_rules() -> Vec<Rule> {
    vec![Rule {
        name: "fuse_elementwise_binary",
        pattern: Pattern::Op(
            TensorOpPattern::Elementwise(AstOp::Add), // This will be made generic later
            vec![
                Pattern::Op(
                    TensorOpPattern::Elementwise(AstOp::Mul), // This will be made generic later
                    vec![Pattern::Capture(0), Pattern::Capture(1)],
                ),
                Pattern::Capture(2),
            ],
        ),
        replacer: Box::new(
            |rewriter, new_graph, old_graph, memo, captures, top_node_id| {
                // This replacer handles a pattern like: Op1(Op2(a, b), c)
                // It fuses them into a single FusedElementwise node.

                let a_id = captures[&0];
                let b_id = captures[&1];
                let c_id = captures[&2];

                // Recursively rewrite the leaf inputs (a, b, c)
                let new_a_id = rewriter.rewrite_node(new_graph, old_graph, a_id, memo);
                let new_b_id = rewriter.rewrite_node(new_graph, old_graph, b_id, memo);
                let new_c_id = rewriter.rewrite_node(new_graph, old_graph, c_id, memo);

                // The new fused node will have inputs [a, b, c]
                let new_src = vec![new_a_id, new_b_id, new_c_id];

                // Build the AST for the fused operation using the overloaded operators,
                // which will handle type promotion correctly.
                let a_ast = AstNode::capture(0, old_graph.get_view(a_id).dtype());
                let b_ast = AstNode::capture(1, old_graph.get_view(b_id).dtype());
                let c_ast = AstNode::capture(2, old_graph.get_view(c_id).dtype());

                let fused_ast = a_ast * b_ast + c_ast;

                // The shape and dtype of the fused op is the same as the original top-level node
                let top_node_view = old_graph.get_view(top_node_id);

                new_graph.add_node(
                    TensorOp::FusedElementwise(fused_ast),
                    new_src,
                    top_node_view.dtype(),
                    top_node_view.shape(),
                )
            },
        ),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{DType, Op as AstOp};
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
        let rewriter = Rewriter::new(rules);
        let new_graph = rewriter.rewrite(&graph);

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

        // Check that the inputs to the fused node are correct
        let inputs = new_graph.inputs.borrow();
        assert_eq!(inputs.len(), 3);
        let a_new_id = inputs[0];
        let b_new_id = inputs[1];
        let c_new_id = inputs[2];
        assert_eq!(output_node.src, vec![a_new_id, b_new_id, c_new_id]);
    }
}

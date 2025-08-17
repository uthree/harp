use crate::graph::{Graph, GraphNode};
use std::collections::HashMap;
use std::rc::Rc;

type RewriterFn = Box<dyn Fn(&[GraphNode]) -> GraphNode>;
type ConditionFn = Box<dyn Fn(&[GraphNode]) -> bool>;

pub struct GraphRewriteRule {
    pattern: GraphNode,
    rewriter: RewriterFn,
    condition: ConditionFn,
}

impl GraphRewriteRule {
    pub fn new(
        pattern: GraphNode,
        rewriter: impl Fn(&[GraphNode]) -> GraphNode + 'static,
        condition: impl Fn(&[GraphNode]) -> bool + 'static,
    ) -> Rc<Self> {
        Rc::new(Self {
            pattern,
            rewriter: Box::new(rewriter),
            condition: Box::new(condition),
        })
    }
}

pub struct GraphRewriter {
    pub name: String,
    rules: Vec<Rc<GraphRewriteRule>>,
}

impl GraphRewriter {
    pub fn with_rules(name: &str, rules: Vec<Rc<GraphRewriteRule>>) -> Self {
        Self {
            name: name.to_string(),
            rules,
        }
    }

    pub fn rewrite(&self, graph: Graph) -> Graph {
        let mut new_graph = graph.clone();
        let mut memo = HashMap::new();

        new_graph.outputs = new_graph
            .outputs
            .iter()
            .map(|node| self.apply(node, &mut memo))
            .collect();

        new_graph
    }

    fn apply(&self, node: &GraphNode, memo: &mut HashMap<GraphNode, GraphNode>) -> GraphNode {
        if let Some(rewritten) = memo.get(node) {
            return rewritten.clone();
        }

        // Post-order traversal: rewrite children first
        let new_srcs: Vec<GraphNode> = node.src.iter().map(|src| self.apply(src, memo)).collect();

        let mut current_node = if new_srcs.iter().zip(&node.src).any(|(n, o)| n != o) {
            let new_data = crate::graph::GraphNodeData {
                op: node.op.clone(),
                src: new_srcs,
                dtype: node.dtype.clone(),
                view: node.view.clone(),
            };
            GraphNode(std::rc::Rc::new(new_data))
        } else {
            node.clone()
        };

        // Then, try to rewrite the current node
        loop {
            let mut applied_rule = false;
            for rule in &self.rules {
                if let Some(captures) = current_node.matches(&rule.pattern)
                    && (rule.condition)(&captures) {
                        current_node = (rule.rewriter)(&captures);
                        applied_rule = true;
                        break; // Restart with the new node
                    }
            }
            if !applied_rule {
                break;
            }
        }

        memo.insert(node.clone(), current_node.clone());
        current_node
    }
}

#[macro_export]
macro_rules! graphpat {
    (| $($capture: pat_param),* | $pattern: expr, if $condition: expr => $rewriter: expr) => {
        {
            let mut counter = 0..;
            $(
                let $capture = $crate::graph::GraphNode::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = move |captured_nodes: &[$crate::graph::GraphNode]| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $rewriter
            };
            let condition = move |captured_nodes: &[$crate::graph::GraphNode]| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $condition
            };
            $crate::graph::pattern::GraphRewriteRule::new(pattern, rewriter, condition)
        }
    };
    (| $($capture: pat_param),* | $pattern: expr => $rewriter: expr ) => {
        {
            let mut counter = 0..;
            $(
                let $capture = $crate::graph::GraphNode::capture(counter.next().unwrap());
            )*
            let pattern = $pattern;
            let rewriter = move |captured_nodes: &[$crate::graph::GraphNode]| {
                let mut counter = 0..;
                $(
                    let $capture = captured_nodes[counter.next().unwrap()].clone();
                )*
                $rewriter
            };
            $crate::graph::pattern::GraphRewriteRule::new(pattern, rewriter, |_| true)
        }
    };
}

#[macro_export]
macro_rules! graph_rewriter {
    ($name:expr, $($rule:expr),*) => {
        $crate::graph::pattern::GraphRewriter::with_rules($name, vec![$($rule),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::expr::Expr as ShapeExpr;
    use crate::{graph_rewriter, graphpat};

    #[test]
    fn test_graphpat_macro_simple() {
        // Rule: --a => a
        let rule = graphpat!(|a| -(-a) => a);

        let mut graph = Graph::new();
        let a_node = graph.add_input(vec![ShapeExpr::from(1)], &DType::F32);
        let pattern_match = -(-a_node.clone());

        let captures = pattern_match.matches(&rule.pattern).unwrap();
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0], a_node);

        let rewritten = (rule.rewriter)(&captures);
        assert_eq!(rewritten, a_node);
    }

    #[test]
    fn test_graphpat_macro_with_condition() {
        // Rule: a + b => b + a if a == some_node
        let mut graph = Graph::new();
        let a_node = graph.add_input(vec![ShapeExpr::from(1)], &DType::F32);
        let b_node = graph.add_input(vec![ShapeExpr::from(1)], &DType::F32);
        let c_node = graph.add_input(vec![ShapeExpr::from(1)], &DType::F32);

        let a_node_clone = a_node.clone();
        let rule = graphpat!(|a, _b| a + _b, if a == a_node_clone => _b + a);

        // Condition should be true
        let pattern_match_true = a_node.clone() + b_node.clone();
        let captures_true = pattern_match_true.matches(&rule.pattern).unwrap();
        assert!((rule.condition)(&captures_true));
        let rewritten_true = (rule.rewriter)(&captures_true);
        assert_eq!(
            rewritten_true.op,
            crate::graph::GraphOp::Elementwise(crate::ast::AstOp::Add)
        );
        assert_eq!(rewritten_true.src[0], b_node);
        assert_eq!(rewritten_true.src[1], a_node);

        // Condition should be false
        let pattern_match_false = c_node.clone() + b_node.clone();
        let captures_false = pattern_match_false.matches(&rule.pattern).unwrap();
        assert!(!(rule.condition)(&captures_false));
    }

    #[test]
    fn test_graph_rewriter_macro() {
        let rule1 = graphpat!(|a| -(-a) => a);
        let rule2 = graphpat!(|a, b| a + b => b + a);
        let rewriter = graph_rewriter!("TestRewriter", rule1, rule2);

        assert_eq!(rewriter.name, "TestRewriter");
        assert_eq!(rewriter.rules.len(), 2);
    }
}

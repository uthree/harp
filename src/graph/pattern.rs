use crate::graph::{Graph, GraphNode};
use std::collections::{HashMap, HashSet};

pub trait Suggester {
    fn suggest(&self, node: &GraphNode) -> Vec<GraphNode>;
}

pub struct GraphRewriteRule {
    pub name: String,
    lhs: GraphNode,
    rhs: GraphNode,
}

impl GraphRewriteRule {
    pub fn new(name: &str, lhs: GraphNode, rhs: GraphNode) -> Self {
        Self {
            name: name.to_string(),
            lhs,
            rhs,
        }
    }

    pub fn apply(&self, node: &GraphNode) -> Option<GraphNode> {
        if let Some(captures) = node.matches(&self.lhs) {
            Some(self.build_rhs(&captures))
        } else {
            None
        }
    }

    fn build_rhs(&self, captures: &[GraphNode]) -> GraphNode {
        self.substitute(&self.rhs, captures)
    }

    fn substitute(&self, pattern: &GraphNode, captures: &[GraphNode]) -> GraphNode {
        if let crate::graph::GraphOp::Capture(pos) = &pattern.op {
            return captures[*pos].clone();
        }

        let new_srcs = pattern
            .src
            .iter()
            .map(|src| self.substitute(src, captures))
            .collect();

        let mut new_node = pattern.clone();
        let new_data = GraphNode(std::rc::Rc::new(crate::graph::GraphNodeData {
            op: pattern.op.clone(),
            src: new_srcs,
            dtype: pattern.dtype.clone(),
            view: pattern.view.clone(),
        }));
        new_node.0 = new_data.0;
        new_node
    }
}

impl Suggester for GraphRewriteRule {
    fn suggest(&self, node: &GraphNode) -> Vec<GraphNode> {
        self.apply(node).into_iter().collect()
    }
}

pub struct GraphRewriter {
    suggesters: Vec<Box<dyn Suggester>>,
}

impl GraphRewriter {
    pub fn new() -> Self {
        Self {
            suggesters: Vec::new(),
        }
    }

    pub fn with_suggester(mut self, suggester: Box<dyn Suggester>) -> Self {
        self.add_suggester(suggester);
        self
    }

    pub fn add_suggester(&mut self, suggester: Box<dyn Suggester>) {
        self.suggesters.push(suggester);
    }

    pub fn rewrite(&self, graph: Graph) -> Graph {
        let mut new_graph = graph.clone();
        let mut memo = HashMap::new();
        let mut visited = HashSet::new();

        for output in &mut new_graph.outputs {
            *output = self.rewrite_node(output, &mut memo, &mut visited);
        }

        new_graph
    }

    fn rewrite_node(
        &self,
        node: &GraphNode,
        memo: &mut HashMap<GraphNode, GraphNode>,
        visited: &mut HashSet<GraphNode>,
    ) -> GraphNode {
        if let Some(rewritten) = memo.get(node) {
            return rewritten.clone();
        }

        if !visited.insert(node.clone()) {
            // Cycle detected, or already processing this node.
            // This can happen in complex graphs. Return original node.
            return node.clone();
        }

        let new_srcs: Vec<GraphNode> = node
            .src
            .iter()
            .map(|src| self.rewrite_node(src, memo, visited))
            .collect();

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

        // Recursively apply rewrites until no more rules match
        loop {
            let suggestions = self
                .suggesters
                .iter()
                .flat_map(|s| s.suggest(&current_node))
                .collect::<Vec<_>>();

            if suggestions.is_empty() {
                break;
            }

            // For simplicity, we take the first suggestion.
            // More sophisticated strategies could be used here.
            current_node = suggestions.into_iter().next().unwrap();
        }

        memo.insert(node.clone(), current_node.clone());
        visited.remove(node);

        current_node
    }
}

#[macro_export]
macro_rules! graphpat {
    // Base case for variables and captures
    ($v:ident) => {
        $crate::graph::GraphNode::capture(stringify!($v).parse::<usize>().unwrap())
    };
    // Binary operators
    ($lhs:tt $op:tt $rhs:tt) => {{
        let op = match stringify!($op) {
            "+" => $crate::ast::AstOp::Add,
            "-" => $crate::ast::AstOp::Sub,
            "*" => $crate::ast::AstOp::Mul,
            "/" => $crate::ast::AstOp::Div,
            "%" => $crate::ast::AstOp::Rem,
            _ => panic!("Unsupported binary operator in graphpat"),
        };
        let lhs_node = graphpat!($lhs);
        let rhs_node = graphpat!($rhs);
        // In a real scenario, we'd need to propagate dtype and view properly.
        // For pattern matching, we can often use DType::Any and an empty view.
        let dummy_view = $crate::graph::shape::view::View::new_contiguous(vec![]);
        GraphNode(std::rc::Rc::new($crate::graph::GraphNodeData {
            op: $crate::graph::GraphOp::Elementwise(op),
            src: vec![lhs_node, rhs_node],
            dtype: $crate::ast::DType::Any,
            view: dummy_view,
        }))
    }};
}

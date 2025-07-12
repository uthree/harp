use crate::dot::ToDot;
use crate::dtype::DType;
use crate::op::*;
use std::collections::HashMap;
use std::convert::Into;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::rc::Rc;

// --- Node & NodeData Structs ---
#[derive(Debug, Clone)]
pub(crate) struct NodeData {
    pub op: Box<dyn Operator>,
    pub src: Vec<Node>,
}

impl PartialEq for NodeData {
    fn eq(&self, other: &Self) -> bool {
        *self.op == *other.op && self.src == other.src
    }
}
impl Eq for NodeData {}

/// A node in the computation graph.
///
/// `Node` is an immutable, reference-counted handle to the underlying graph data.
/// Cloning a `Node` is cheap as it only increments a reference counter.
///
/// Nodes can be combined using standard arithmetic operators to build larger graphs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node(Rc<NodeData>);

impl Node {
    /// Creates a new node with the given operator and source nodes.
    pub fn new(op: impl Operator + 'static, src: Vec<Node>) -> Self {
        Self(Rc::new(NodeData {
            op: Box::new(op),
            src,
        }))
    }

    /// Returns a reference to the node's operator.
    pub fn op(&self) -> &dyn Operator {
        &*self.0.op
    }

    /// Returns a reference to the node's source (child) nodes.
    pub fn src(&self) -> &Vec<Node> {
        &self.0.src
    }

    fn build_dot_recursive(
        node: &Node,
        nodes: &mut Vec<String>,
        edges: &mut Vec<String>,
        visited: &mut HashMap<*const NodeData, String>,
        counter: &mut usize,
    ) {
        let node_ptr = Rc::as_ptr(&node.0);
        if visited.contains_key(&node_ptr) {
            return;
        }

        let node_id = format!("node{}", *counter);
        *counter += 1;
        visited.insert(node_ptr, node_id.clone());

        let label = if let Some(const_op) = node.op().as_any().downcast_ref::<Const>() {
            let full_type_name = const_op.0.type_name();
            let short_type_name = full_type_name.split("::").last().unwrap_or(full_type_name);
            format!("Const\n({:?})\n<{}>", const_op.0, short_type_name)
        } else {
            node.op().name().to_string()
        };
        let shape = if node.op().as_any().is::<Const>() {
            "ellipse"
        } else {
            "box"
        };
        nodes.push(format!("{node_id} [label=\"{label}\", shape=\"{shape}\"];"));

        for src_node in node.src() {
            Self::build_dot_recursive(src_node, nodes, edges, visited, counter);
            let src_id = visited.get(&Rc::as_ptr(&src_node.0)).unwrap();
            edges.push(format!("{src_id} -> {node_id};"));
        }
    }
}

impl ToDot for Node {
    fn to_dot(&self) -> String {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut visited = HashMap::new();
        let mut counter = 0;
        Node::build_dot_recursive(self, &mut nodes, &mut edges, &mut visited, &mut counter);

        let mut dot = String::from("digraph G {\n");
        dot.push_str("  rankdir=TB;\n\n");
        dot.push_str("  // Nodes\n");
        for node_def in nodes {
            dot.push_str(&format!("  {node_def}\n"));
        }
        dot.push_str("\n  // Edges\n");
        for edge_def in edges {
            dot.push_str(&format!("  {edge_def}\n"));
        }
        dot.push_str("}\n");
        dot
    }
}

impl From<Rc<NodeData>> for Node {
    fn from(arc_node: Rc<NodeData>) -> Self {
        Node(arc_node)
    }
}

// --- Operator Overloads for Node ---
impl<T: Into<Node>> Add<T> for Node {
    type Output = Node;
    fn add(self, rhs: T) -> Self::Output {
        Node(Rc::new(NodeData {
            op: Box::new(OpAdd),
            src: vec![self, rhs.into()],
        }))
    }
}

impl<T: Into<Node>> Mul<T> for Node {
    type Output = Node;
    fn mul(self, rhs: T) -> Self::Output {
        Node(Rc::new(NodeData {
            op: Box::new(OpMul),
            src: vec![self, rhs.into()],
        }))
    }
}

impl Neg for Node {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self * constant(-1.0f32)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Into<Node>> Sub<T> for Node {
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        self + rhs.into().neg()
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Into<Node>> Div<T> for Node {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        self * recip(rhs.into())
    }
}

impl<T: Into<Node>> AddAssign<T> for Node {
    fn add_assign(&mut self, rhs: T) {
        *self = self.clone() + rhs.into();
    }
}

impl<T: Into<Node>> SubAssign<T> for Node {
    fn sub_assign(&mut self, rhs: T) {
        *self = self.clone() - rhs.into();
    }
}

impl<T: Into<Node>> MulAssign<T> for Node {
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone() * rhs.into();
    }
}

impl<T: Into<Node>> DivAssign<T> for Node {
    fn div_assign(&mut self, rhs: T) {
        *self = self.clone() / rhs.into();
    }
}

impl<T: Into<Node>> Rem<T> for Node {
    type Output = Node;
    fn rem(self, rhs: T) -> Self::Output {
        Node(Rc::new(NodeData {
            op: Box::new(OpRem),
            src: vec![self, rhs.into()],
        }))
    }
}

impl<T: Into<Node>> RemAssign<T> for Node {
    fn rem_assign(&mut self, rhs: T) {
        *self = self.clone() % rhs.into();
    }
}

// --- From Implementations for Primitives ---
macro_rules! impl_from_primitive_for_node {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Node {
                fn from(value: $t) -> Self {
                    constant(value)
                }
            }
        )*
    };
}

impl_from_primitive_for_node!(f32, f64, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

// --- Helper Functions ---

pub fn constant<T: DType + 'static>(value: T) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Const(Box::new(value))),
        src: vec![],
    }))
}

pub fn recip(a: Node) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Recip),
        src: vec![a],
    }))
}

pub fn capture(name: &str) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Capture(name.to_string())),
        src: vec![],
    }))
}

pub fn variable(name: &str) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Variable(name.to_string())),
        src: vec![],
    }))
}

pub fn sin(a: Node) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Sin),
        src: vec![a],
    }))
}

pub fn exp2(a: Node) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Exp2),
        src: vec![a],
    }))
}

pub fn log2(a: Node) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Log2),
        src: vec![a],
    }))
}

pub fn sqrt(a: Node) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Sqrt),
        src: vec![a],
    }))
}

pub fn cos(a: Node) -> Node {
    sin(a + constant(std::f32::consts::PI / 2.0))
}

pub fn tan(a: Node) -> Node {
    sin(a.clone()) / cos(a)
}

pub fn ln(a: Node) -> Node {
    log2(a) * constant(std::f64::consts::LN_2)
}

pub fn exp(a: Node) -> Node {
    exp2(a * constant(std::f32::consts::LOG2_E))
}

pub fn pow(base: Node, exp: Node) -> Node {
    exp2(exp * log2(base))
}

pub fn cast<T: DType + Default + 'static>(a: Node) -> Node {
    Node(Rc::new(NodeData {
        op: Box::new(Cast(Box::new(T::default()))),
        src: vec![a],
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pattern::Rewriter;

    #[test]
    #[allow(deprecated)]
    fn test_double_recip_rewrite_with_node_pattern() {
        // 1. Define the graph to be rewritten: recip(recip(a))
        let a = constant(1.0f32);
        let graph = recip(recip(a.clone()));

        // 2. Define the rewrite rule using the macro
        let rule = crate::rewrite_rule!(let x = capture("x"); recip(recip(x.clone())) => x);

        // 3. Apply the rule
        let rewriter = Rewriter::new("double_recip", vec![rule]);
        let rewritten_graph = rewriter.rewrite(graph);

        // 4. Assert that the rewritten graph is `a`
        assert_eq!(rewritten_graph, a);
    }
}

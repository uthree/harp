use dyn_clone::DynClone;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

// --- DType System ---
pub trait DType: Debug + DynClone {}
dyn_clone::clone_trait_object!(DType);

// --- Marker Traits for DTypes ---
pub trait Float: DType {}
pub trait Integer: DType {}
pub trait UnsignedInteger: Integer {}
pub trait SignedInteger: Integer {}

// --- DType Implementations ---
macro_rules! impl_dtype {
    ($($t:ty),*) => { $( impl DType for $t {} )* };
}
macro_rules! impl_float {
    ($($t:ty),*) => { $( impl Float for $t {} )* };
}
macro_rules! impl_integer {
    ($($t:ty),*) => { $( impl Integer for $t {} )* };
}
macro_rules! impl_unsigned {
    ($($t:ty),*) => { $( impl UnsignedInteger for $t {} )* };
}
macro_rules! impl_signed {
    ($($t:ty),*) => { $( impl SignedInteger for $t {} )* };
}

// Implement for Floats
impl_dtype!(f32, f64);
impl_float!(f32, f64);

// Implement for Unsigned Integers
impl_dtype!(u8, u16, u32, u64, usize);
impl_integer!(u8, u16, u32, u64, usize);
impl_unsigned!(u8, u16, u32, u64, usize);

// Implement for Signed Integers
impl_dtype!(i8, i16, i32, i64, isize);
impl_integer!(i8, i16, i32, i64, isize);
impl_signed!(i8, i16, i32, i64, isize);

// --- Operator Trait ---
pub trait Operator: Debug + DynClone + Any {
    fn as_any(&self) -> &dyn Any;
    fn name(&self) -> &'static str;
}
dyn_clone::clone_trait_object!(Operator);

impl PartialEq for dyn Operator {
    fn eq(&self, other: &Self) -> bool {
        self.as_any().type_id() == other.as_any().type_id()
    }
}
impl Eq for dyn Operator {}

// --- Operator Structs ---
macro_rules! def_operators {
    ($($name:ident),*) => {
        $(
            #[derive(Debug, Clone, PartialEq, Eq)]
            pub struct $name;
        )*
    };
}

macro_rules! impl_operator {
    ($($name:ident),*) => {
        $(
            impl Operator for $name {
                fn as_any(&self) -> &dyn Any { self }
                fn name(&self) -> &'static str { stringify!($name) }
            }
        )*
    };
}

def_operators!(OpAdd, OpMul, Load, Store, Recip, Wildcard);
impl_operator!(OpAdd, OpMul, Load, Store, Recip, Wildcard);

#[derive(Debug, Clone)]
pub struct Const(pub Box<dyn DType>);
impl Operator for Const {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Const"
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capture(pub String);
impl Operator for Capture {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &'static str {
        "Capture"
    }
}

// --- Marker Traits for Operators ---
pub trait UnaryOp: Operator {}
pub trait BinaryOp: Operator {}
pub trait CommutativeOp: BinaryOp {}

// --- Implement Marker Traits ---
impl BinaryOp for OpAdd {}
impl CommutativeOp for OpAdd {}
impl BinaryOp for OpMul {}
impl CommutativeOp for OpMul {}

impl UnaryOp for Recip {}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node(Arc<NodeData>);

impl Node {
    pub fn op(&self) -> &Box<dyn Operator> {
        &self.0.op
    }

    pub fn src(&self) -> &Vec<Node> {
        &self.0.src
    }

    pub fn to_dot(&self) -> String {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut visited = HashMap::new();
        let mut counter = 0;
        Self::build_dot_recursive(self, &mut nodes, &mut edges, &mut visited, &mut counter);

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

    fn build_dot_recursive(
        node: &Node,
        nodes: &mut Vec<String>,
        edges: &mut Vec<String>,
        visited: &mut HashMap<*const NodeData, String>,
        counter: &mut usize,
    ) {
        let node_ptr = Arc::as_ptr(&node.0);
        if visited.contains_key(&node_ptr) {
            return;
        }

        let node_id = format!("node{}", *counter);
        *counter += 1;
        visited.insert(node_ptr, node_id.clone());

        let label = if let Some(const_op) = node.op().as_any().downcast_ref::<Const>() {
            format!("Const\n({:?})", const_op.0)
        } else {
            node.op().name().to_string()
        };
        let shape = if node.op().as_any().is::<Const>() {
            "ellipse"
        } else {
            "box"
        };
        nodes.push(format!(
            "{node_id} [label=\"{label}\", shape=\"{shape}\"];"
        ));

        for src_node in node.src() {
            Self::build_dot_recursive(src_node, nodes, edges, visited, counter);
            let src_id = visited.get(&Arc::as_ptr(&src_node.0)).unwrap();
            edges.push(format!("{src_id} -> {node_id};"));
        }
    }
}

impl From<Arc<NodeData>> for Node {
    fn from(arc_node: Arc<NodeData>) -> Self {
        Node(arc_node)
    }
}


// --- Helper Functions ---
pub fn add(a: Node, b: Node) -> Node {
    Node(Arc::new(NodeData {
        op: Box::new(OpAdd),
        src: vec![a, b],
    }))
}

pub fn mul(a: Node, b: Node) -> Node {
    Node(Arc::new(NodeData {
        op: Box::new(OpMul),
        src: vec![a, b],
    }))
}

pub fn constant<T: DType + 'static>(value: T) -> Node {
    Node(Arc::new(NodeData {
        op: Box::new(Const(Box::new(value))),
        src: vec![],
    }))
}

pub fn recip(a: Node) -> Node {
    Node(Arc::new(NodeData {
        op: Box::new(Recip),
        src: vec![a],
    }))
}

pub fn capture(name: &str) -> Node {
    Node(Arc::new(NodeData {
        op: Box::new(Capture(name.to_string())),
        src: vec![],
    }))
}

// --- Operator Overloads for Node ---
impl Add for Node {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        add(self, rhs)
    }
}

impl Mul for Node {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs)
    }
}

impl Neg for Node {
    type Output = Self;
    fn neg(self) -> Self::Output {
        mul(self, constant(-1.0f32))
    }
}

impl Sub for Node {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        add(self, rhs.neg())
    }
}

impl Div for Node {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        mul(self, recip(rhs))
    }
}

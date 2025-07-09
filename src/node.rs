use dyn_clone::DynClone;
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
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

// --- Node & NodeRef Structs ---
#[derive(Debug, Clone)]
pub struct Node {
    pub op: Box<dyn Operator>,
    pub src: Vec<NodeRef>,
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        *self.op == *other.op && self.src == other.src
    }
}
impl Eq for Node {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeRef(Arc<Node>);

impl Deref for NodeRef {
    type Target = Arc<Node>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<Arc<Node>> for NodeRef {
    fn from(arc_node: Arc<Node>) -> Self {
        NodeRef(arc_node)
    }
}

impl Node {
    pub fn to_dot(&self) -> String {
        // ... (implementation remains the same, but uses NodeRef)
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
        visited: &mut HashMap<*const Node, String>,
        counter: &mut usize,
    ) {
        let node_ptr = node as *const Node;
        if visited.contains_key(&node_ptr) {
            return;
        }

        let node_id = format!("node{}", *counter);
        *counter += 1;
        visited.insert(node_ptr, node_id.clone());

        let label = if let Some(const_op) = node.op.as_any().downcast_ref::<Const>() {
            format!("Const\n({:?})", const_op.0)
        } else {
            node.op.name().to_string()
        };
        let shape = if node.op.as_any().is::<Const>() {
            "ellipse"
        } else {
            "box"
        };
        nodes.push(format!(
            "{node_id} [label=\"{label}\", shape=\"{shape}\"];"
        ));

        for src_node in &node.src {
            Self::build_dot_recursive(src_node, nodes, edges, visited, counter);
            let src_id = visited.get(&(src_node.as_ref() as *const Node)).unwrap();
            edges.push(format!("{src_id} -> {node_id};"));
        }
    }
}

// --- Helper Functions ---
pub fn add(a: NodeRef, b: NodeRef) -> NodeRef {
    NodeRef(Arc::new(Node {
        op: Box::new(OpAdd),
        src: vec![a, b],
    }))
}

pub fn mul(a: NodeRef, b: NodeRef) -> NodeRef {
    NodeRef(Arc::new(Node {
        op: Box::new(OpMul),
        src: vec![a, b],
    }))
}

pub fn constant<T: DType + 'static>(value: T) -> NodeRef {
    NodeRef(Arc::new(Node {
        op: Box::new(Const(Box::new(value))),
        src: vec![],
    }))
}

pub fn recip(a: NodeRef) -> NodeRef {
    NodeRef(Arc::new(Node {
        op: Box::new(Recip),
        src: vec![a],
    }))
}

// --- Operator Overloads for NodeRef ---
impl Add for NodeRef {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        add(self, rhs)
    }
}

impl Mul for NodeRef {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        mul(self, rhs)
    }
}

impl Neg for NodeRef {
    type Output = Self;
    fn neg(self) -> Self::Output {
        mul(self, constant(-1.0f32))
    }
}

impl Sub for NodeRef {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        add(self, rhs.neg())
    }
}

impl Div for NodeRef {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        mul(self, recip(rhs))
    }
}

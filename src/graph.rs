use std::cell::RefCell;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Operator {
    // Input node
    Input,

    // unary
    Recip,
    Sin,
    Log2,
    Exp2,
    Sqrt,

    // binary
    Add,
    Mul,
    Mod,
    LessThan,

    // dimensional
    Sum(usize),
    CumSum(usize),
    Prod(usize),
    CumProd(usize),
    Max(usize),
    ArgMax(usize),

    // shape control
    Contiguous,
}

#[derive(Debug, Clone)]
pub struct Node {
    operator: Operator,
    inputs: Vec<Arc<RefCell<Self>>>,
    parent: Arc<RefCell<Graph>>,
}

#[derive(Debug, Clone)]
pub struct Graph {}

impl Graph {
    pub fn new() -> Arc<RefCell<Graph>> {
        let graph = Graph {};
        Arc::new(RefCell::new(graph))
    }
}

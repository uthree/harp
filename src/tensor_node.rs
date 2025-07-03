use crate::operator::Operator;
use crate::prelude::*;
use crate::shape::tracker::ShapeTracker;
use crate::unique_id;
use std::cell::RefCell;
use std::ops::Add;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float16,
    BFloat16,
    Float32,
    Float64,
}

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub struct TensorNodeStore {
    id: usize,
    shape_tracker: ShapeTracker,
    operator: Operator,
    inputs: Vec<TensorNode>,
    dtype: DataType,
    graph: Graph,
}

#[derive(Clone, PartialEq)]
pub struct TensorNode {
    store: Arc<RefCell<TensorNodeStore>>,
}

impl TensorNode {
    pub fn new(
        operator: Operator,
        inputs: Vec<TensorNode>,
        shape_tracker: ShapeTracker,
        dtype: DataType,
        graph: Graph,
    ) -> TensorNode {
        TensorNode {
            store: Arc::new(RefCell::new(TensorNodeStore {
                id: unique_id::next_id(),
                shape_tracker,
                operator,
                inputs,
                dtype,
                graph,
            })),
        }
    }
}

impl Add for TensorNode {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        let mut store = self.store.borrow_mut();
        let output_node = TensorNode::new(
            Operator::Add,
            vec![self.clone(), rhs.clone()],
            store.shape_tracker.clone(),
            store.dtype.clone(),
            store.graph.clone(),
        );
        store.graph.apply_node(output_node.clone());
        output_node
    }
}

impl std::ops::Mul for TensorNode {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        let mut store = self.store.borrow_mut();
        let output_node = TensorNode::new(
            Operator::Mul,
            vec![self.clone(), rhs.clone()],
            store.shape_tracker.clone(),
            store.dtype.clone(),
            store.graph.clone(),
        );
        store.graph.apply_node(output_node.clone());
        output_node
    }
}

impl std::ops::Div for TensorNode {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.recip()
    }
}

impl TensorNode {
    pub fn recip(self) -> Self {
        let mut store = self.store.borrow_mut();
        let output_node = TensorNode::new(
            Operator::Recip,
            vec![self.clone()],
            store.shape_tracker.clone(),
            store.dtype.clone(),
            store.graph.clone(),
        );
        store.graph.apply_node(output_node.clone());
        output_node
    }
}

impl std::fmt::Debug for TensorNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let store = self.store.borrow();
        write!(
            f,
            "Node {{ op={:?}, dtype={:?} id={}, with {} inputs }}",
            store.operator,
            store.dtype,
            store.id,
            store.inputs.len()
        )
    }
}

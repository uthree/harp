use crate::operator::Operator;
use crate::shape::tracker::ShapeTracker;
use std::cell::RefCell;
use std::sync::Weak;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug)]
pub struct TensorNodeStore {
    id: usize,
    shape_tracker: ShapeTracker,
    operator: Operator,
    inputs: Vec<TensorNode>,
    dtype: DataType,
}

impl TensorNodeStore {
    pub(crate) fn new(
        shape_tracker: ShapeTracker,
        operator: Operator,
        inputs: Vec<TensorNode>,
        dtype: DataType,
    ) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::SeqCst),
            shape_tracker,
            operator,
            inputs,
            dtype,
        }
    }
}

pub type TensorNode = Weak<RefCell<TensorNodeStore>>;

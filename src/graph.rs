use std::{cell::RefCell, sync::Arc};

#[derive(Debug, Clone)]
pub struct Graph_ {
    id_counter: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Graph {
    content: Arc<RefCell<Graph>>,
}

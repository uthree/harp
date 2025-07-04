use crate::prelude::*;
use std::{cell::RefCell, sync::Arc};

pub struct Graph_ {
    id_counter: usize,
    nodes: Vec<Tensor>,
}

pub struct Graph {
    content: Arc<RefCell<Graph_>>,
}

impl Graph {
    pub fn new() -> Self {
        let content = Graph_ {
            id_counter: 0,
            nodes: vec![],
        };
        let content = Arc::new(RefCell::new(content));
        Graph { content }
    }
}

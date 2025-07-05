use crate::prelude::*;
use crate::tensor::Tensor_;
use std::{cell::RefCell, sync::Arc};

pub struct Graph_ {
    input_nodes: Vec<Tensor>,
    nodes: Vec<Tensor>,
}

pub struct Graph {
    content: Arc<RefCell<Graph_>>,
}

impl Graph {
    pub fn new() -> Self {
        let content = Graph_ {
            input_nodes: vec![],
            nodes: vec![],
        };
        let content = Arc::new(RefCell::new(content));
        Graph { content }
    }
}

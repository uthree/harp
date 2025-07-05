use crate::prelude::*;
use std::{cell::RefCell, sync::Arc};

pub struct GraphData {
    input_nodes: Vec<Tensor>,
    nodes: Vec<Tensor>,
}

pub struct Graph {
    data: Arc<RefCell<GraphData>>,
}

impl Graph {
    pub fn new() -> Self {
        let content = GraphData {
            input_nodes: vec![],
            nodes: vec![],
        };
        let data = Arc::new(RefCell::new(content));
        Graph { data }
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Graph {
            data: self.data.clone(),
        }
    }
}

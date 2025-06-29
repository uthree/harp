use std::cell::RefCell;
use std::sync::Arc;

struct Graph {}

impl Graph {
    // initialize graph
    fn new() -> Arc<RefCell<Graph>> {
        let ctx = Graph {};
        let ptr = Arc::new(RefCell::new(ctx));
        ptr
    }
}

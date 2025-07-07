use crate::{operator::Operator, shape::tracker::ShapeTracker};
use std::fmt;

/// Represents a node in the computation graph.
///
/// Each `Node` encapsulates an operation (`Operator`) and the `ShapeTracker`
/// that describes the shape of the tensor produced by this node.
#[derive(Debug)]
pub struct Node {
    /// The operation associated with this node.
    op: Box<dyn Operator>,
    /// The shape tracker describing the output shape of this node.
    pub shape: ShapeTracker,
}

impl Node {
    /// Creates a new `Node` with the given operator and shape.
    ///
    /// # Arguments
    ///
    /// * `op` - An object implementing the `Operator` trait. It must have a `'static` lifetime.
    /// * `shape` - The `ShapeTracker` for the node's output.
    ///
    /// # Returns
    ///
    /// A new `Node` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::node::Node;
    /// use harp::operator::Input;
    /// use harp::shape::tracker::ShapeTracker;
    ///
    /// let shape = ShapeTracker::new(vec![1, 2, 3]);
    /// let node = Node::new(Input, shape.clone());
    ///
    /// assert_eq!(format!("{:?}", node.op()), "Input");
    /// assert_eq!(node.shape, shape);
    /// ```
    pub fn new(op: impl Operator + 'static, shape: ShapeTracker) -> Self {
        Self {
            op: Box::new(op),
            shape,
        }
    }

    /// Returns a reference to the operator associated with this node.
    ///
    /// # Returns
    ///
    /// A reference to `dyn Operator`.
    pub fn op(&self) -> &dyn Operator {
        &*self.op
    }
}

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}\n{:?}", self.op, self.shape)
    }
}

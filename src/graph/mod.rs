pub mod elementwise;
pub mod shape;

use crate::ast::{ConstLiteral, DType};
use crate::graph::shape::{view::View, Expr as ShapeExpr};
pub use elementwise::ElementwiseOp;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct GraphNodeData {
    op: GraphOp,
    dtype: DType,
    view: View,
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

#[derive(Debug)]
pub struct Graph {
    inputs: Vec<Weak<GraphNodeData>>,
    outputs: Vec<GraphNode>,
    shape_variables: Vec<ShapeVariableSignature>,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            inputs: vec![],
            outputs: vec![],
            shape_variables: vec![],
        }
    }

    // initialize input node
    pub fn input(&mut self, dtype: DType, shape: Vec<ShapeExpr>) -> GraphNode {
        let view = View::new_contiguous(shape);
        let node_data = GraphNodeData {
            op: GraphOp::Input,
            dtype,
            view,
        };
        let rc_node_data = Rc::new(node_data);
        let node = GraphNode(rc_node_data.clone());
        self.inputs.push(Rc::downgrade(&rc_node_data));
        node
    }

    // apply output node
    pub fn output(&mut self, node: GraphNode) {
        self.outputs.push(node);
    }

    pub fn shape_var(&mut self, var_name: &str, default: impl Into<isize>) -> ShapeExpr {
        self.shape_variables.push(ShapeVariableSignature {
            name: { var_name.to_string() },
            default: default.into(),
        });
        ShapeExpr::Var(var_name.to_string())
    }
}

#[derive(Debug)]
pub enum GraphOp {
    Input,
    Const(ConstLiteral), // initialize single element tensor, shape=[],
    Elementwise(ElementwiseOp),
}

#[derive(Debug)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>,
    pub inputs: Vec<TensorSignature>,
    pub outputs: Vec<TensorSignature>,
}

#[derive(Debug)]
pub struct ShapeVariableSignature {
    pub name: String,
    pub default: isize,
}

#[derive(Debug)]
pub struct TensorSignature {
    pub dtype: DType,
    pub shape: Vec<ShapeExpr>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_output() {
        let mut graph = Graph::new();

        // Create an input node
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into()]);

        // Add it as output
        graph.output(input_node);

        // Check that we have one input and one output
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.outputs.len(), 1);

        // Check that the input weak reference is still valid
        assert!(graph.inputs[0].upgrade().is_some());
    }

    #[test]
    fn test_multiple_inputs_outputs() {
        let mut graph = Graph::new();

        // Create multiple inputs
        let input1 = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let input2 = graph.input(DType::Usize, vec![4.into()]);

        // Add them as outputs
        graph.output(input1);
        graph.output(input2);

        assert_eq!(graph.inputs.len(), 2);
        assert_eq!(graph.outputs.len(), 2);

        // Check that both input weak references are still valid
        assert!(graph.inputs[0].upgrade().is_some());
        assert!(graph.inputs[1].upgrade().is_some());
    }

    #[test]
    fn test_input_weak_reference() {
        let mut graph = Graph::new();

        // Create an input node
        let input_node = graph.input(DType::F32, vec![2.into()]);

        // The weak reference should be valid while the node exists
        assert!(graph.inputs[0].upgrade().is_some());

        // Drop the node
        drop(input_node);

        // Now the weak reference should be invalid
        assert!(graph.inputs[0].upgrade().is_none());
    }
}

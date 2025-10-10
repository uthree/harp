use crate::graph::GraphNode;
use std::rc::Rc;

/// Unique identifier for tensors in the autograd graph
pub type TensorId = usize;

/// Trait for defining gradient functions.
/// Each operation implements this to specify how gradients flow backward.
pub trait GradFn: std::fmt::Debug {
    /// Compute gradients for inputs given the gradient of the output.
    ///
    /// # Arguments
    /// * `grad_output` - The gradient flowing back from the output
    /// * `inputs` - The input GraphNodes that were used in the forward pass
    ///
    /// # Returns
    /// A vector of gradients, one for each input. If an input doesn't need gradients,
    /// the corresponding entry can be None.
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>>;

    /// Get a human-readable name for this gradient function (for debugging)
    fn name(&self) -> &'static str;
}

/// Metadata for gradient tracking.
/// Stores information needed to compute gradients during backward pass.
#[derive(Clone)]
pub struct TensorMeta {
    /// The GraphNode representing this tensor in the computation graph
    pub graph_node: GraphNode,

    /// The gradient function that computes gradients for this operation
    pub grad_fn: Option<Rc<dyn GradFn>>,

    /// Input tensors that this tensor depends on
    /// Stored as (TensorId, GraphNode) pairs
    pub inputs: Vec<(TensorId, GraphNode)>,

    /// Whether this tensor requires gradient computation
    pub requires_grad: bool,
}

impl TensorMeta {
    /// Create a new TensorMeta for a leaf tensor (no gradient function)
    pub fn leaf(graph_node: GraphNode, requires_grad: bool) -> Self {
        TensorMeta {
            graph_node,
            grad_fn: None,
            inputs: Vec::new(),
            requires_grad,
        }
    }

    /// Create a new TensorMeta for a non-leaf tensor (has gradient function)
    pub fn non_leaf(
        graph_node: GraphNode,
        grad_fn: Rc<dyn GradFn>,
        inputs: Vec<(TensorId, GraphNode)>,
    ) -> Self {
        TensorMeta {
            graph_node,
            grad_fn: Some(grad_fn),
            inputs,
            requires_grad: true, // Non-leaf tensors always require grad
        }
    }
}

// Example gradient functions for basic operations

/// Gradient function for addition: grad flows equally to both inputs
#[derive(Debug)]
pub struct AddBackward;

impl GradFn for AddBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = x + y:
        // dL/dx = dL/dz * dz/dx = dL/dz * 1 = dL/dz
        // dL/dy = dL/dz * dz/dy = dL/dz * 1 = dL/dz
        vec![Some(grad_output.clone()), Some(grad_output)]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Gradient function for subtraction
#[derive(Debug)]
pub struct SubBackward;

impl GradFn for SubBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = x - y:
        // dL/dx = dL/dz * dz/dx = dL/dz * 1 = dL/dz
        // dL/dy = dL/dz * dz/dy = dL/dz * (-1) = -dL/dz
        vec![Some(grad_output.clone()), Some(-grad_output)]
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Gradient function for multiplication
#[derive(Debug)]
pub struct MulBackward;

impl GradFn for MulBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = x * y:
        // dL/dx = dL/dz * dz/dx = dL/dz * y
        // dL/dy = dL/dz * dz/dy = dL/dz * x
        assert_eq!(inputs.len(), 2, "MulBackward expects 2 inputs");
        let x = &inputs[0];
        let y = &inputs[1];

        vec![
            Some(grad_output.clone() * y.clone()),
            Some(grad_output * x.clone()),
        ]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Gradient function for division
#[derive(Debug)]
pub struct DivBackward;

impl GradFn for DivBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = x / y:
        // dL/dx = dL/dz * dz/dx = dL/dz * (1/y)
        // dL/dy = dL/dz * dz/dy = dL/dz * (-x/y^2)
        assert_eq!(inputs.len(), 2, "DivBackward expects 2 inputs");
        let x = &inputs[0];
        let y = &inputs[1];

        let grad_x = grad_output.clone() * y.clone().recip();
        let grad_y = (-grad_output) * x.clone() * (y.clone() * y.clone()).recip();

        vec![Some(grad_x), Some(grad_y)]
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

/// Gradient function for negation
#[derive(Debug)]
pub struct NegBackward;

impl GradFn for NegBackward {
    fn backward(&self, grad_output: GraphNode, _inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = -x:
        // dL/dx = dL/dz * dz/dx = dL/dz * (-1) = -dL/dz
        vec![Some(-grad_output)]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Gradient function for reciprocal
#[derive(Debug)]
pub struct RecipBackward;

impl GradFn for RecipBackward {
    fn backward(&self, grad_output: GraphNode, inputs: &[GraphNode]) -> Vec<Option<GraphNode>> {
        // For z = 1/x:
        // dL/dx = dL/dz * dz/dx = dL/dz * (-1/x^2)
        assert_eq!(inputs.len(), 1, "RecipBackward expects 1 input");
        let x = &inputs[0];

        let grad_x = (-grad_output) * (x.clone() * x.clone()).recip();

        vec![Some(grad_x)]
    }

    fn name(&self) -> &'static str {
        "RecipBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::Graph;

    #[test]
    fn test_add_backward() {
        let mut graph = Graph::new();
        let grad = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let x = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let y = graph.input(DType::F32, vec![2.into(), 3.into()]);

        let grad_fn = AddBackward;
        let grads = grad_fn.backward(grad.clone(), &[x, y]);

        assert_eq!(grads.len(), 2);
        assert!(grads[0].is_some());
        assert!(grads[1].is_some());
    }

    #[test]
    fn test_mul_backward() {
        let mut graph = Graph::new();
        let grad = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let x = graph.input(DType::F32, vec![2.into(), 3.into()]);
        let y = graph.input(DType::F32, vec![2.into(), 3.into()]);

        let grad_fn = MulBackward;
        let grads = grad_fn.backward(grad.clone(), &[x.clone(), y.clone()]);

        assert_eq!(grads.len(), 2);
        assert!(grads[0].is_some());
        assert!(grads[1].is_some());
    }
}

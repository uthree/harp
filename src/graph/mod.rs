pub mod ops;
pub mod shape;
use crate::ast::{AstNode, ConstLiteral, DType};
pub use crate::graph::ops::ReduceOps;
use crate::graph::ops::{CumulativeOp, ElementwiseOp, ReduceOp};
use crate::graph::shape::{view::View, Expr as ShapeExpr};
use std::fmt;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug)]
pub struct GraphNodeData {
    pub op: GraphOp,
    pub dtype: DType,
    pub view: View,
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

impl GraphNode {
    pub(crate) fn new(op: GraphOp, dtype: DType, view: View) -> GraphNode {
        GraphNode(Rc::new(GraphNodeData { op, dtype, view }))
    }

    pub(crate) fn from_rc(rc: Rc<GraphNodeData>) -> GraphNode {
        GraphNode(rc)
    }

    /// Get the strong reference count for this node
    /// Used to detect branching in graph optimization
    pub fn strong_count(&self) -> usize {
        Rc::strong_count(&self.0)
    }

    /// Cast this tensor to a different data type
    pub fn cast(self, target_dtype: DType) -> Self {
        // viewはそのまま継承
        let result_view = self.view.clone();

        GraphNode::new(
            GraphOp::Cast(self.clone(), target_dtype.clone()),
            target_dtype,
            result_view,
        )
    }

    /// Get the input nodes for this node
    pub fn input_nodes(&self) -> Vec<GraphNode> {
        match &self.op {
            GraphOp::Input(_) | GraphOp::Const(_) => vec![],
            GraphOp::View(input) | GraphOp::Contiguous(input) | GraphOp::Cast(input, _) => {
                vec![input.clone()]
            }
            GraphOp::Reduce(_, _, input)
            | GraphOp::Cumulative(_, _, input)
            | GraphOp::Fold(_, _, _, _, input) => {
                vec![input.clone()]
            }
            GraphOp::Elementwise(op) => {
                use crate::graph::ops::ElementwiseOp;
                match op {
                    ElementwiseOp::Add(a, b)
                    | ElementwiseOp::Mul(a, b)
                    | ElementwiseOp::Max(a, b)
                    | ElementwiseOp::Mod(a, b)
                    | ElementwiseOp::LessThan(a, b)
                    | ElementwiseOp::Eq(a, b) => vec![a.clone(), b.clone()],
                    ElementwiseOp::Neg(a)
                    | ElementwiseOp::Recip(a)
                    | ElementwiseOp::Sin(a)
                    | ElementwiseOp::Sqrt(a)
                    | ElementwiseOp::Log2(a)
                    | ElementwiseOp::Exp2(a) => vec![a.clone()],
                    ElementwiseOp::Select(cond, true_val, false_val) => {
                        vec![cond.clone(), true_val.clone(), false_val.clone()]
                    }
                }
            }
            GraphOp::FusedElementwise(_, inputs) => inputs.clone(),
            GraphOp::FusedReduce(_, _, input) => vec![input.clone()],
            GraphOp::FusedElementwiseReduce(_, inputs, _, _) => inputs.clone(),
            GraphOp::FusedElementwiseCumulative(_, inputs, _) => inputs.clone(),
        }
    }
}

// PartialEq, Eq, Hash are based on pointer address, not content
impl PartialEq for GraphNode {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for GraphNode {}

impl std::hash::Hash for GraphNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Rc::as_ptr(&self.0), state)
    }
}

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub inputs: Vec<Weak<GraphNodeData>>,
    pub outputs: Vec<GraphNode>,
    pub shape_variables: Vec<ShapeVariableSignature>,
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
        let input_index = self.inputs.len();
        let view = View::new_contiguous(shape);
        let node_data = GraphNodeData {
            op: GraphOp::Input(input_index),
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
    Input(usize),                                    // Input with index
    Const(ConstLiteral),                             // initialize single element tensor, shape=[],
    Elementwise(ElementwiseOp),                      // 要素ごとの演算
    Reduce(ReduceOp, usize, GraphNode),              // 軸を縮約する: (op, axis, input)
    Cumulative(ops::CumulativeOp, usize, GraphNode), // 累積演算: (op, axis, input)
    View(GraphNode),                                 // view変更操作
    Contiguous(GraphNode), // ContiguousなViewに並べ直す（入力のメモリレイアウトを連続に変換）
    Cast(GraphNode, DType), // 型変換: (input, target_dtype)
    Fold(usize, usize, usize, usize, GraphNode), // Fold operation (col2im): (dim, window_size, stride, dilation, input)
    // 融合済みの演算子
    FusedElementwise(AstNode, Vec<GraphNode>), // Capture(n)がn番目のGraphNodeに対応する
    FusedReduce(ReduceOp, Vec<usize>, GraphNode), // 複数の軸でReduceする
    FusedElementwiseReduce(AstNode, Vec<GraphNode>, ReduceOp, Vec<usize>), // FusedElementwiseの直後にFusedReduceする
    FusedElementwiseCumulative(AstNode, Vec<GraphNode>, CumulativeOp), // FusedElementwiseの直後にCumlativeする
}

impl fmt::Display for GraphOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphOp::Input(idx) => write!(f, "Input[{}]", idx),
            GraphOp::Const(_) => write!(f, "Const"),
            GraphOp::Elementwise(op) => write!(f, "{}", op),
            GraphOp::Reduce(op, axis, _) => write!(f, "{}[{}]", op, axis),
            GraphOp::Cumulative(op, axis, _) => write!(f, "{}[{}]", op, axis),
            GraphOp::View(_) => write!(f, "View"),
            GraphOp::Contiguous(_) => write!(f, "Contiguous"),
            GraphOp::Cast(_, dtype) => write!(f, "Cast({})", dtype),
            GraphOp::Fold(dim, window_size, stride, dilation, _) => {
                write!(
                    f,
                    "Fold[dim={}, win={}, stride={}, dilation={}]",
                    dim, window_size, stride, dilation
                )
            }
            GraphOp::FusedElementwise(_, _) => write!(f, "FusedElementwise"),
            GraphOp::FusedReduce(op, axes, _) => write!(f, "Fused{}[{:?}]", op, axes),
            GraphOp::FusedElementwiseReduce(_, _, op, axes) => {
                write!(f, "FusedER-{}{:?}", op, axes)
            }
            GraphOp::FusedElementwiseCumulative(_, _, op) => {
                write!(f, "FusedEC-{}", op)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>,
    pub inputs: Vec<ArraySignature>,
    pub outputs: Vec<ArraySignature>,
}

#[derive(Debug, Clone)]
pub struct ShapeVariableSignature {
    pub name: String,
    pub default: isize,
}

#[derive(Debug, Clone)]
pub struct ArraySignature {
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

    #[test]
    fn test_view_transformations() {
        let mut graph = Graph::new();

        // Create an input node with shape [2, 3]
        let input_node = graph.input(DType::F32, vec![2.into(), 3.into()]);

        // Test unsqueeze
        let unsqueezed = input_node.clone().unsqueeze(1);
        assert_eq!(unsqueezed.view.shape(), &[2.into(), 1.into(), 3.into()]);

        // Test squeeze
        let squeezed = unsqueezed.squeeze(1);
        assert_eq!(squeezed.view.shape(), &[2.into(), 3.into()]);

        // Test permute
        let permuted = input_node.clone().permute(vec![1, 0]);
        assert_eq!(permuted.view.shape(), &[3.into(), 2.into()]);

        // Test expand
        let expanded = input_node
            .clone()
            .unsqueeze(1)
            .expand(vec![2.into(), 5.into(), 3.into()]);
        assert_eq!(expanded.view.shape(), &[2.into(), 5.into(), 3.into()]);
    }
}

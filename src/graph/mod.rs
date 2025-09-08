use crate::ast::{ConstLiteral, DType};
pub mod ops;
pub mod shape;
use crate::graph::shape::{view::View, Expr as ShapeExpr};
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct GraphNode(pub(crate) Rc<GraphNodeData>);

#[derive(Debug, Clone)]
pub struct GraphNodeData {
    pub op: GraphOp,
    pub src: Vec<GraphNode>,
    pub dtype: DType,
    pub view: View,
}

#[derive(Debug, Clone)]
pub struct Graph {
    pub inputs: Vec<GraphNode>,
    pub outputs: Vec<GraphNode>,
    pub shape_variables: Vec<ShapeVariableSignature>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>,
    pub inputs: Vec<TensorSignature>,
    pub outputs: Vec<TensorSignature>,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    pub name: String,
    pub default: isize,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct TensorSignature {
    pub dtype: DType,
    pub shape: Vec<ShapeExpr>,
}

#[derive(Debug, Clone)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Max,
    Neg,
    Recip,
    Sqrt,
    Sin,
    Log2,
    Exp2,
    Rem,
    IntDiv,
}

#[derive(Debug, Clone)]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

#[derive(Debug, Clone)]
pub enum GraphOp {
    Input(Vec<ShapeExpr>),
    Const(ConstLiteral, Vec<ShapeExpr>),
    Elementwise(ElementwiseOp),
    Cast,
    Rand(Vec<ShapeExpr>),
    Arange(usize),
    Reduce(ReduceOp, usize),
    Cumulative(ReduceOp, usize),
    Contiguous,
    View,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Graph {
        Graph {
            inputs: vec![],
            outputs: vec![],
            shape_variables: vec![],
        }
    }

    // 自身のシグネチャを返す
    pub fn signature(&self) -> GraphSignature {
        let inputs = self
            .inputs
            .iter()
            .map(|node| TensorSignature {
                dtype: node.dtype.clone(),
                shape: node.shape().to_vec(),
            })
            .collect();
        let outputs = self
            .outputs
            .iter()
            .map(|node| TensorSignature {
                dtype: node.dtype.clone(),
                shape: node.shape().to_vec(),
            })
            .collect();
        GraphSignature {
            shape_variables: self.shape_variables.clone(),
            inputs,
            outputs,
        }
    }

    // 新たにshape variable (動的shape用の変数を作成する)
    pub fn shape_var(&mut self, name: &str, default: isize) -> ShapeExpr {
        let var = ShapeVariableSignature {
            name: name.to_string(),
            default,
        };
        self.shape_variables.push(var);
        ShapeExpr::Var(name.to_string())
    }

    // 新たに入力変数を作る
    pub fn input(&mut self, dtype: DType, shape: Vec<ShapeExpr>) -> GraphNode {
        let node = GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Input(shape.clone()),
            src: vec![],
            dtype,
            view: View::new_contiguous(shape),
        }));
        self.inputs.push(node.clone());
        node
    }

    // 出力としてノードを登録する
    pub fn output(&mut self, node: GraphNode) {
        self.outputs.push(node);
    }
}

impl GraphNode {
    pub fn shape(&self) -> &[ShapeExpr] {
        self.view.shape()
    }
}

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_operations() {
        let mut graph = Graph::new();

        // Test shape_var
        let n = graph.shape_var("N", 128);
        assert_eq!(
            graph.shape_variables,
            vec![ShapeVariableSignature {
                name: "N".to_string(),
                default: 128
            }]
        );
        assert_eq!(n, ShapeExpr::Var("N".to_string()));

        // Test input
        let x = graph.input(DType::F32, vec![n.clone()]);
        assert_eq!(graph.inputs.len(), 1);
        assert_eq!(graph.inputs[0].dtype, DType::F32);
        assert_eq!(graph.inputs[0].shape(), vec![n.clone()]);

        // Test output
        graph.output(x.clone());
        assert_eq!(graph.outputs.len(), 1);
        assert!(Rc::ptr_eq(&graph.outputs[0].0, &x.0));

        // Test signature
        let signature = graph.signature();
        let expected_signature = GraphSignature {
            shape_variables: vec![ShapeVariableSignature {
                name: "N".to_string(),
                default: 128,
            }],
            inputs: vec![TensorSignature {
                dtype: DType::F32,
                shape: vec![n.clone()],
            }],
            outputs: vec![TensorSignature {
                dtype: DType::F32,
                shape: vec![n],
            }],
        };
        assert_eq!(signature, expected_signature);
    }
}

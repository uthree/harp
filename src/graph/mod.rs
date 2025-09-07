use crate::ast::{ConstLiteral, DType};
pub mod ops;
pub mod shape;
use crate::graph::shape::Expr as ShapeExpr;
use std::ops::Deref;
use std::rc::Rc;

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

#[derive(Debug, Clone)]
pub struct GraphNodeData {
    pub op: GraphOp,
    pub src: Vec<GraphNode>,
    pub dtype: DType,
    pub shape: Vec<ShapeExpr>, // ここではあえてView構造体を持たずに、論理的なshapeだけを考慮する。具体的なメモリアクセスに関してはlowererが担当する。
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
}

#[derive(Debug, Clone)]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
}

#[derive(Debug, Clone)]
pub enum GraphOp {
    Input, // 入力
    Const(ConstLiteral, Vec<ShapeExpr>),
    Elementwise(ElementwiseOp),
    Cast,
    Rand(Vec<ShapeExpr>),
    Arange(usize),
    Reshape(Vec<ShapeExpr>),
    Reduce(ReduceOp, usize),
    Cumulative(ReduceOp, usize),
    Contiguous,
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
                shape: node.shape.clone(),
            })
            .collect();
        let outputs = self
            .outputs
            .iter()
            .map(|node| TensorSignature {
                dtype: node.dtype.clone(),
                shape: node.shape.clone(),
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
            default: default,
        };
        self.shape_variables.push(var);
        ShapeExpr::Var(name.to_string())
    }

    // 新たに入力変数を作る
    pub fn input(&mut self, dtype: DType, shape: Vec<ShapeExpr>) -> GraphNode {
        let node = GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Input,
            src: vec![],
            dtype,
            shape,
        }));
        self.inputs.push(node.clone());
        node
    }

    // 出力としてノードを登録する
    pub fn output(&mut self, node: GraphNode) {
        self.outputs.push(node);
    }
}

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

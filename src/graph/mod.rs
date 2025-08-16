pub mod shape;

use crate::ast::{AstOp, DType};
use crate::graph::shape::expr::Expr as ShapeExpr;
use std::ops::Deref;
use std::rc::{Rc, Weak};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>, // Shapeを決定するための変数。
    pub inputs: Vec<TensorSignature>,                 // 入力の型
    pub outputs: Vec<TensorSignature>,                // 出力の型
}

impl GraphSignature {
    pub fn new() -> Self {
        Self::default()
    }
}

// Shapeを決定するのに使う変数（整数）のシグチャ。これを導入することにより、異なるサイズのテンソルであっても、同じカーネルや計算グラフを流用できる。
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    pub name: String,         // 変数名
    pub condition: ShapeExpr, // その値が利用可能かどうか判定するための式
    pub default: isize,       // デフォルト値, ベンチマークや最適化のために使用する。
}

// 入出力テンソルの型を表現する構造体。
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSignature {
    pub dtype: DType, // データ型
    pub shape: Vec<ShapeExpr>, // 形状
                      // ちなみにViewに関しては、入出力の時点では常にContiguousであるとする。
}

#[derive(Debug, Clone, PartialEq)]
pub enum GraphOp {
    Input { shape: Vec<ShapeExpr>, dtype: DType },
    Contiguous,           // 要素をContiguous現在のViewでな配置に並べ直す。
    Elementwise(AstOp),   // apply element-wise operator
    Reduce(AstOp, usize), // reduce dimension
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNodeData {
    op: GraphOp,
    src: Vec<GraphNode>,
    dtype: DType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode(Rc<GraphNodeData>);

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph {
    pub signature: GraphSignature,
    pub inputs: Vec<GraphNode>,
    pub outputs: Vec<GraphNode>,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new input node to the graph.
    /// This method creates a new input node and adds it to the graph's input list.
    /// It also updates the graph's signature to reflect the new input.
    pub fn add_input(&mut self, shape: Vec<ShapeExpr>, dtype: &DType) -> GraphNode {
        // Create the tensor signature for the new input.
        let tensor_signature = TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        };

        // Update the graph's signature.
        self.signature.inputs.push(tensor_signature);

        // Create the new input graph node.
        let input_node = GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Input {
                shape,
                dtype: dtype.clone(),
            },
            src: vec![],
            dtype: dtype.clone(),
        }));

        // Add the new node to the graph's input nodes list.
        self.inputs.push(input_node.clone());

        // Return the created node.
        input_node
    }
}

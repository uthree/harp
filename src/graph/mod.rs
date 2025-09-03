use std::ops::Deref;
use std::rc::Rc;

use crate::ast::{AstNode, ConstLiteral, DType};
pub mod ops;
pub mod shape;
use crate::graph::shape::Expr as ShapeExpr;

#[derive(Default, Debug, Clone, PartialEq)]
pub struct GraphSignature {
    pub shape_variables: Vec<ShapeVariableSignature>, // Shapeを決定するための変数。
    pub inputs: Vec<TensorSignature>,                 // 入力の型
    pub outputs: Vec<TensorSignature>,                // 出力の型
}

// Shapeを決定するのに使う変数（整数）のシグチャ。これを導入することにより、異なるサイズのテンソルであっても、同じカーネルや計算グラフを流用できる。
#[derive(Default, Debug, Clone, PartialEq)]
pub struct ShapeVariableSignature {
    pub name: String,   // 変数名
    pub default: isize, // デフォルト値, ベンチマークや最適化のために使用する。
}

// 入出力テンソルの型を表現する構造体。
#[derive(Default, Debug, Clone, PartialEq)]
pub struct TensorSignature {
    pub dtype: DType, // データ型
    pub shape: Vec<ShapeExpr>,
    // ちなみにViewに関しては、入出力の時点では常にContiguousであるとする。
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
    Input,                               // 入力ノード
    Const(ConstLiteral, Vec<ShapeExpr>), // 定数ノード, 任意のshapeの定数
    Cast,                                // 型変換ノード
    Rand(Vec<ShapeExpr>),                // 任意のshapeの一様乱数を生成
    Arange(usize), // 任意の長さの１次元Tensorを生成、 内容は [0, 1, 2, ..., n],
    Reshape(Vec<ShapeExpr>),
    Elementwise(AstNode),         // Capture(n) が src[n]に対応する要素ごとの演算
    Reduce(ReduceOp, Vec<usize>), // 1つ以上の軸をReduce
    ElemenWiseReduce(AstNode, ReduceOp, Vec<usize>), //element-wiseな演算をしながらReduceする
    Cumulative(ReduceOp, usize),  // 累積演算
    ElementwiseCumulative(AstNode, ReduceOp, usize), // Element-wise演算をしつつ累積する
    Contiguous,                   // 連続なメモリ配置に並べ直す
}
#[derive(Debug, Clone)]
pub struct GraphNodeData {
    op: GraphOp,
    src: Vec<GraphNode>,
    dtype: DType,
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

#[derive(Default)]
pub struct Graph {
    pub signature: GraphSignature,
    pub inputs: Vec<GraphNode>,
    pub outputs: Vec<GraphNode>,
}

impl Graph {
    // Initialize graph
    pub fn new() -> Self {
        Self::default()
    }

    // new input node
    pub fn input(&mut self, shape: Vec<ShapeExpr>, dtype: DType) -> GraphNode {
        let input_node = GraphNode(Rc::new(GraphNodeData {
            op: GraphOp::Input,
            src: vec![],
            dtype,
        }));
        self.inputs.push(input_node.clone());
        input_node
    }
}

impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GraphNode {}

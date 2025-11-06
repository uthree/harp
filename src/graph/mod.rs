use crate::graph::{ops::GraphOp, shape::View};
use std::{
    collections::HashMap,
    ops::Deref,
    rc::{Rc, Weak},
};
pub mod ops;
pub mod shape;

/// 各軸の並列化戦略
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum AxisStrategy {
    #[default]
    Auto, // 最適化パスで自動決定
    Sequential {
        simd_width: Option<usize>,
    }, // 逐次実行（オプションでSIMD幅を指定）
    Thread {
        simd_width: Option<usize>,
    }, // スレッドで並列化（オプションでSIMD幅を指定）
    ThreadGroup {
        simd_width: Option<usize>,
    }, // スレッドグループ/ブロック（オプションでSIMD幅を指定）
}

#[derive(Debug)]
pub struct Graph {
    inputs: HashMap<String, Weak<GraphNodeData>>, // Rcの参照カウントに影響を与えないために、Weak参照で保持する。
    outputs: HashMap<String, GraphNode>,
}

#[derive(Debug, Clone)]
pub struct GraphNodeData {
    pub dtype: DType,
    pub op: GraphOp,
    pub src: Vec<GraphNode>, // 入力ノード
    pub view: View,
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

// AstNoderのDTypeとは異なり、VecやPtrは扱わない。
#[derive(Debug, Clone)]
pub enum DType {
    Unknown, // 未定または未知, プレースホルダー
    F32,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    // 初期化
    pub fn new() -> Self {
        Self {
            inputs: HashMap::new(),
            outputs: HashMap::new(),
        }
    }

    // 入力ノードを新規作成, builderパターンを使う
    pub fn input(&mut self, name: &str) -> InputNodeBuilder<'_> {
        InputNodeBuilder {
            graph: self,
            name: name.to_string(),
            dtype: None,
            shape: None,
        }
    }

    // 出力ノードを登録
    pub fn output(&mut self, name: &str, output_node: GraphNode) {
        self.outputs.insert(name.to_string(), output_node);
    }

    // 出力ノードのマップへのアクセス
    pub fn outputs(&self) -> &HashMap<String, GraphNode> {
        &self.outputs
    }

    // 入力ノードのマップへのアクセス
    pub fn inputs(&self) -> &HashMap<String, Weak<GraphNodeData>> {
        &self.inputs
    }
}

pub struct InputNodeBuilder<'a> {
    graph: &'a mut Graph,
    name: String,
    dtype: Option<DType>,
    shape: Option<Vec<shape::Expr>>,
}

impl<'a> InputNodeBuilder<'a> {
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = Some(dtype);
        self
    }

    // TIPS: 入力ノードの形状(View)は必ずContinguousである必要がある。
    pub fn with_shape<E: Into<shape::Expr> + Clone, I: IntoIterator<Item = E>>(
        mut self,
        shape: I,
    ) -> Self {
        self.shape = Some(shape.into_iter().map(|e| e.into()).collect());
        self
    }

    pub fn build(self) -> GraphNode {
        let dtype = self.dtype.unwrap_or(DType::Unknown);
        let view = if let Some(shape) = self.shape {
            View::contiguous(shape)
        } else {
            View::contiguous(Vec::<isize>::new())
        };

        let node = GraphNode::new(dtype, GraphOp::Input, vec![], view);
        self.graph.inputs.insert(self.name, Rc::downgrade(&node.0));
        node
    }
}

impl GraphNode {
    pub fn new(dtype: DType, op: GraphOp, src: Vec<GraphNode>, view: View) -> Self {
        Self(Rc::new(GraphNodeData {
            dtype,
            op,
            src,
            view,
        }))
    }

    /// ノードのポインタを取得（トポロジカルソートなどで識別に使用）
    pub fn as_ptr(&self) -> *const GraphNodeData {
        Rc::as_ptr(&self.0)
    }
}

// .0 のように書かなくても内部のデータを読み取れるようにする
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
    fn test_graph_new() {
        let graph = Graph::new();
        assert_eq!(graph.inputs.len(), 0);
        assert_eq!(graph.outputs.len(), 0);
    }

    #[test]
    fn test_input_node_creation() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // 入力ノードが作成されたことを確認
        assert_eq!(graph.inputs.len(), 1);
        assert!(graph.inputs.contains_key("x"));

        // ノードのプロパティを確認
        match input.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        match &input.op {
            GraphOp::Input => {}
            _ => panic!("Expected GraphOp::Input"),
        }

        assert_eq!(input.view.ndim(), 2);
        assert_eq!(input.view.shape().len(), 2);
        assert!(input.view.is_contiguous());
    }

    #[test]
    fn test_input_node_default_dtype() {
        let mut graph = Graph::new();
        let input = graph.input("x").with_shape(vec![5]).build();

        // デフォルトのDTypeはUnknown
        match input.dtype {
            DType::Unknown => {}
            _ => panic!("Expected DType::Unknown as default"),
        }
    }

    #[test]
    fn test_input_node_empty_shape() {
        let mut graph = Graph::new();
        let input = graph.input("scalar").with_dtype(DType::F32).build();

        // 空のshapeはスカラーを表す
        assert_eq!(input.view.ndim(), 0);
        assert!(input.view.is_contiguous());
    }

    #[test]
    fn test_output_node_registration() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        graph.output("y", input.clone());

        assert_eq!(graph.outputs.len(), 1);
        assert!(graph.outputs.contains_key("y"));
    }

    #[test]
    fn test_multiple_inputs() {
        let mut graph = Graph::new();
        let input1 = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let input2 = graph
            .input("y")
            .with_dtype(DType::F32)
            .with_shape(vec![20])
            .build();

        assert_eq!(graph.inputs.len(), 2);
        assert!(graph.inputs.contains_key("x"));
        assert!(graph.inputs.contains_key("y"));

        assert_eq!(input1.view.ndim(), 1);
        assert_eq!(input2.view.ndim(), 1);
    }

    #[test]
    fn test_graph_node_new() {
        let node = GraphNode::new(
            DType::F32,
            GraphOp::Input,
            vec![],
            View::contiguous(vec![3, 4]),
        );

        match node.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        assert_eq!(node.src.len(), 0);
        assert_eq!(node.view.ndim(), 2);
    }

    // 演算のテスト

    #[test]
    fn test_add_operation() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = a + b;

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Add,
                ..
            } => {}
            _ => panic!("Expected Add operation"),
        }

        assert_eq!(result.src.len(), 2);
        assert_eq!(result.view.ndim(), 1);
        assert_eq!(result.view.shape()[0], shape::Expr::from(10));
    }

    #[test]
    fn test_mul_operation() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 5])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 5])
            .build();

        let result = a * b;

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Mul,
                ..
            } => {}
            _ => panic!("Expected Mul operation"),
        }

        assert_eq!(result.src.len(), 2);
        assert_eq!(result.view.ndim(), 2);
    }

    #[test]
    fn test_neg_operation() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = -a;

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Neg,
                ..
            } => {}
            _ => panic!("Expected Neg operation"),
        }

        assert_eq!(result.src.len(), 1);
        assert_eq!(result.view.ndim(), 1);
    }

    #[test]
    fn test_sub_operation() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = a - b;

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        // a - b = a + (-b) なので、トップレベルはAdd
        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Add,
                ..
            } => {}
            _ => panic!("Expected Add operation at top level"),
        }

        assert_eq!(result.src.len(), 2);

        // 右側のオペランドはNeg演算であることを確認
        match &result.src[1].op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Neg,
                ..
            } => {}
            _ => panic!("Expected Neg operation for right operand"),
        }
    }

    #[test]
    fn test_rem_operation() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = a % b;

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Rem,
                ..
            } => {}
            _ => panic!("Expected Rem operation"),
        }

        assert_eq!(result.src.len(), 2);
    }

    #[test]
    fn test_recip_operation() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = ops::recip(a);

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Recip,
                ..
            } => {}
            _ => panic!("Expected Recip operation"),
        }

        assert_eq!(result.src.len(), 1);
    }

    #[test]
    fn test_max_operation() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = ops::max(a, b);

        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Max,
                ..
            } => {}
            _ => panic!("Expected Max operation"),
        }

        assert_eq!(result.src.len(), 2);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_shape_mismatch() {
        // 異なるshapeのノード同士の演算はpanicする
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![20])
            .build();

        // これはpanicするべき
        let _result = a + b;
    }

    #[test]
    fn test_complex_expression() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        let c = graph
            .input("c")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        // (a + b) * c
        let result = (a + b) * c;

        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Mul,
                ..
            } => {}
            _ => panic!("Expected Mul operation at top level"),
        }

        assert_eq!(result.src.len(), 2);

        // 左側のノードがAdd演算であることを確認
        match &result.src[0].op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Add,
                ..
            } => {}
            _ => panic!("Expected Add operation in left operand"),
        }
    }

    #[test]
    fn test_dtype_inference() {
        let mut graph = Graph::new();
        let unknown = graph.input("unknown").with_shape(vec![10]).build();
        let f32_node = graph
            .input("f32")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let result = unknown + f32_node;

        // UnknownとF32を組み合わせた場合、F32になるべき
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32 after inference"),
        }
    }
}

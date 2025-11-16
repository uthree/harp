use std::{
    collections::{BTreeMap, HashMap},
    ops::Deref,
    rc::{Rc, Weak},
};
pub mod hlops;
pub mod ops;
pub mod shape;
pub mod strategy;
pub mod visualization;

// Re-export commonly used types
pub use ops::{ElementwiseOp, GraphOp, ReduceOp};
pub use shape::{Expr, View};
pub use strategy::{CumulativeStrategy, ElementwiseStrategy, ReduceStrategy};

#[derive(Debug, Clone)]
pub struct Graph {
    inputs: HashMap<String, Weak<GraphNodeData>>, // Rcの参照カウントに影響を与えないために、Weak参照で保持する。
    outputs: BTreeMap<String, GraphNode>,         // BTreeMapでキー順にソートされた順序を保証
    shape_var_defaults: HashMap<String, isize>,   // 動的shape変数のデフォルト値（必須）
}

#[derive(Debug, Clone)]
pub struct GraphNodeData {
    pub dtype: DType,
    pub op: GraphOp,
    pub src: Vec<GraphNode>, // 入力ノード
    pub view: View,
    pub elementwise_strategies: Vec<ElementwiseStrategy>, // Element-wise演算の各軸の並列化・最適化戦略
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

// AstNoderのDTypeとは異なり、VecやPtrは扱わない。
#[derive(Debug, Clone, PartialEq)]
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
            outputs: BTreeMap::new(),
            shape_var_defaults: HashMap::new(),
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
    pub fn outputs(&self) -> &BTreeMap<String, GraphNode> {
        &self.outputs
    }

    // 入力ノードのマップへのアクセス
    pub fn inputs(&self) -> &HashMap<String, Weak<GraphNodeData>> {
        &self.inputs
    }

    // 入力ノードを登録（最適化時に使用）
    pub fn register_input(&mut self, name: String, input_node: GraphNode) {
        use std::rc::Rc;
        let weak_ref = Rc::downgrade(&input_node.0);
        self.inputs.insert(name, weak_ref);
    }

    // shape変数のデフォルト値を設定
    pub fn set_shape_var_default(&mut self, name: impl Into<String>, default_value: isize) {
        self.shape_var_defaults.insert(name.into(), default_value);
    }

    // shape変数のデフォルト値を取得
    pub fn shape_var_defaults(&self) -> &HashMap<String, isize> {
        &self.shape_var_defaults
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
        let ndim = view.ndim();
        // デフォルトは全軸Sequential（simd_width=1, unroll_factor=1）
        let elementwise_strategies = vec![ElementwiseStrategy::sequential(); ndim];
        Self(Rc::new(GraphNodeData {
            dtype,
            op,
            src,
            view,
            elementwise_strategies,
        }))
    }

    /// Rcから直接GraphNodeを作成（最適化時に使用）
    pub fn from_rc(rc: Rc<GraphNodeData>) -> Self {
        Self(rc)
    }

    pub fn with_elementwise_strategies(
        dtype: DType,
        op: GraphOp,
        src: Vec<GraphNode>,
        view: View,
        elementwise_strategies: Vec<ElementwiseStrategy>,
    ) -> Self {
        assert_eq!(
            view.ndim(),
            elementwise_strategies.len(),
            "elementwise_strategies length must match view ndim"
        );
        Self(Rc::new(GraphNodeData {
            dtype,
            op,
            src,
            view,
            elementwise_strategies,
        }))
    }

    /// ノードのポインタを取得（トポロジカルソートなどで識別に使用）
    pub fn as_ptr(&self) -> *const GraphNodeData {
        Rc::as_ptr(&self.0)
    }

    /// スカラー定数ノードを作成
    ///
    /// # 例
    /// ```
    /// use harp::prelude::*;
    ///
    /// // F32のスカラー定数
    /// let const_node = GraphNode::constant(2.5f32);
    /// ```
    pub fn constant<L: Into<crate::ast::Literal>>(value: L) -> Self {
        let literal = value.into();
        let dtype = match &literal {
            crate::ast::Literal::F32(_) => DType::F32,
            crate::ast::Literal::Int(_) => DType::Unknown, // Intは将来的に追加
        };
        // 定数はスカラー（shape=[]）
        let view = View::contiguous(Vec::<isize>::new());
        Self::new(dtype, GraphOp::Const(literal), vec![], view)
    }

    /// 指定軸を縮約（汎用）
    pub fn reduce(&self, op: ops::ReduceOp, axis: usize) -> Self {
        ops::reduce(self.clone(), op, axis)
    }

    /// 指定軸の合計
    pub fn reduce_sum(&self, axis: usize) -> Self {
        ops::reduce_sum(self.clone(), axis)
    }

    /// 指定軸の積
    pub fn reduce_mul(&self, axis: usize) -> Self {
        ops::reduce_mul(self.clone(), axis)
    }

    /// 指定軸の最大値
    pub fn reduce_max(&self, axis: usize) -> Self {
        ops::reduce_max(self.clone(), axis)
    }

    /// 逆数を計算（1/x）
    pub fn recip(self) -> Self {
        let view = self.view.clone();
        let dtype = self.dtype.clone();
        Self::new(
            dtype,
            ops::GraphOp::Elementwise {
                op: ops::ElementwiseOp::Recip,
                elementwise_strategies: None,
            },
            vec![self],
            view,
        )
    }

    /// 要素ごとの最大値
    pub fn max(self, other: Self) -> Self {
        let dtype = ops::infer_dtype(&self.dtype, &other.dtype);
        let view = ops::infer_view(&self.view, &other.view);
        Self::new(
            dtype,
            ops::GraphOp::Elementwise {
                op: ops::ElementwiseOp::Max,
                elementwise_strategies: None,
            },
            vec![self, other],
            view,
        )
    }

    /// Viewを変更した新しいノードを作成
    ///
    /// このメソッドは、既存のノードに対してView操作（permute, unsqueeze, expand等）を
    /// 適用した新しいノードを作成します。
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 4])
    ///     .build();
    ///
    /// // Viewを変更（転置）
    /// let transposed_view = a.view.clone().permute(vec![1, 0]);
    /// let a_transposed = a.view(transposed_view);
    /// ```
    pub fn view(&self, new_view: View) -> Self {
        Self::new(
            self.dtype.clone(),
            GraphOp::View(new_view.clone()),
            vec![self.clone()],
            new_view,
        )
    }

    /// テンソルの形状を変更（reshape）
    ///
    /// 要素数が同じで、現在のViewが連続している場合のみ使用可能です。
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    /// use harp::graph::shape::Expr;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a")
    ///     .with_dtype(DType::F32)
    ///     .with_shape(vec![3, 4])
    ///     .build();
    ///
    /// // (3, 4) -> (12,) にreshape
    /// let flattened = a.reshape(vec![Expr::from(12)]);
    ///
    /// // (3, 4) -> (2, 6) にreshape
    /// let reshaped = a.reshape(vec![Expr::from(2), Expr::from(6)]);
    /// ```
    pub fn reshape(&self, new_shape: Vec<Expr>) -> Self {
        let new_view = self.view.clone().reshape(new_shape);
        self.view(new_view)
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
    fn test_graph_to_dot() {
        let mut graph = Graph::new();

        // 入力ノードを作成
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // 計算グラフを構築
        let c = a + b;

        // 出力ノードを登録
        graph.output("c", c);

        // DOT形式で出力
        let dot = graph.to_dot();

        // 基本的な構造を確認
        assert!(dot.contains("digraph G {"));
        assert!(dot.contains("rankdir=LR"));
        assert!(dot.contains("Output: c"));
        assert!(dot.contains("Input"));
        assert!(dot.contains("Elementwise"));
        assert!(dot.contains("DType: F32"));
        assert!(dot.contains("Shape: [10, 20]"));
    }

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

        // 減算演算は a + (-b) として実装される
        match &result.op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Add,
                ..
            } => {}
            _ => panic!("Expected Add operation (a - b = a + (-b))"),
        }

        assert_eq!(result.src.len(), 2);

        // 左側のオペランドは入力a
        match &result.src[0].op {
            GraphOp::Input => {}
            _ => panic!("Expected Input operation for left operand"),
        }

        // 右側のオペランドは -b (Neg演算)
        match &result.src[1].op {
            GraphOp::Elementwise {
                op: ops::ElementwiseOp::Neg,
                ..
            } => {}
            _ => panic!("Expected Neg operation for right operand"),
        }

        // -b の入力は b
        match &result.src[1].src[0].op {
            GraphOp::Input => {}
            _ => panic!("Expected Input operation for negated operand"),
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

    #[test]
    fn test_elementwise_strategy_default() {
        let default_strategy = ElementwiseStrategy::default();
        assert_eq!(
            default_strategy,
            ElementwiseStrategy::Sequential {
                simd_width: 1,
                unroll_factor: 1
            }
        );
    }

    #[test]
    fn test_elementwise_strategy_sequential() {
        let strategy = ElementwiseStrategy::sequential();
        assert_eq!(
            strategy,
            ElementwiseStrategy::Sequential {
                simd_width: 1,
                unroll_factor: 1
            }
        );

        let strategy_simd = ElementwiseStrategy::sequential_simd(4);
        assert_eq!(
            strategy_simd,
            ElementwiseStrategy::Sequential {
                simd_width: 4,
                unroll_factor: 1
            }
        );

        let strategy_unroll = ElementwiseStrategy::sequential_unroll(2);
        assert_eq!(
            strategy_unroll,
            ElementwiseStrategy::Sequential {
                simd_width: 1,
                unroll_factor: 2
            }
        );

        let strategy_both = ElementwiseStrategy::sequential_simd_unroll(4, 2);
        assert_eq!(
            strategy_both,
            ElementwiseStrategy::Sequential {
                simd_width: 4,
                unroll_factor: 2
            }
        );
    }

    #[test]
    fn test_elementwise_strategy_thread() {
        let strategy = ElementwiseStrategy::thread();
        assert_eq!(
            strategy,
            ElementwiseStrategy::Thread {
                simd_width: 1,
                unroll_factor: 1
            }
        );

        let strategy_simd = ElementwiseStrategy::thread_simd(8);
        assert_eq!(
            strategy_simd,
            ElementwiseStrategy::Thread {
                simd_width: 8,
                unroll_factor: 1
            }
        );

        let strategy_unroll = ElementwiseStrategy::thread_unroll(4);
        assert_eq!(
            strategy_unroll,
            ElementwiseStrategy::Thread {
                simd_width: 1,
                unroll_factor: 4
            }
        );

        let strategy_both = ElementwiseStrategy::thread_simd_unroll(8, 4);
        assert_eq!(
            strategy_both,
            ElementwiseStrategy::Thread {
                simd_width: 8,
                unroll_factor: 4
            }
        );
    }

    #[test]
    fn test_elementwise_strategy_thread_group() {
        let strategy = ElementwiseStrategy::thread_group();
        assert_eq!(
            strategy,
            ElementwiseStrategy::ThreadGroup {
                simd_width: 1,
                unroll_factor: 1
            }
        );

        let strategy_simd = ElementwiseStrategy::thread_group_simd(16);
        assert_eq!(
            strategy_simd,
            ElementwiseStrategy::ThreadGroup {
                simd_width: 16,
                unroll_factor: 1
            }
        );

        let strategy_unroll = ElementwiseStrategy::thread_group_unroll(8);
        assert_eq!(
            strategy_unroll,
            ElementwiseStrategy::ThreadGroup {
                simd_width: 1,
                unroll_factor: 8
            }
        );

        let strategy_both = ElementwiseStrategy::thread_group_simd_unroll(16, 8);
        assert_eq!(
            strategy_both,
            ElementwiseStrategy::ThreadGroup {
                simd_width: 16,
                unroll_factor: 8
            }
        );
    }

    #[test]
    fn test_elementwise_strategy_accessors() {
        let strategy = ElementwiseStrategy::sequential_simd_unroll(4, 2);
        assert_eq!(strategy.simd_width(), 4);
        assert_eq!(strategy.unroll_factor(), 2);

        let strategy = ElementwiseStrategy::thread_simd_unroll(8, 4);
        assert_eq!(strategy.simd_width(), 8);
        assert_eq!(strategy.unroll_factor(), 4);

        let strategy = ElementwiseStrategy::thread_group_simd_unroll(16, 8);
        assert_eq!(strategy.simd_width(), 16);
        assert_eq!(strategy.unroll_factor(), 8);
    }

    #[test]
    fn test_reduce_strategy_default() {
        let default_strategy = ReduceStrategy::default();
        assert_eq!(
            default_strategy,
            ReduceStrategy::Sequential { unroll_factor: 1 }
        );
    }

    #[test]
    fn test_reduce_strategy_sequential() {
        let strategy = ReduceStrategy::sequential();
        assert_eq!(strategy, ReduceStrategy::Sequential { unroll_factor: 1 });

        let strategy_unroll = ReduceStrategy::sequential_unroll(4);
        assert_eq!(
            strategy_unroll,
            ReduceStrategy::Sequential { unroll_factor: 4 }
        );
    }

    #[test]
    fn test_reduce_strategy_accessors() {
        let strategy = ReduceStrategy::sequential_unroll(8);
        assert_eq!(strategy.unroll_factor(), 8);
    }

    #[test]
    fn test_cumulative_strategy_default() {
        let default_strategy = CumulativeStrategy::default();
        assert_eq!(
            default_strategy,
            CumulativeStrategy::Sequential { unroll_factor: 1 }
        );
    }

    #[test]
    fn test_cumulative_strategy_sequential() {
        let strategy = CumulativeStrategy::sequential();
        assert_eq!(
            strategy,
            CumulativeStrategy::Sequential { unroll_factor: 1 }
        );

        let strategy_unroll = CumulativeStrategy::sequential_unroll(4);
        assert_eq!(
            strategy_unroll,
            CumulativeStrategy::Sequential { unroll_factor: 4 }
        );
    }

    #[test]
    fn test_cumulative_strategy_accessors() {
        let strategy = CumulativeStrategy::sequential_unroll(8);
        assert_eq!(strategy.unroll_factor(), 8);
    }

    #[test]
    fn test_reduce_sum() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20, 30])
            .build();

        // 軸1を縮約（10, 20, 30 -> 10, 30）
        let result = input.reduce_sum(1);

        // 型が保持されていることを確認
        match result.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        // Viewのshapeが正しく縮約されていることを確認
        assert_eq!(result.view.ndim(), 2);
        assert_eq!(result.view.shape().len(), 2);

        // Reduceオペレーションが正しく設定されていることを確認
        match &result.op {
            GraphOp::Reduce { op, axis, .. } => {
                assert_eq!(*op, ReduceOp::Sum);
                assert_eq!(*axis, 1);
            }
            _ => panic!("Expected GraphOp::Reduce"),
        }
    }

    #[test]
    fn test_reduce_mul() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 10])
            .build();

        // 軸0を縮約（5, 10 -> 10）
        let result = input.reduce_mul(0);

        // Viewのshapeが正しく縮約されていることを確認
        assert_eq!(result.view.ndim(), 1);
        assert_eq!(result.view.shape().len(), 1);

        // Reduceオペレーションが正しく設定されていることを確認
        match &result.op {
            GraphOp::Reduce { op, axis, .. } => {
                assert_eq!(*op, ReduceOp::Prod);
                assert_eq!(*axis, 0);
            }
            _ => panic!("Expected GraphOp::Reduce"),
        }
    }

    #[test]
    fn test_reduce_max() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 4, 5])
            .build();

        // 軸2を縮約（3, 4, 5 -> 3, 4）
        let result = input.reduce_max(2);

        // Viewのshapeが正しく縮約されていることを確認
        assert_eq!(result.view.ndim(), 2);

        // Reduceオペレーションが正しく設定されていることを確認
        match &result.op {
            GraphOp::Reduce { op, axis, .. } => {
                assert_eq!(*op, ReduceOp::Max);
                assert_eq!(*axis, 2);
            }
            _ => panic!("Expected GraphOp::Reduce"),
        }
    }

    #[test]
    fn test_view_method() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 4])
            .build();

        // Viewを変更（転置）
        let transposed_view = input.view.clone().permute(vec![1, 0]);
        let transposed = input.view(transposed_view.clone());

        // dtypeが保持されていることを確認
        match transposed.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        // Viewが正しく設定されていることを確認
        assert_eq!(transposed.view, transposed_view);
        assert_eq!(transposed.view.ndim(), 2);

        // GraphOp::Viewが設定されていることを確認
        match &transposed.op {
            GraphOp::View(v) => {
                assert_eq!(*v, transposed_view);
            }
            _ => panic!("Expected GraphOp::View"),
        }

        // 元のノードが入力として保持されていることを確認
        assert_eq!(transposed.src.len(), 1);
    }

    #[test]
    fn test_view_method_unsqueeze() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 4])
            .build();

        // Viewを変更（次元追加）
        let unsqueezed_view = input.view.clone().unsqueeze(0);
        let unsqueezed = input.view(unsqueezed_view.clone());

        // Viewが正しく設定されていることを確認
        assert_eq!(unsqueezed.view.ndim(), 3);

        // GraphOp::Viewが設定されていることを確認
        match &unsqueezed.op {
            GraphOp::View(v) => {
                assert_eq!(*v, unsqueezed_view);
            }
            _ => panic!("Expected GraphOp::View"),
        }
    }

    #[test]
    fn test_reduce_to_scalar() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        // 唯一の軸を縮約してスカラーに（10 -> []）
        let result = input.reduce_sum(0);

        // スカラー（ndim=0）になることを確認
        assert_eq!(result.view.ndim(), 0);
        assert_eq!(result.view.shape().len(), 0);
    }

    #[test]
    #[should_panic(expected = "axis 3 is out of bounds")]
    fn test_reduce_out_of_bounds() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // 存在しない軸3を指定してパニック
        let _result = input.reduce_sum(3);
    }

    #[test]
    fn test_reduce_generic() {
        let mut graph = Graph::new();
        let input = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![5, 10, 15])
            .build();

        // ReduceOpを直接指定
        let result = input.reduce(ReduceOp::Sum, 1);

        match &result.op {
            GraphOp::Reduce { op, axis, .. } => {
                assert_eq!(*op, ReduceOp::Sum);
                assert_eq!(*axis, 1);
            }
            _ => panic!("Expected GraphOp::Reduce"),
        }
    }

    #[test]
    fn test_constant_f32() {
        // F32定数ノードを作成
        let const_node = GraphNode::constant(2.5f32);

        // dtypeがF32であることを確認
        match const_node.dtype {
            DType::F32 => {}
            _ => panic!("Expected DType::F32"),
        }

        // スカラー（ndim=0）であることを確認
        assert_eq!(const_node.view.ndim(), 0);
        assert_eq!(const_node.view.shape().len(), 0);

        // GraphOp::Constであることを確認
        match &const_node.op {
            GraphOp::Const(crate::ast::Literal::F32(v)) => {
                assert_eq!(*v, 2.5f32);
            }
            _ => panic!("Expected GraphOp::Const with F32 literal"),
        }

        // 入力ノードがないことを確認
        assert_eq!(const_node.src.len(), 0);
    }

    #[test]
    fn test_constant_isize() {
        // isize定数ノードを作成
        let const_node = GraphNode::constant(42isize);

        // スカラーであることを確認
        assert_eq!(const_node.view.ndim(), 0);

        // GraphOp::Constであることを確認
        match &const_node.op {
            GraphOp::Const(crate::ast::Literal::Int(v)) => {
                assert_eq!(*v, 42);
            }
            _ => panic!("Expected GraphOp::Const with Int literal"),
        }
    }

    #[test]
    fn test_constant_usize() {
        // usize定数ノードを作成
        let const_node = GraphNode::constant(100usize);

        // スカラーであることを確認
        assert_eq!(const_node.view.ndim(), 0);

        // GraphOp::Constであることを確認
        match &const_node.op {
            GraphOp::Const(crate::ast::Literal::Int(v)) => {
                assert_eq!(*v, 100);
            }
            _ => panic!("Expected GraphOp::Const with Int literal"),
        }
    }

    #[test]
    fn test_reshape() {
        use crate::graph::shape::Expr;

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 4])
            .build();

        // (3, 4) -> (12,) にreshape
        let flattened = a.reshape(vec![Expr::from(12)]);
        assert_eq!(flattened.view.shape(), &[Expr::from(12)]);

        // GraphOp::Viewであることを確認
        match &flattened.op {
            GraphOp::View(v) => {
                assert_eq!(v.shape(), &[Expr::from(12)]);
            }
            _ => panic!("Expected GraphOp::View"),
        }

        // (3, 4) -> (2, 6) にreshape
        let reshaped = a.reshape(vec![Expr::from(2), Expr::from(6)]);
        assert_eq!(reshaped.view.shape(), &[Expr::from(2), Expr::from(6)]);
    }

    #[test]
    #[should_panic(expected = "reshape can only be applied to contiguous views")]
    fn test_reshape_non_contiguous() {
        use crate::graph::shape::Expr;

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![3, 4])
            .build();

        // Permuteして非連続にする
        let transposed = a.view(a.view.clone().permute(vec![1, 0]));

        // 非連続なViewに対してreshapeを試みる（panicするはず）
        let _ = transposed.reshape(vec![Expr::from(12)]);
    }

    #[test]
    fn test_recip_method() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // メソッド形式でrecipを呼び出し
        let result = a.recip();

        // 正しいGraphOpが生成されたことを確認
        match &result.op {
            GraphOp::Elementwise { op, .. } => {
                assert!(matches!(op, ops::ElementwiseOp::Recip));
            }
            _ => panic!("Expected Elementwise::Recip"),
        }

        // 形状とDTypeが保持されていることを確認
        assert_eq!(result.view.shape().len(), 2);
        assert!(matches!(result.dtype, DType::F32));
    }

    #[test]
    fn test_max_method() {
        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();
        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // メソッド形式でmaxを呼び出し: a.max(b)
        let result = a.max(b);

        // 正しいGraphOpが生成されたことを確認
        match &result.op {
            GraphOp::Elementwise { op, .. } => {
                assert!(matches!(op, ops::ElementwiseOp::Max));
            }
            _ => panic!("Expected Elementwise::Max"),
        }

        // 形状とDTypeが保持されていることを確認
        assert_eq!(result.view.shape().len(), 2);
        assert!(matches!(result.dtype, DType::F32));
    }

    #[test]
    fn test_method_chaining() {
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

        // メソッドチェーン: a.max(b).recip().reduce_sum(0)
        let result = a.max(b).recip().reduce_sum(0);

        // 結果がスカラーであることを確認
        assert_eq!(result.view.shape().len(), 0);
    }
}

use std::{
    collections::{BTreeMap, HashMap},
    ops::Deref,
    rc::{Rc, Weak},
};
pub mod hlops;
pub mod hlops_conv;
pub mod node_view_ops;
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
    Bool,    // boolean (internally u8: 0 = false, non-zero = true)
    F32,     // 32-bit floating point
    Complex, // complex number (lowered to two F32 values: real and imaginary)
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

    /// このGraphを実行（tinygradスタイル）
    ///
    /// # 引数
    /// - `inputs`: 入力ノード名 -> データのマッピング
    ///
    /// # 戻り値
    /// 出力ノード名 -> 計算結果のマッピング
    pub fn realize(
        &self,
        inputs: HashMap<String, Vec<f32>>,
    ) -> Result<HashMap<String, Vec<f32>>, String> {
        self.realize_with_device(inputs, None)
    }

    /// このGraphを実行（デバイス指定版）
    pub fn realize_with_device(
        &self,
        _inputs: HashMap<String, Vec<f32>>,
        _device: Option<crate::backend::Device>,
    ) -> Result<HashMap<String, Vec<f32>>, String> {
        // TODO: 実装を完成させる
        // 現在は簡易版として、エラーを返す
        Err(
            "Graph::realize() is not yet fully implemented. Use device pipeline directly for now."
                .to_string(),
        )
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
            crate::ast::Literal::Bool(_) => DType::Bool,
            crate::ast::Literal::F32(_) => DType::F32,
            crate::ast::Literal::Int(_) => DType::Unknown, // Intは将来的に追加
        };
        // 定数はスカラー（shape=[]）
        let view = View::contiguous(Vec::<isize>::new());
        Self::new(dtype, GraphOp::Const(literal), vec![], view)
    }

    /// 複素数スカラー定数ノードを作成
    ///
    /// # 例
    /// ```
    /// use harp::prelude::*;
    ///
    /// // 複素数定数 (1.0 + 2.0i)
    /// let complex_const = GraphNode::complex_constant(1.0, 2.0);
    ///
    /// // タプルから
    /// let complex_const2 = GraphNode::complex_constant_from((3.0, 4.0));
    /// ```
    pub fn complex_constant(re: f32, im: f32) -> Self {
        let view = View::contiguous(Vec::<isize>::new());
        Self::new(
            DType::Complex,
            GraphOp::ComplexConst { re, im },
            vec![],
            view,
        )
    }

    /// タプルから複素数スカラー定数ノードを作成
    pub fn complex_constant_from(value: (f32, f32)) -> Self {
        Self::complex_constant(value.0, value.1)
    }

    /// 指定した形状の一様乱数テンソルを生成 [0, 1)
    ///
    /// # 引数
    /// - `shape`: テンソルの形状（静的な`usize`/`isize`または動的な`Expr`を受け付ける）
    ///
    /// # 例
    /// ```ignore
    /// use harp::prelude::*;
    ///
    /// // 静的な形状: 2x3の乱数テンソル
    /// let rand_node = GraphNode::rand(vec![2, 3]);
    ///
    /// // 動的な形状: Expr型を使用
    /// let batch_size = shape::Expr::var("batch");
    /// let rand_node = GraphNode::rand(vec![batch_size, 64.into()]);
    /// ```
    pub fn rand<E: Into<shape::Expr> + Clone, I: IntoIterator<Item = E>>(shape: I) -> Self {
        let shape_exprs: Vec<shape::Expr> = shape.into_iter().map(|e| e.into()).collect();
        let view = View::contiguous(shape_exprs);
        Self::new(
            DType::F32,
            GraphOp::Rand {
                elementwise_strategies: None,
            },
            vec![],
            view,
        )
    }

    /// 連番テンソル `[0, 1, 2, ..., size-1]` を生成
    ///
    /// PyTorchの`torch.arange(size)`に相当します。
    /// 他の範囲やステップが必要な場合は、この結果に演算を適用してください：
    /// - `arange(n) + start` → `[start, start+1, ..., start+n-1]`
    /// - `arange(n) * step` → `[0, step, 2*step, ..., (n-1)*step]`
    /// - `arange(n) * step + start` → `[start, start+step, ..., start+(n-1)*step]`
    ///
    /// # 例
    /// ```
    /// use harp::graph::GraphNode;
    ///
    /// // 基本形: [0, 1, 2, 3, 4]
    /// let x = GraphNode::arange(5);
    ///
    /// // [1, 2, 3, 4, 5] (start=1)
    /// let x = GraphNode::arange(5) + 1.0f32;
    ///
    /// // [0.0, 0.5, 1.0, 1.5, 2.0] (step=0.5)
    /// let x = GraphNode::arange(5) * 0.5f32;
    ///
    /// // [10.0, 12.0, 14.0, 16.0, 18.0] (start=10, step=2)
    /// let x = GraphNode::arange(5) * 2.0f32 + 10.0f32;
    /// ```
    pub fn arange<E: Into<shape::Expr>>(size: E) -> Self {
        let size_expr: shape::Expr = size.into();
        let view = View::contiguous(vec![size_expr]);
        Self::new(
            DType::F32,
            GraphOp::Arange {
                elementwise_strategies: None,
            },
            vec![],
            view,
        )
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

    /// 累積演算（汎用）
    pub fn cumulative(&self, op: ops::CumulativeOp, axis: usize) -> Self {
        ops::cumulative(self.clone(), op, axis)
    }

    /// 累積和（cumulative sum）
    ///
    /// 指定軸に沿って累積和を計算します。
    /// 例: [1, 2, 3, 4] -> [1, 3, 6, 10]
    pub fn cumsum(&self, axis: usize) -> Self {
        ops::cumsum(self.clone(), axis)
    }

    /// 累積積（cumulative product）
    ///
    /// 指定軸に沿って累積積を計算します。
    /// 例: [1, 2, 3, 4] -> [1, 2, 6, 24]
    pub fn cumprod(&self, axis: usize) -> Self {
        ops::cumprod(self.clone(), axis)
    }

    /// elementwise演算の後に累積演算を行う融合ノードを作成
    pub fn fused_elementwise_cumulative(
        inputs: Vec<Self>,
        expr: crate::ast::AstNode,
        cumulative_op: ops::CumulativeOp,
        axis: usize,
    ) -> Self {
        ops::fused_elementwise_cumulative(inputs, expr, cumulative_op, axis)
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

    /// 複数のテンソルを指定した軸で結合（staticメソッド）
    ///
    /// # 引数
    /// - `inputs`: 結合するテンソルのベクター
    /// - `axis`: 結合する軸
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a").with_dtype(DType::F32).with_shape([2, 3]).build();
    /// let b = graph.input("b").with_dtype(DType::F32).with_shape([2, 5]).build();
    ///
    /// // axis=1で結合: [2, 3] + [2, 5] => [2, 8]
    /// let c = GraphNode::concat(vec![a, b], 1);
    /// ```
    pub fn concat(inputs: Vec<Self>, axis: usize) -> Self {
        ops::concat(inputs, axis)
    }

    /// このテンソルと別のテンソルを指定した軸で結合
    ///
    /// # 引数
    /// - `other`: 結合するテンソル
    /// - `axis`: 結合する軸
    ///
    /// # 例
    /// ```no_run
    /// use harp::prelude::*;
    ///
    /// let mut graph = Graph::new();
    /// let a = graph.input("a").with_dtype(DType::F32).with_shape([2, 3]).build();
    /// let b = graph.input("b").with_dtype(DType::F32).with_shape([2, 5]).build();
    ///
    /// // axis=1で結合: [2, 3] + [2, 5] => [2, 8]
    /// let c = a.cat(b, 1);
    /// ```
    pub fn cat(self, other: Self, axis: usize) -> Self {
        ops::concat(vec![self, other], axis)
    }
}

// .0 のように書かなくても内部のデータを読み取れるようにする
impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

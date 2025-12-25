use std::{
    collections::{BTreeMap, HashMap},
    ops::Deref,
    rc::Rc,
};
pub mod conv;
pub mod custom_builder;
pub mod hlops;
pub mod hlops_conv;
pub mod node_view_ops;
pub mod ops;
pub mod shape;
pub mod strategy;

// Re-export commonly used types
pub use conv::IntoSpatialParams;
pub use ops::{CumulativeOp, ElementwiseOp, GraphOp, ReduceOp, custom_placeholders};
pub use shape::{Expr, View};
// Note: ElementwiseStrategy was removed - parallelization is now handled at AST level
pub use strategy::{CumulativeStrategy, ReduceStrategy};

/// 入力バッファのメタデータ
#[derive(Debug, Clone)]
pub struct InputMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<Expr>,
}

/// 出力バッファのメタデータ
#[derive(Debug, Clone)]
pub struct OutputMeta {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<Expr>,
}

#[derive(Debug, Clone)]
pub struct Graph {
    input_metas: Vec<InputMeta>,               // 入力バッファのメタデータ
    output_metas: Vec<OutputMeta>,             // 出力バッファのメタデータ
    output_nodes: BTreeMap<String, GraphNode>, // 出力ノード（名前→ノード）
    shape_var_defaults: HashMap<String, i64>,  // 動的shape変数のデフォルト値（必須）
    subgraphs: HashMap<String, Graph>,         // サブグラフ定義（DSLのgraph main以外）
}

#[derive(Debug, Clone)]
pub struct GraphNodeData {
    pub dtype: DType,
    pub op: GraphOp,
    pub src: Vec<GraphNode>, // 入力ノード
    pub view: View,
    pub name: Option<String>, // デバッグ/DSL用のラベル（重複可）
}

#[derive(Debug, Clone)]
pub struct GraphNode(Rc<GraphNodeData>);

// AstNoderのDTypeとは異なり、VecやPtrは扱わない。
#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Unknown, // 未定または未知, プレースホルダー
    Bool,    // boolean (internally u8: 0 = false, non-zero = true)
    I64,     // 64-bit signed integer (for indexing/counters)
    I32,     // 32-bit signed integer
    F32,     // 32-bit floating point
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
            input_metas: Vec::new(),
            output_metas: Vec::new(),
            output_nodes: BTreeMap::new(),
            shape_var_defaults: HashMap::new(),
            subgraphs: HashMap::new(),
        }
    }

    /// 入力ノードを新規作成
    ///
    /// # Example
    /// ```
    /// use harp::{Graph, DType};
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, [4, 8]);
    /// ```
    pub fn input<E, I>(&mut self, name: &str, dtype: DType, shape: I) -> GraphNode
    where
        E: Into<shape::Expr> + Clone,
        I: IntoIterator<Item = E>,
    {
        let shape: Vec<shape::Expr> = shape.into_iter().map(|e| e.into()).collect();
        let view = View::contiguous(shape.clone());
        let node = GraphNode::new(
            dtype.clone(),
            GraphOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
        );
        // メタデータを登録
        self.input_metas.push(InputMeta {
            name: name.to_string(),
            dtype,
            shape,
        });
        node
    }

    /// 名前付きバッファノードを作成（Kernelノード用）
    ///
    /// Graph.inputsには登録されない内部バッファを作成します。
    /// 主にKernelノードの出力バッファとして使用されます。
    pub fn buffer<E, I>(name: &str, dtype: DType, shape: I) -> GraphNode
    where
        E: Into<shape::Expr> + Clone,
        I: IntoIterator<Item = E>,
    {
        let shape: Vec<shape::Expr> = shape.into_iter().map(|e| e.into()).collect();
        let view = View::contiguous(shape);
        GraphNode::new(
            dtype,
            GraphOp::Buffer {
                name: name.to_string(),
            },
            vec![],
            view,
        )
    }

    /// 出力ノードを登録
    ///
    /// 出力ノードを登録します。
    /// 複数の出力をサポートしています。
    /// 出力のメタデータ（dtype, shape）は自動的に記録されます。
    ///
    /// # Example
    /// ```
    /// use harp::{Graph, DType};
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, [10]);
    /// let y = &x + 1.0f32;
    /// let z = &x * 2.0f32;
    /// graph.output("y", y);  // 1つ目の出力
    /// graph.output("z", z);  // 2つ目の出力（複数出力サポート）
    /// ```
    pub fn output(&mut self, name: &str, output_node: GraphNode) {
        // メタデータを登録
        self.output_metas.push(OutputMeta {
            name: name.to_string(),
            dtype: output_node.dtype.clone(),
            shape: output_node.view.shape().to_vec(),
        });

        // 出力ノードを直接登録
        self.output_nodes.insert(name.to_string(), output_node);
    }

    /// 出力ノードのマップを取得
    pub fn outputs(&self) -> &BTreeMap<String, GraphNode> {
        &self.output_nodes
    }

    /// 出力ノードを設定（最適化時に使用）
    pub fn set_output_node(&mut self, name: String, node: GraphNode) {
        self.output_nodes.insert(name, node);
    }

    /// 出力ノードを一括設定（最適化時に使用）
    pub fn set_output_nodes(&mut self, nodes: BTreeMap<String, GraphNode>) {
        self.output_nodes = nodes;
    }

    /// 出力名のリストを取得（ソート済み）
    pub fn output_names(&self) -> Vec<String> {
        self.output_nodes.keys().cloned().collect()
    }

    /// 入力メタデータへのアクセス
    pub fn input_metas(&self) -> &[InputMeta] {
        &self.input_metas
    }

    /// 出力メタデータへのアクセス
    pub fn output_metas(&self) -> &[OutputMeta] {
        &self.output_metas
    }

    /// 入力メタデータを登録（最適化時に使用）
    pub fn register_input_meta(&mut self, name: String, dtype: DType, shape: Vec<Expr>) {
        // 重複チェック
        if !self.input_metas.iter().any(|m| m.name == name) {
            self.input_metas.push(InputMeta { name, dtype, shape });
        }
    }

    /// 入力メタデータをコピー（最適化時にグラフを再構築する際に使用）
    pub fn copy_input_metas_from(&mut self, other: &Graph) {
        self.input_metas = other.input_metas.clone();
    }

    /// 出力メタデータをコピー（最適化時にグラフを再構築する際に使用）
    pub fn copy_output_metas_from(&mut self, other: &Graph) {
        self.output_metas = other.output_metas.clone();
    }

    // shape変数のデフォルト値を設定
    pub fn set_shape_var_default(&mut self, name: impl Into<String>, default_value: i64) {
        self.shape_var_defaults.insert(name.into(), default_value);
    }

    // shape変数のデフォルト値を取得
    pub fn shape_var_defaults(&self) -> &HashMap<String, i64> {
        &self.shape_var_defaults
    }

    /// サブグラフを追加
    ///
    /// DSLでmain以外のgraphを定義した場合に使用します。
    /// サブグラフはSubgraphCall演算から参照されます。
    pub fn add_subgraph(&mut self, name: impl Into<String>, subgraph: Graph) {
        self.subgraphs.insert(name.into(), subgraph);
    }

    /// サブグラフを取得
    pub fn subgraph(&self, name: &str) -> Option<&Graph> {
        self.subgraphs.get(name)
    }

    /// サブグラフを可変参照で取得
    pub fn subgraph_mut(&mut self, name: &str) -> Option<&mut Graph> {
        self.subgraphs.get_mut(name)
    }

    /// 全てのサブグラフへのアクセス
    pub fn subgraphs(&self) -> &HashMap<String, Graph> {
        &self.subgraphs
    }

    /// サブグラフをコピー（最適化時にグラフを再構築する際に使用）
    pub fn copy_subgraphs_from(&mut self, other: &Graph) {
        self.subgraphs = other.subgraphs.clone();
    }
}
impl GraphNode {
    pub fn new(dtype: DType, op: GraphOp, src: Vec<GraphNode>, view: View) -> Self {
        Self(Rc::new(GraphNodeData {
            dtype,
            op,
            src,
            view,
            name: None,
        }))
    }

    /// 名前付きでノードを作成
    pub fn new_named(
        dtype: DType,
        op: GraphOp,
        src: Vec<GraphNode>,
        view: View,
        name: impl Into<String>,
    ) -> Self {
        Self(Rc::new(GraphNodeData {
            dtype,
            op,
            src,
            view,
            name: Some(name.into()),
        }))
    }

    /// Rcから直接GraphNodeを作成（最適化時に使用）
    pub fn from_rc(rc: Rc<GraphNodeData>) -> Self {
        Self(rc)
    }

    /// ノードのポインタを取得（トポロジカルソートなどで識別に使用）
    pub fn as_ptr(&self) -> *const GraphNodeData {
        Rc::as_ptr(&self.0)
    }

    /// ノードに名前を設定（新しいノードを返す）
    ///
    /// DSLからの変換時に変数名を保持するために使用します。
    /// 名前は重複可能で、デバッグや可読性向上のためのラベルとして機能します。
    ///
    /// # 例
    /// ```
    /// use harp::graph::{GraphNode, DType};
    ///
    /// let node = GraphNode::arange(10).with_name("indices");
    /// assert_eq!(node.name(), Some("indices"));
    /// ```
    pub fn with_name(self, name: impl Into<String>) -> Self {
        Self(Rc::new(GraphNodeData {
            dtype: self.dtype.clone(),
            op: self.op.clone(),
            src: self.src.clone(),
            view: self.view.clone(),
            name: Some(name.into()),
        }))
    }

    /// ノードの名前を取得
    pub fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
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
            crate::ast::Literal::I64(_) => DType::I64, // i64 リテラルは I64 として扱う
            crate::ast::Literal::I32(_) => DType::I32,
        };
        // 定数はスカラー（shape=[]）
        let view = View::contiguous(Vec::<isize>::new());
        Self::new(dtype, GraphOp::Const(literal), vec![], view)
    }

    /// 指定した形状でゼロ埋めされたテンソルを作成
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    ///
    /// # 例
    /// ```
    /// use harp::graph::GraphNode;
    ///
    /// // 3x4のゼロテンソル
    /// let zeros = GraphNode::zeros(vec![3, 4]);
    /// ```
    pub fn zeros<E: Into<shape::Expr> + Clone, I: IntoIterator<Item = E>>(shape: I) -> Self {
        Self::full(0.0f32, shape)
    }

    /// 指定した形状で1埋めされたテンソルを作成
    ///
    /// # 引数
    /// - `shape`: テンソルの形状
    ///
    /// # 例
    /// ```
    /// use harp::graph::GraphNode;
    ///
    /// // 2x3の1テンソル
    /// let ones = GraphNode::ones(vec![2, 3]);
    /// ```
    pub fn ones<E: Into<shape::Expr> + Clone, I: IntoIterator<Item = E>>(shape: I) -> Self {
        Self::full(1.0f32, shape)
    }

    /// 指定した形状で定数値埋めされたテンソルを作成
    ///
    /// # 引数
    /// - `value`: 埋める値
    /// - `shape`: テンソルの形状
    ///
    /// # 例
    /// ```
    /// use harp::graph::GraphNode;
    ///
    /// // 2x3のテンソル、すべて3.14で埋める
    /// let filled = GraphNode::full(3.14f32, vec![2, 3]);
    /// ```
    pub fn full<E: Into<shape::Expr> + Clone, I: IntoIterator<Item = E>>(
        value: f32,
        shape: I,
    ) -> Self {
        let shape_exprs: Vec<shape::Expr> = shape.into_iter().map(|e| e.into()).collect();
        let view = View::contiguous(shape_exprs);
        let literal = crate::ast::Literal::F32(value);
        // ConstFill: 形状全体を定数で埋める
        Self::new(DType::F32, GraphOp::ConstFill(literal), vec![], view)
    }

    /// 指定した形状の一様乱数テンソルを生成 [0, 1)
    ///
    /// # 引数
    /// - `shape`: テンソルの形状（静的な`usize`/`isize`または動的な`Expr`を受け付ける）
    ///
    /// # 例
    /// ```
    /// use harp::graph::{GraphNode, Expr};
    ///
    /// // 静的な形状: 2x3の乱数テンソル
    /// let rand_node = GraphNode::rand(vec![2, 3]);
    ///
    /// // 動的な形状: Expr型を使用
    /// let batch_size = Expr::Var("batch".to_string());
    /// let rand_node = GraphNode::rand(vec![batch_size, 64.into()]);
    /// ```
    pub fn rand<E: Into<shape::Expr> + Clone, I: IntoIterator<Item = E>>(shape: I) -> Self {
        let shape_exprs: Vec<shape::Expr> = shape.into_iter().map(|e| e.into()).collect();
        let view = View::contiguous(shape_exprs);
        Self::new(DType::F32, GraphOp::Rand {}, vec![], view)
    }

    /// 連番テンソル `[0, 1, 2, ..., size-1]` を生成（I32型）
    ///
    /// PyTorchの`torch.arange(size, dtype=torch.int32)`に相当します。
    /// 浮動小数点が必要な場合は`.cast(DType::F32)`を使用してください。
    ///
    /// # 例
    /// ```
    /// use harp::graph::{GraphNode, DType};
    ///
    /// // 基本形: [0, 1, 2, 3, 4] (I32)
    /// let indices = GraphNode::arange(5);
    ///
    /// // floatに変換: [0.0, 1.0, 2.0, 3.0, 4.0] (F32)
    /// let floats = GraphNode::arange(5).cast(DType::F32);
    ///
    /// // float演算と組み合わせ: [0.0, 0.5, 1.0, 1.5, 2.0]
    /// let scaled = GraphNode::arange(5).cast(DType::F32) * 0.5f32;
    /// ```
    pub fn arange<E: Into<shape::Expr>>(size: E) -> Self {
        let size_expr: shape::Expr = size.into();
        let view = View::contiguous(vec![size_expr]);
        Self::new(DType::I32, GraphOp::Arange {}, vec![], view)
    }

    /// 型変換（キャスト）
    ///
    /// テンソルの要素をターゲット型に変換します。
    ///
    /// # 引数
    /// - `target_dtype`: 変換先の型
    ///
    /// # 例
    /// ```
    /// use harp::graph::{GraphNode, DType};
    ///
    /// // I32からF32への変換
    /// let indices = GraphNode::arange(5);  // [0, 1, 2, 3, 4] (I32)
    /// let floats = indices.cast(DType::F32);  // [0.0, 1.0, 2.0, 3.0, 4.0] (F32)
    /// ```
    pub fn cast(&self, target_dtype: DType) -> Self {
        // 既に同じ型なら何もしない
        if self.dtype == target_dtype {
            return self.clone();
        }
        Self::new(
            target_dtype.clone(),
            GraphOp::Cast { target_dtype },
            vec![self.clone()],
            self.view.clone(),
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

    /// 逆数を計算（1/x）
    pub fn recip(self) -> Self {
        let view = self.view.clone();
        let dtype = self.dtype.clone();
        Self::new(
            dtype,
            ops::GraphOp::Elementwise {
                op: ops::ElementwiseOp::Recip,
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
    /// let a = graph.input("a", DType::F32, [2, 3]);
    /// let b = graph.input("b", DType::F32, [2, 5]);
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
    /// let a = graph.input("a", DType::F32, [2, 3]);
    /// let b = graph.input("b", DType::F32, [2, 5]);
    ///
    /// // axis=1で結合: [2, 3] + [2, 5] => [2, 8]
    /// let c = a.cat(b, 1);
    /// ```
    pub fn cat(self, other: Self, axis: usize) -> Self {
        ops::concat(vec![self, other], axis)
    }

    /// サブグラフ呼び出しノードを作成
    ///
    /// DSLコンパイラから使用されます。
    /// サブグラフの全出力をまとめて表現するノードを作成します。
    ///
    /// # 引数
    /// - `name`: サブグラフ名
    /// - `inputs`: サブグラフへの入力テンソル
    /// - `output_dtype`: 出力の型（単一出力の場合）または最初の出力の型
    /// - `output_view`: 出力のView（単一出力の場合）または最初の出力のView
    pub fn subgraph_call(
        name: impl Into<String>,
        inputs: Vec<Self>,
        output_dtype: DType,
        output_view: View,
    ) -> Self {
        Self::new(
            output_dtype,
            GraphOp::SubgraphCall { name: name.into() },
            inputs,
            output_view,
        )
    }

    /// サブグラフの特定出力を取り出すノードを作成
    ///
    /// 複数出力を持つサブグラフから特定の出力を取り出します。
    ///
    /// # 引数
    /// - `subgraph_call`: SubgraphCallノード
    /// - `output_index`: 取り出す出力のインデックス
    /// - `output_name`: 出力名
    /// - `output_dtype`: 出力の型
    /// - `output_view`: 出力のView
    pub fn subgraph_output(
        subgraph_call: Self,
        output_index: usize,
        output_name: impl Into<String>,
        output_dtype: DType,
        output_view: View,
    ) -> Self {
        Self::new(
            output_dtype,
            GraphOp::SubgraphOutput {
                output_index,
                output_name: output_name.into(),
            },
            vec![subgraph_call],
            output_view,
        )
    }
}

// .0 のように書かなくても内部のデータを読み取れるようにする
impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

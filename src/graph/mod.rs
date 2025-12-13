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
pub mod visualization;

// Re-export commonly used types
pub use conv::{ConvParams, IntoSpatialParams};
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
    input_metas: Vec<InputMeta>,                // 入力バッファのメタデータ
    output_metas: Vec<OutputMeta>,              // 出力バッファのメタデータ
    sink: Option<GraphNode>,                    // Sinkノード（グラフのルート）
    shape_var_defaults: HashMap<String, isize>, // 動的shape変数のデフォルト値（必須）
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
#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Unknown, // 未定または未知, プレースホルダー
    Bool,    // boolean (internally u8: 0 = false, non-zero = true)
    I32,     // 32-bit signed integer
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
            input_metas: Vec::new(),
            output_metas: Vec::new(),
            sink: None,
            shape_var_defaults: HashMap::new(),
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

    /// 名前付きバッファノードを作成（Customノード用）
    ///
    /// Graph.inputsには登録されない内部バッファを作成します。
    /// 主にCustomノードの出力バッファとして使用されます。
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
    /// 複数の出力をサポートしており、各出力に対してSinkノードが更新されます。
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

        // Sinkノードを作成/更新（出力Bufferは不要、メタデータで管理）
        self.update_sink(name.to_string(), output_node);
    }

    /// Sinkノードを作成または更新
    fn update_sink(&mut self, name: String, output_node: GraphNode) {
        use crate::ast::AstNode;

        match &mut self.sink {
            None => {
                // 最初の出力: Sinkノードを新規作成
                // 空のProgramを作成（後でProgramRootAbsorptionSuggesterが関数を追加）
                let empty_program = AstNode::Program {
                    functions: vec![],
                    entry_point: "harp_main".to_string(),
                };

                let sink = GraphNode::new(
                    DType::Unknown, // Sinkは型を持たない
                    GraphOp::ProgramRoot {
                        ast: empty_program,
                        outputs: vec![name],
                    },
                    vec![output_node], // 出力ノードのみ（Bufferは不要）
                    View::contiguous(Vec::<isize>::new()), // スカラービュー
                );

                self.sink = Some(sink);
            }
            Some(existing_sink) => {
                // 追加の出力: 既存Sinkを更新
                let mut new_src = existing_sink.src.clone();
                new_src.push(output_node);

                let (ast, mut outputs) = match &existing_sink.op {
                    GraphOp::ProgramRoot { ast, outputs } => (ast.clone(), outputs.clone()),
                    _ => panic!("Expected Sink node"),
                };
                outputs.push(name);

                let new_sink = GraphNode::new(
                    DType::Unknown,
                    GraphOp::ProgramRoot { ast, outputs },
                    new_src,
                    View::contiguous(Vec::<isize>::new()),
                );

                self.sink = Some(new_sink);
            }
        }
    }

    /// 出力ノードのマップを取得
    ///
    /// Sinkノードから出力情報を再構築してBTreeMapとして返します。
    /// 最適化後、Sinkのsrcが空の場合はSink自体を出力として返します。
    pub fn outputs(&self) -> BTreeMap<String, GraphNode> {
        let mut result = BTreeMap::new();

        if let Some(sink) = &self.sink
            && let GraphOp::ProgramRoot { outputs, .. } = &sink.op
        {
            if sink.src.is_empty() {
                // 最適化後: srcが空の場合はSink自体を返す
                for name in outputs.iter() {
                    result.insert(name.clone(), sink.clone());
                }
            } else {
                // srcは [output_node0, output_node1, ...] の順
                for (i, name) in outputs.iter().enumerate() {
                    if i < sink.src.len() {
                        result.insert(name.clone(), sink.src[i].clone());
                    }
                }
            }
        }

        result
    }

    /// Sinkノードへのアクセス
    pub fn sink(&self) -> Option<&GraphNode> {
        self.sink.as_ref()
    }

    /// Sinkノードを設定（最適化時に使用）
    pub fn set_sink(&mut self, sink: GraphNode) {
        self.sink = Some(sink);
    }

    /// 出力名のリストを取得（ソート済み）
    pub fn output_names(&self) -> Vec<String> {
        if let Some(sink) = &self.sink
            && let GraphOp::ProgramRoot { outputs, .. } = &sink.op
        {
            let mut names = outputs.clone();
            names.sort();
            return names;
        }
        Vec::new()
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
impl GraphNode {
    pub fn new(dtype: DType, op: GraphOp, src: Vec<GraphNode>, view: View) -> Self {
        Self(Rc::new(GraphNodeData {
            dtype,
            op,
            src,
            view,
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
    /// ```ignore
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

    /// 複素数テンソルから実部を取り出す
    ///
    /// # 例
    /// ```
    /// use harp::graph::{Graph, GraphNode, DType};
    ///
    /// let mut graph = Graph::new();
    /// let z = graph.input("z", DType::Complex, [4]);
    /// let re = z.real();  // 実部（F32）
    /// assert_eq!(re.dtype, DType::F32);
    /// ```
    ///
    /// # パニック
    /// 入力がComplex型でない場合にパニックします。
    pub fn real(&self) -> Self {
        assert_eq!(
            self.dtype,
            DType::Complex,
            "real() can only be applied to Complex tensors"
        );
        Self::new(
            DType::F32,
            GraphOp::Real {},
            vec![self.clone()],
            self.view.clone(),
        )
    }

    /// 複素数テンソルから虚部を取り出す
    ///
    /// # 例
    /// ```
    /// use harp::graph::{Graph, GraphNode, DType};
    ///
    /// let mut graph = Graph::new();
    /// let z = graph.input("z", DType::Complex, [4]);
    /// let im = z.imag();  // 虚部（F32）
    /// assert_eq!(im.dtype, DType::F32);
    /// ```
    ///
    /// # パニック
    /// 入力がComplex型でない場合にパニックします。
    pub fn imag(&self) -> Self {
        assert_eq!(
            self.dtype,
            DType::Complex,
            "imag() can only be applied to Complex tensors"
        );
        Self::new(
            DType::F32,
            GraphOp::Imag {},
            vec![self.clone()],
            self.view.clone(),
        )
    }

    /// 実部と虚部のテンソルから複素数テンソルを構築する
    ///
    /// # 引数
    /// - `real`: 実部（F32テンソル）
    /// - `imag`: 虚部（F32テンソル）- realと同じshapeである必要がある
    ///
    /// # 例
    /// ```
    /// use harp::graph::{Graph, GraphNode, DType};
    ///
    /// let mut graph = Graph::new();
    /// let re = graph.input("re", DType::F32, [4]);
    /// let im = graph.input("im", DType::F32, [4]);
    /// let z = GraphNode::complex_from_parts(re, im);
    /// assert_eq!(z.dtype, DType::Complex);
    /// ```
    ///
    /// # パニック
    /// - realまたはimagがF32型でない場合
    /// - realとimagのshapeが一致しない場合
    pub fn complex_from_parts(real: Self, imag: Self) -> Self {
        assert_eq!(
            real.dtype,
            DType::F32,
            "real part must be F32, got {:?}",
            real.dtype
        );
        assert_eq!(
            imag.dtype,
            DType::F32,
            "imag part must be F32, got {:?}",
            imag.dtype
        );
        assert_eq!(
            real.view.shape(),
            imag.view.shape(),
            "real and imag must have the same shape: {:?} vs {:?}",
            real.view.shape(),
            imag.view.shape()
        );
        Self::new(
            DType::Complex,
            GraphOp::ComplexFromParts {},
            vec![real.clone(), imag],
            real.view.clone(),
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
}

// .0 のように書かなくても内部のデータを読み取れるようにする
impl Deref for GraphNode {
    type Target = GraphNodeData;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

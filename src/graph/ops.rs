use crate::ast::Literal;
use crate::graph::shape::View;
use crate::graph::{CumulativeStrategy, DType, GraphNode, ReduceStrategy};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

#[derive(Debug, Clone)]
pub enum GraphOp {
    Input,          // 入力ノード
    Const(Literal), // 定数ノード, shape=[], ndim=0のスカラーを初期化する。
    ComplexConst {
        re: f32,
        im: f32,
    }, // 複素数定数ノード (real, imaginary)
    View(View),     // Viewを変更する
    Contiguous,     // Viewに従って要素を並べ直す。
    Elementwise {
        op: ElementwiseOp,
    }, // 要素ごとに演算を行う
    Reduce {
        op: ReduceOp,
        axis: usize,
        reduce_strategy: Option<ReduceStrategy>,
    }, // 縮約
    Cumulative {
        op: CumulativeOp,
        axis: usize,
        cumulative_strategy: Option<CumulativeStrategy>,
    }, // 累積
    // 融合演算
    // expr内のWildcard("0"), Wildcard("1")等がsrc[0], src[1]に対応
    FusedElementwise {
        expr: crate::ast::AstNode,
    }, // 複数のelementwise演算を融合
    FusedElementwiseReduce {
        expr: crate::ast::AstNode,
        reduce_op: ReduceOp,
        axis: usize,
        reduce_strategy: Option<ReduceStrategy>,
    }, // elementwise -> reduce パターンを融合
    FusedElementwiseCumulative {
        expr: crate::ast::AstNode,
        cumulative_op: CumulativeOp,
        axis: usize,
        cumulative_strategy: Option<CumulativeStrategy>,
    }, // elementwise -> cumulative パターンを融合
    FusedReduce {
        ops: Vec<ReduceOp>,
        axis: usize,
        reduce_strategy: Option<ReduceStrategy>,
    }, // 複数のreduce演算を融合（同じ軸）
    Pad {
        padding: Vec<(usize, usize)>, // 各軸の(前, 後)パディング量
        value: f32,                   // パディング値
    }, // テンソルをパディングで拡張
    Slice {
        ranges: Vec<(usize, usize)>, // 各軸の(start, end)範囲（endは含まない）
    }, // テンソルの一部を切り出し
    Concat {
        axis: usize, // 結合する軸
    }, // 複数のテンソルを指定軸で結合（torch.catに相当）
    Fold {
        output_size: Vec<usize>, // 出力サイズ (unfold前のサイズ)
        kernel_size: Vec<usize>, // カーネルサイズ
        stride: Vec<usize>,      // ストライド
        dilation: Vec<usize>,    // 膨張率
        groups: usize,           // グループ数
    }, // unfoldの逆操作（col2im）
    /// 一様乱数でテンソルを初期化 [0, 1)
    Rand,
    /// 連番テンソルを生成 [0, 1, 2, ..., n-1]
    Arange,
    /// 型変換（キャスト）
    Cast {
        target_dtype: DType,
    },
    /// 複素数テンソルから実部を取り出す
    /// 入力: Complex tensor, 出力: F32 tensor (same shape)
    Real,
    /// 複素数テンソルから虚部を取り出す
    /// 入力: Complex tensor, 出力: F32 tensor (same shape)
    Imag,
    /// 実部と虚部のF32テンソルから複素数テンソルを構築する
    /// 入力: 2つのF32 tensor (real, imag), 出力: Complex tensor (same shape)
    ComplexFromParts,
    /// カスタムAST演算
    ///
    /// `AstNode::Function`または`AstNode::Program`を保持し、lowering時にほぼパススルーで使用される。
    /// - 単一カーネル: `AstNode::Function`を保持
    /// - 複数カーネル（融合後）: `AstNode::Program`を保持（複数のFunctionとmain関数を含む）
    ///
    /// 関数内ではプレースホルダー変数を使用（[`custom_placeholders`]参照）。
    ///
    /// # プレースホルダー変数
    /// - `input0`, `input1`, ... : 入力バッファへのポインタ
    /// - `output` : 出力バッファへのポインタ
    /// - `shape0`, `shape1`, ... : 各軸のサイズ
    /// - `ridx0`, `ridx1`, ... : ループインデックス変数
    ///
    /// # Example
    /// ```
    /// use harp::prelude::*;
    /// use harp::ast::{AstNode, DType as AstDType, Scope, helper::*};
    /// use harp::graph::ops::custom_placeholders as ph;
    ///
    /// let mut graph = Graph::new();
    /// let x = graph.input("x", DType::F32, vec![10]);
    ///
    /// // x^2 を計算するカスタム関数を構築
    /// let func = function(
    ///     None::<String>,  // 名前はlowering時に設定
    ///     vec![],  // パラメータはlowering時に設定
    ///     AstDType::Tuple(vec![]),
    ///     range(
    ///         &ph::ridx(0),
    ///         const_int(0),
    ///         const_int(1),
    ///         var(&ph::shape(0)),
    ///         block(vec![
    ///             store(
    ///                 var(ph::OUTPUT),
    ///                 var(&ph::ridx(0)),
    ///                 load(var(&ph::input(0)), var(&ph::ridx(0)), AstDType::F32)
    ///                   * load(var(&ph::input(0)), var(&ph::ridx(0)), AstDType::F32),
    ///             ),
    ///         ], Scope::new()),
    ///     ),
    /// );
    /// let y = x.custom_function(func);
    /// ```
    Custom {
        /// AstNode::Function または AstNode::Program
        ast: crate::ast::AstNode,
    },
}

#[derive(Debug, Clone)]
pub enum ElementwiseOp {
    Add,
    Mul,
    Max,
    Rem,
    Idiv,
    Neg,
    Recip,
    Log2, // 底が2の対数
    Exp2, // 2の累乗
    Sin,  // 正弦
    Sqrt, // 平方根
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,  // 合計
    Prod, // 積
    Max,  // 最大値
}

/// 累積演算の種類
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CumulativeOp {
    Sum,  // 累積和（cumsum）
    Prod, // 累積積（cumprod）
}

/// カスタム関数で使用するプレースホルダー変数の規約
///
/// Custom関数内では以下のプレースホルダー変数を使用します：
/// - `input0`, `input1`, ... : 入力バッファへのポインタ (src[0], src[1], ...)
/// - `output` : 出力バッファへのポインタ
/// - `shape0`, `shape1`, ... : 各軸のサイズ (node.view.shape()[0], [1], ...)
/// - `ridx0`, `ridx1`, ... : ループインデックス変数
///
/// これらの変数は lowering 時に実際のパラメータにマッピングされます。
pub mod custom_placeholders {
    /// 入力バッファのプレースホルダー名を生成
    pub fn input(index: usize) -> String {
        format!("input{}", index)
    }

    /// 出力バッファのプレースホルダー名
    pub const OUTPUT: &str = "output";

    /// Shape変数のプレースホルダー名を生成
    pub fn shape(axis: usize) -> String {
        format!("shape{}", axis)
    }

    /// ループインデックス変数のプレースホルダー名を生成
    pub fn ridx(axis: usize) -> String {
        format!("ridx{}", axis)
    }
}

// DTypeの推論：両方が同じならそれを使う、片方がUnknownなら他方を使う
pub fn infer_dtype(dtype1: &DType, dtype2: &DType) -> DType {
    match (dtype1, dtype2) {
        (DType::Unknown, d) | (d, DType::Unknown) => d.clone(),
        (d1, d2) if std::mem::discriminant(d1) == std::mem::discriminant(d2) => d1.clone(),
        _ => DType::Unknown, // 異なる型の場合はUnknown
    }
}

// Viewの推論：完全に同じshapeのみを許可
// shapeの変更は明示的に行う必要がある（expand, unsqueezeなどを使用）
// ただしスカラー（ndim=0）は任意のテンソルにブロードキャスト可能
pub fn infer_view(view1: &View, view2: &View) -> View {
    let shape1 = view1.shape();
    let shape2 = view2.shape();

    // 両方が同じshapeの場合のみ許可
    if shape1 == shape2 {
        return View::contiguous(shape1.to_vec());
    }

    // スカラー（ndim=0）は任意のテンソルにブロードキャスト可能
    if shape1.is_empty() {
        return View::contiguous(shape2.to_vec());
    }
    if shape2.is_empty() {
        return View::contiguous(shape1.to_vec());
    }

    // 異なるshapeの場合はエラー
    panic!(
        "Shape mismatch: {:?} and {:?}. Shape transformations must be explicit (use expand, unsqueeze, etc.)",
        shape1, shape2
    );
}

// 演算子のトレイトを実装

/// 二項演算のelementwise演算子を実装するマクロ
macro_rules! impl_binary_elementwise_op {
    ($trait:ident, $method:ident, $op:ident) => {
        impl<T: Into<GraphNode>> $trait<T> for GraphNode {
            type Output = Self;
            fn $method(self, rhs: T) -> Self::Output {
                let rhs = rhs.into();
                let dtype = infer_dtype(&self.dtype, &rhs.dtype);
                let view = infer_view(&self.view, &rhs.view);
                GraphNode::new(
                    dtype,
                    GraphOp::Elementwise {
                        op: ElementwiseOp::$op,
                    },
                    vec![self, rhs],
                    view,
                )
            }
        }
    };
}

// マクロを使って演算子を実装
impl_binary_elementwise_op!(Add, add, Add);
impl_binary_elementwise_op!(Mul, mul, Mul);
impl_binary_elementwise_op!(Rem, rem, Rem);

// Neg: -a (単項演算なので別途実装)
impl Neg for GraphNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        let dtype = self.dtype.clone();
        let view = self.view.clone();
        GraphNode::new(
            dtype,
            GraphOp::Elementwise {
                op: ElementwiseOp::Neg,
            },
            vec![self],
            view,
        )
    }
}

// Sub: a - b = a + (-b)
impl<T: Into<GraphNode>> Sub<T> for GraphNode {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: T) -> Self::Output {
        // Subtraction is implemented as addition of negation: a - b = a + (-b)
        self + (-rhs.into())
    }
}

// Div: a / b = a * recip(b)
impl<T: Into<GraphNode>> Div<T> for GraphNode {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: T) -> Self::Output {
        // Division is implemented as multiplication by reciprocal: a / b = a * recip(b)
        self * recip(rhs.into())
    }
}

// Into<GraphNode> implementations for numeric types
impl From<bool> for GraphNode {
    fn from(value: bool) -> Self {
        GraphNode::constant(value)
    }
}

impl From<f32> for GraphNode {
    fn from(value: f32) -> Self {
        GraphNode::constant(value)
    }
}

impl From<isize> for GraphNode {
    fn from(value: isize) -> Self {
        GraphNode::constant(value)
    }
}

impl From<i32> for GraphNode {
    fn from(value: i32) -> Self {
        GraphNode::constant(value as isize)
    }
}

impl From<i64> for GraphNode {
    fn from(value: i64) -> Self {
        GraphNode::constant(value as isize)
    }
}

impl From<&GraphNode> for GraphNode {
    fn from(value: &GraphNode) -> Self {
        value.clone()
    }
}

// Complex number from tuple (re, im)
impl From<(f32, f32)> for GraphNode {
    fn from(value: (f32, f32)) -> Self {
        GraphNode::complex_constant(value.0, value.1)
    }
}

// Reverse operations: numeric op GraphNode
macro_rules! impl_reverse_ops {
    ($ty:ty) => {
        impl Add<GraphNode> for $ty {
            type Output = GraphNode;
            fn add(self, rhs: GraphNode) -> GraphNode {
                GraphNode::from(self) + rhs
            }
        }

        impl Sub<GraphNode> for $ty {
            type Output = GraphNode;
            fn sub(self, rhs: GraphNode) -> GraphNode {
                GraphNode::from(self) - rhs
            }
        }

        impl Mul<GraphNode> for $ty {
            type Output = GraphNode;
            fn mul(self, rhs: GraphNode) -> GraphNode {
                GraphNode::from(self) * rhs
            }
        }

        impl Div<GraphNode> for $ty {
            type Output = GraphNode;
            fn div(self, rhs: GraphNode) -> GraphNode {
                GraphNode::from(self) / rhs
            }
        }

        impl Rem<GraphNode> for $ty {
            type Output = GraphNode;
            fn rem(self, rhs: GraphNode) -> GraphNode {
                GraphNode::from(self) % rhs
            }
        }
    };
}

impl_reverse_ops!(f32);
impl_reverse_ops!(isize);
impl_reverse_ops!(i32);
impl_reverse_ops!(i64);

// Reference-based operator overloading to avoid cloning
// &GraphNode op T
impl<T: Into<GraphNode>> Add<T> for &GraphNode {
    type Output = GraphNode;
    fn add(self, rhs: T) -> GraphNode {
        self.clone() + rhs
    }
}

impl<T: Into<GraphNode>> Sub<T> for &GraphNode {
    type Output = GraphNode;
    fn sub(self, rhs: T) -> GraphNode {
        self.clone() - rhs
    }
}

impl<T: Into<GraphNode>> Mul<T> for &GraphNode {
    type Output = GraphNode;
    fn mul(self, rhs: T) -> GraphNode {
        self.clone() * rhs
    }
}

impl<T: Into<GraphNode>> Div<T> for &GraphNode {
    type Output = GraphNode;
    fn div(self, rhs: T) -> GraphNode {
        self.clone() / rhs
    }
}

impl<T: Into<GraphNode>> Rem<T> for &GraphNode {
    type Output = GraphNode;
    fn rem(self, rhs: T) -> GraphNode {
        self.clone() % rhs
    }
}

impl Neg for &GraphNode {
    type Output = GraphNode;
    fn neg(self) -> GraphNode {
        -self.clone()
    }
}

// numeric op &GraphNode
macro_rules! impl_reverse_ops_for_ref {
    ($ty:ty) => {
        impl Add<&GraphNode> for $ty {
            type Output = GraphNode;
            fn add(self, rhs: &GraphNode) -> GraphNode {
                GraphNode::from(self) + rhs.clone()
            }
        }

        impl Sub<&GraphNode> for $ty {
            type Output = GraphNode;
            fn sub(self, rhs: &GraphNode) -> GraphNode {
                GraphNode::from(self) - rhs.clone()
            }
        }

        impl Mul<&GraphNode> for $ty {
            type Output = GraphNode;
            fn mul(self, rhs: &GraphNode) -> GraphNode {
                GraphNode::from(self) * rhs.clone()
            }
        }

        impl Div<&GraphNode> for $ty {
            type Output = GraphNode;
            fn div(self, rhs: &GraphNode) -> GraphNode {
                GraphNode::from(self) / rhs.clone()
            }
        }

        impl Rem<&GraphNode> for $ty {
            type Output = GraphNode;
            fn rem(self, rhs: &GraphNode) -> GraphNode {
                GraphNode::from(self) % rhs.clone()
            }
        }
    };
}

impl_reverse_ops_for_ref!(f32);
impl_reverse_ops_for_ref!(isize);
impl_reverse_ops_for_ref!(i32);
impl_reverse_ops_for_ref!(i64);

// ヘルパー関数: Recip (逆数)
pub fn recip(node: GraphNode) -> GraphNode {
    let view = node.view.clone();
    let dtype = node.dtype.clone();
    GraphNode::new(
        dtype,
        GraphOp::Elementwise {
            op: ElementwiseOp::Recip,
        },
        vec![node],
        view,
    )
}

// ヘルパー関数: Max
pub fn max(lhs: GraphNode, rhs: GraphNode) -> GraphNode {
    let dtype = infer_dtype(&lhs.dtype, &rhs.dtype);
    let view = infer_view(&lhs.view, &rhs.view);
    GraphNode::new(
        dtype,
        GraphOp::Elementwise {
            op: ElementwiseOp::Max,
        },
        vec![lhs, rhs],
        view,
    )
}

// ヘルパー関数: Reduce（汎用）
pub fn reduce(node: GraphNode, op: ReduceOp, axis: usize) -> GraphNode {
    let dtype = node.dtype.clone();
    let view = node.view.clone();

    // 指定された軸を縮約した新しいViewを作成
    let mut new_shape = view.shape().to_vec();
    if axis >= new_shape.len() {
        panic!(
            "Reduce: axis {} is out of bounds for shape {:?}",
            axis, new_shape
        );
    }
    new_shape.remove(axis);
    let reduced_view = View::contiguous(new_shape);

    GraphNode::new(
        dtype,
        GraphOp::Reduce {
            op,
            axis,
            reduce_strategy: None,
        },
        vec![node],
        reduced_view,
    )
}

// ヘルパー関数: Reduce Sum（指定軸の合計）
pub fn reduce_sum(node: GraphNode, axis: usize) -> GraphNode {
    reduce(node, ReduceOp::Sum, axis)
}

// ヘルパー関数: Reduce Mul（指定軸の積）
pub fn reduce_mul(node: GraphNode, axis: usize) -> GraphNode {
    reduce(node, ReduceOp::Prod, axis)
}

// ヘルパー関数: Reduce Max（指定軸の最大値）
pub fn reduce_max(node: GraphNode, axis: usize) -> GraphNode {
    reduce(node, ReduceOp::Max, axis)
}

// ヘルパー関数: Cumulative（汎用累積演算）
pub fn cumulative(node: GraphNode, op: CumulativeOp, axis: usize) -> GraphNode {
    let dtype = node.dtype.clone();
    let view = node.view.clone();

    // 累積演算はshapeを変更しない（軸を保持）
    if axis >= view.shape().len() {
        panic!(
            "Cumulative: axis {} is out of bounds for shape {:?}",
            axis,
            view.shape()
        );
    }

    GraphNode::new(
        dtype,
        GraphOp::Cumulative {
            op,
            axis,
            cumulative_strategy: None,
        },
        vec![node],
        view,
    )
}

// ヘルパー関数: Cumulative Sum（累積和）
pub fn cumsum(node: GraphNode, axis: usize) -> GraphNode {
    cumulative(node, CumulativeOp::Sum, axis)
}

// ヘルパー関数: Cumulative Prod（累積積）
pub fn cumprod(node: GraphNode, axis: usize) -> GraphNode {
    cumulative(node, CumulativeOp::Prod, axis)
}

// === 融合ノード生成ヘルパー関数 ===

/// 複数のelementwise演算を融合したノードを作成
///
/// expr内のWildcard("0"), Wildcard("1")等がsrc[0], src[1]に対応します。
///
/// # 例
/// ```no_run
/// use harp::prelude::*;
/// use harp::ast::helper::wildcard;
/// let mut graph = Graph::new();
/// let a = graph.input("a", DType::F32, [4]);
/// let b = graph.input("b", DType::F32, [4]);
/// let c = graph.input("c", DType::F32, [4]);
///
/// // (a + b) * c を融合して生成
/// let expr = (wildcard("0") + wildcard("1")) * wildcard("2");
/// let result = fused_elementwise(vec![a, b, c], expr);
/// ```
pub fn fused_elementwise(inputs: Vec<GraphNode>, expr: crate::ast::AstNode) -> GraphNode {
    // DTypeとViewは最初の入力から継承（全入力が同じshapeであることを前提）
    if inputs.is_empty() {
        panic!("fused_elementwise requires at least one input");
    }

    let dtype = inputs[0].dtype.clone();
    let view = inputs[0].view.clone();

    GraphNode::new(dtype, GraphOp::FusedElementwise { expr }, inputs, view)
}

/// elementwise演算とそれに続くreduce演算を融合したノードを作成
///
/// expr内のWildcard("0"), Wildcard("1")等がsrc[0], src[1]に対応します。
///
/// # 例
/// ```no_run
/// use harp::prelude::*;
/// use harp::ast::helper::wildcard;
/// let mut graph = Graph::new();
/// let a = graph.input("a", DType::F32, [4, 8]);
/// let b = graph.input("b", DType::F32, [4, 8]);
///
/// // reduce_sum(a * b, axis=1) を融合して生成
/// let expr = wildcard("0") * wildcard("1");
/// let result = fused_elementwise_reduce(vec![a, b], expr, ReduceOp::Sum, 1);
/// ```
pub fn fused_elementwise_reduce(
    inputs: Vec<GraphNode>,
    expr: crate::ast::AstNode,
    reduce_op: ReduceOp,
    axis: usize,
) -> GraphNode {
    if inputs.is_empty() {
        panic!("fused_elementwise_reduce requires at least one input");
    }

    let dtype = inputs[0].dtype.clone();
    let view = inputs[0].view.clone();

    // 指定された軸を縮約した新しいViewを作成
    let mut new_shape = view.shape().to_vec();
    if axis >= new_shape.len() {
        panic!(
            "fused_elementwise_reduce: axis {} is out of bounds for shape {:?}",
            axis, new_shape
        );
    }
    new_shape.remove(axis);
    let reduced_view = View::contiguous(new_shape);

    GraphNode::new(
        dtype,
        GraphOp::FusedElementwiseReduce {
            expr,
            reduce_op,
            axis,
            reduce_strategy: None,
        },
        inputs,
        reduced_view,
    )
}

/// 複数のelementwise演算の後に累積演算を行う融合ノードを作成
///
/// expr内のWildcard("0"), Wildcard("1")等がsrc[0], src[1]に対応します。
///
/// # 例
/// ```no_run
/// use harp::prelude::*;
/// use harp::ast::helper::wildcard;
/// use harp::graph::ops::{CumulativeOp, fused_elementwise_cumulative};
/// let mut graph = Graph::new();
/// let x = graph.input("x", DType::F32, [4, 8]);
///
/// // 入力を二乗してから累積和: cumsum(x^2)
/// let expr = wildcard("0") * wildcard("0");
/// let squared_cumsum = fused_elementwise_cumulative(vec![x], expr, CumulativeOp::Sum, 1);
/// ```
pub fn fused_elementwise_cumulative(
    inputs: Vec<GraphNode>,
    expr: crate::ast::AstNode,
    cumulative_op: CumulativeOp,
    axis: usize,
) -> GraphNode {
    if inputs.is_empty() {
        panic!("fused_elementwise_cumulative requires at least one input");
    }

    let dtype = inputs[0].dtype.clone();
    let view = inputs[0].view.clone();

    // 累積演算はshapeを変えないので、同じViewを使用
    if axis >= view.shape().len() {
        panic!(
            "fused_elementwise_cumulative: axis {} is out of bounds for shape {:?}",
            axis,
            view.shape()
        );
    }

    GraphNode::new(
        dtype,
        GraphOp::FusedElementwiseCumulative {
            expr,
            cumulative_op,
            axis,
            cumulative_strategy: None,
        },
        inputs,
        view,
    )
}

/// 複数のテンソルを指定した軸で結合する
///
/// # 引数
/// - `inputs`: 結合するテンソルのベクター（2つ以上）
/// - `axis`: 結合する軸
///
/// # パニック
/// - inputsが空の場合
/// - 結合軸以外の次元が一致しない場合
/// - axisがテンソルの次元数以上の場合
///
/// # 例
/// ```no_run
/// use harp::prelude::*;
/// use harp::graph::ops::concat;
///
/// let mut graph = Graph::new();
/// let a = graph.input("a", DType::F32, [2, 3]);
/// let b = graph.input("b", DType::F32, [2, 5]);
///
/// // axis=1で結合: [2, 3] + [2, 5] => [2, 8]
/// let c = concat(vec![a, b], 1);
/// ```
pub fn concat(inputs: Vec<GraphNode>, axis: usize) -> GraphNode {
    if inputs.is_empty() {
        panic!("concat requires at least one input");
    }

    if inputs.len() == 1 {
        return inputs.into_iter().next().unwrap();
    }

    let first = &inputs[0];
    let first_shape = first.view.shape();
    let ndim = first_shape.len();

    if axis >= ndim {
        panic!(
            "concat: axis {} is out of bounds for tensor with {} dimensions",
            axis, ndim
        );
    }

    // 結合軸以外の次元が一致することを確認
    for (i, input) in inputs.iter().enumerate().skip(1) {
        let shape = input.view.shape();
        if shape.len() != ndim {
            panic!(
                "concat: all inputs must have the same number of dimensions. \
                 Input 0 has {} dimensions, but input {} has {} dimensions",
                ndim,
                i,
                shape.len()
            );
        }

        for (dim, (s1, s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
            if dim != axis && s1 != s2 {
                panic!(
                    "concat: dimension mismatch at axis {}. \
                     Input 0 has size {:?}, but input {} has size {:?}",
                    dim, s1, i, s2
                );
            }
        }
    }

    // 出力形状を計算：結合軸のサイズは全入力の合計
    let mut output_shape = first_shape.to_vec();
    let mut concat_size = first_shape[axis].clone();
    for input in inputs.iter().skip(1) {
        concat_size += input.view.shape()[axis].clone();
    }
    output_shape[axis] = concat_size;

    // DTypeは最初の入力から継承
    let dtype = first.dtype.clone();
    let output_view = crate::graph::shape::View::contiguous(output_shape);

    GraphNode::new(dtype, GraphOp::Concat { axis }, inputs, output_view)
}

/// 複数のreduce演算を融合したノードを作成
///
/// # 例
/// ```no_run
/// use harp::prelude::*;
/// // 同じ入力に対して sum と max を同時に計算
/// // 注意: 出力は複数のテンソルになるため、現在の設計では未対応
/// // 将来的には tuple 出力として実装予定
/// ```
pub fn fused_reduce(node: GraphNode, ops: Vec<ReduceOp>, axis: usize) -> GraphNode {
    let dtype = node.dtype.clone();
    let view = node.view.clone();

    // 指定された軸を縮約した新しいViewを作成
    let mut new_shape = view.shape().to_vec();
    if axis >= new_shape.len() {
        panic!(
            "fused_reduce: axis {} is out of bounds for shape {:?}",
            axis, new_shape
        );
    }
    new_shape.remove(axis);
    let reduced_view = View::contiguous(new_shape);

    GraphNode::new(
        dtype,
        GraphOp::FusedReduce {
            ops,
            axis,
            reduce_strategy: None,
        },
        vec![node],
        reduced_view,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Graph, View};

    #[test]
    fn test_from_f32() {
        let node = GraphNode::from(2.5f32);
        assert_eq!(node.dtype, DType::F32);
        assert!(node.view.shape().is_empty()); // Scalar

        match &node.op {
            GraphOp::Const(Literal::F32(v)) => assert_eq!(*v, 2.5),
            _ => panic!("Expected F32 constant"),
        }
    }

    #[test]
    fn test_from_isize() {
        let node = GraphNode::from(42isize);
        assert!(node.view.shape().is_empty());

        match &node.op {
            GraphOp::Const(Literal::Int(v)) => assert_eq!(*v, 42),
            _ => panic!("Expected Int constant"),
        }
    }

    #[test]
    fn test_from_i32() {
        let node = GraphNode::from(100i32);
        match &node.op {
            GraphOp::Const(Literal::Int(v)) => assert_eq!(*v, 100),
            _ => panic!("Expected Int constant"),
        }
    }

    #[test]
    fn test_add_with_numeric() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [4]);

        // GraphNode + f32
        let result = x.clone() + 2.0f32;
        assert_eq!(result.view.shape(), &[4.into()]);
        match &result.op {
            GraphOp::Elementwise { op, .. } => {
                assert!(matches!(op, ElementwiseOp::Add));
            }
            _ => panic!("Expected Elementwise Add"),
        }
    }

    #[test]
    fn test_mul_with_numeric() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [8]);

        // GraphNode * isize
        let result = x * 3isize;
        assert_eq!(result.view.shape(), &[8.into()]);
    }

    #[test]
    fn test_sub_with_numeric() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [10]);

        // GraphNode - f32
        let result = x - 1.0f32;
        // Sub is Add(a, Neg(b))
        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Add,
                ..
            } => {}
            _ => panic!("Expected Add (for subtraction)"),
        }
    }

    #[test]
    fn test_div_with_numeric() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [5]);

        // GraphNode / f32
        let result = x / 2.0f32;
        // Div is Mul(a, Recip(b))
        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Mul,
                ..
            } => {}
            _ => panic!("Expected Mul (for division)"),
        }
    }

    #[test]
    fn test_reverse_add() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [4]);

        // f32 + GraphNode
        let result = 2.0f32 + x;
        assert_eq!(result.view.shape(), &[4.into()]);
        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Add,
                ..
            } => {}
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_reverse_mul() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [6]);

        // i32 * GraphNode
        let result = 5i32 * x;
        assert_eq!(result.view.shape(), &[6.into()]);
    }

    #[test]
    fn test_reverse_sub() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [3]);

        // f32 - GraphNode
        let result = 10.0f32 - x;
        assert_eq!(result.view.shape(), &[3.into()]);
    }

    #[test]
    fn test_reverse_div() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [7]);

        // f32 / GraphNode (1 / x)
        let result = 1.0f32 / x;
        assert_eq!(result.view.shape(), &[7.into()]);
    }

    #[test]
    fn test_mixed_operations() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [4]);

        // Complex expression: 2.0 * x + 1.0
        let result = 2.0f32 * x + 1.0f32;
        assert_eq!(result.view.shape(), &[4.into()]);

        // Result should be Add(Mul(...), Const)
        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Add,
                ..
            } => {}
            _ => panic!("Expected Add"),
        }
    }

    #[test]
    fn test_scalar_broadcast() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [2, 3]);

        // Scalar constant should broadcast to [2, 3]
        let result = x + 1.0f32;
        assert_eq!(result.view.shape(), &[2.into(), 3.into()]);
    }

    #[test]
    fn test_infer_view_with_scalar() {
        let scalar_view = View::contiguous(Vec::<isize>::new());
        let tensor_view = View::contiguous(vec![4, 8]);

        // Scalar + Tensor should give Tensor's shape
        let result = infer_view(&scalar_view, &tensor_view);
        assert_eq!(result.shape(), &[4.into(), 8.into()]);

        // Tensor + Scalar should also give Tensor's shape
        let result2 = infer_view(&tensor_view, &scalar_view);
        assert_eq!(result2.shape(), &[4.into(), 8.into()]);
    }

    #[test]
    fn test_reference_add() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [4]);

        // &GraphNode + numeric (no clone needed)
        let result = &x + 2.0f32;
        assert_eq!(result.view.shape(), &[4.into()]);

        // Can still use x
        let result2 = &x * 3.0f32;
        assert_eq!(result2.view.shape(), &[4.into()]);
    }

    #[test]
    fn test_reference_mul() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [5]);

        let result = &x * 2.0f32;
        match &result.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Mul,
                ..
            } => {}
            _ => panic!("Expected Mul"),
        }
    }

    #[test]
    fn test_reference_sub_div() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [3]);

        let _sub_result = &x - 1.0f32;
        let _div_result = &x / 2.0f32;

        // x is still usable
        let _final = &x + 10.0f32;
    }

    #[test]
    fn test_reverse_reference_ops() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [4]);

        // numeric op &GraphNode
        let result = 2.0f32 * &x;
        assert_eq!(result.view.shape(), &[4.into()]);

        let result2 = 10.0f32 - &x;
        assert_eq!(result2.view.shape(), &[4.into()]);

        let result3 = 1.0f32 / &x;
        assert_eq!(result3.view.shape(), &[4.into()]);
    }

    #[test]
    fn test_complex_expression_with_references() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [8]);

        // Complex expression without clone
        let result = 2.0f32 * &x + 1.0f32;
        assert_eq!(result.view.shape(), &[8.into()]);

        // x * x without consuming x
        let squared = &x * &x;
        assert_eq!(squared.view.shape(), &[8.into()]);

        // Still can use x
        let _final = &x + 100.0f32;
    }

    #[test]
    fn test_neg_reference() {
        let mut graph = Graph::new();
        let x = graph.input("x", DType::F32, [4]);

        let neg_x = -&x;
        match &neg_x.op {
            GraphOp::Elementwise {
                op: ElementwiseOp::Neg,
                ..
            } => {}
            _ => panic!("Expected Neg"),
        }

        // x is still usable
        let _sum = &x + 1.0f32;
    }

    #[test]
    fn test_complex_constant() {
        let node = GraphNode::complex_constant(1.0, 2.0);
        assert_eq!(node.dtype, DType::Complex);
        assert!(node.view.shape().is_empty()); // Scalar

        match &node.op {
            GraphOp::ComplexConst { re, im } => {
                assert_eq!(*re, 1.0);
                assert_eq!(*im, 2.0);
            }
            _ => panic!("Expected ComplexConst"),
        }
    }

    #[test]
    fn test_complex_from_tuple() {
        let node: GraphNode = (3.0f32, 4.0f32).into();
        assert_eq!(node.dtype, DType::Complex);
        assert!(node.view.shape().is_empty());

        match &node.op {
            GraphOp::ComplexConst { re, im } => {
                assert_eq!(*re, 3.0);
                assert_eq!(*im, 4.0);
            }
            _ => panic!("Expected ComplexConst"),
        }
    }

    #[test]
    fn test_complex_constant_from() {
        let node = GraphNode::complex_constant_from((5.0, -1.0));
        assert_eq!(node.dtype, DType::Complex);

        match &node.op {
            GraphOp::ComplexConst { re, im } => {
                assert_eq!(*re, 5.0);
                assert_eq!(*im, -1.0);
            }
            _ => panic!("Expected ComplexConst"),
        }
    }
}

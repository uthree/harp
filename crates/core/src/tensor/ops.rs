//! TensorOp - Tensor演算の定義
//!
//! 入力テンソルをTensorOp内に埋め込む設計。
//! MapReduce演算で全てのElementwise/Reduce演算を統一的に表現。

use std::sync::Arc;

use crate::ast::{AstNode, DType, Literal};

use super::TensorInner;

/// 入力テンソルへの参照
///
/// 計算グラフ内での入力参照に使用。
/// `Arc<TensorInner>`の型エイリアス。
pub type InputRef = Arc<TensorInner>;

/// Tensor演算の種類
///
/// 全ての演算が入力テンソルを内包する。
/// 演算はselfを消費（move）する設計。
#[derive(Clone)]
pub enum TensorOp {
    // ============================================================
    // ソース演算（入力なし）
    // ============================================================
    /// 入力バッファ（外部から与えられるデータ）
    Buffer { name: String },

    /// スカラー定数
    Const(Literal),

    /// 定数値でテンソル全体を埋める（zeros, ones, full）
    ConstFill(Literal),

    /// 一様乱数 [0, 1)
    Rand,

    /// 連番テンソル [0, 1, 2, ..., n-1]
    Arange,

    /// contiguous()実行後の状態
    /// バッファにデータが格納されている。
    Executed,

    // ============================================================
    // 単項演算（1入力）
    // ============================================================
    /// View変更（メモリコピーなし）
    View { input: InputRef },

    /// Viewに従って要素を並べ直す（実体化）
    Contiguous { input: InputRef },

    /// 分岐点を作成（バッファコピー）
    Clone { input: InputRef },

    // ============================================================
    // 統一計算演算（Compute）
    // ============================================================
    /// 統一された計算演算
    ///
    /// - Elementwise: reduce_op = None, axes = []
    /// - Reduce: expr = Wildcard("0"), reduce_op = Some(...), axes = [...]
    /// - Fused: 任意のexpr + reduce_op
    MapReduce {
        /// 入力テンソル群（Wildcard("0"), Wildcard("1"), ... に対応）
        inputs: Vec<InputRef>,
        /// 計算式（AstNode）
        expr: AstNode,
        /// 縮約演算（オプション）
        reduce_op: Option<ReduceOp>,
        /// 縮約軸（空の場合はElementwise）
        axes: Vec<usize>,
        /// 次元を維持するか
        keepdim: bool,
    },

    // ============================================================
    // 構造演算
    // ============================================================
    /// 結合
    Concat { inputs: Vec<InputRef>, axis: usize },

    /// Scatter-Add演算（AtomicAddを使用）
    ///
    /// GatherBackwardの実装に使用。
    /// target[...][index[...]][...] += src[...]
    ScatterAdd {
        /// 累積先テンソル（初期値、通常はzeros）
        target: InputRef,
        /// インデックステンソル
        index: InputRef,
        /// ソーステンソル
        src: InputRef,
        /// 次元
        dim: usize,
    },
}

/// Elementwise演算の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementwiseOp {
    // 二項演算
    Add,
    Mul,
    Max,
    Rem,
    Idiv,

    // 単項演算
    Neg,
    Recip,
    Log2,
    Exp2,
    Sin,
    Sqrt,
    Floor,

    // ビット演算（二項）
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,

    // ビット演算（単項）
    BitNot,
}

/// 縮約演算の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
}

/// パディング値の種類
///
/// 各リダクション演算の単位元に対応:
/// - Zero: 0.0 (Sumの単位元)
/// - One: 1.0 (Prodの単位元)
/// - NegInf: -∞ (Maxの単位元)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadValue {
    /// 0.0 - Sum演算の単位元
    Zero,
    /// 1.0 - Prod演算の単位元
    One,
    /// 負の無限大 - Max演算の単位元
    NegInf,
}

impl PadValue {
    /// パディング値を浮動小数点数として取得
    pub fn as_f32(&self) -> f32 {
        match self {
            PadValue::Zero => 0.0,
            PadValue::One => 1.0,
            PadValue::NegInf => f32::NEG_INFINITY,
        }
    }

    /// パディング値を倍精度浮動小数点数として取得
    pub fn as_f64(&self) -> f64 {
        match self {
            PadValue::Zero => 0.0,
            PadValue::One => 1.0,
            PadValue::NegInf => f64::NEG_INFINITY,
        }
    }

    /// ReduceOpに対応するパディング値を取得
    pub fn for_reduce_op(op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => PadValue::Zero,
            ReduceOp::Prod => PadValue::One,
            ReduceOp::Max => PadValue::NegInf,
        }
    }
}

impl TensorOp {
    /// 型変換演算を作成（MapReduceとして統一）
    ///
    /// Cast演算をElementwise MapReduceとして表現することで、
    /// 他のelementwise演算と融合可能になる。
    pub fn cast(input: InputRef, target_dtype: DType) -> Self {
        use crate::ast::helper::*;
        Self::MapReduce {
            inputs: vec![input],
            expr: cast(wildcard("0"), target_dtype),
            reduce_op: None,
            axes: vec![],
            keepdim: false,
        }
    }

    /// Elementwise演算を作成
    pub fn elementwise(inputs: Vec<InputRef>, expr: AstNode) -> Self {
        Self::MapReduce {
            inputs,
            expr,
            reduce_op: None,
            axes: vec![],
            keepdim: false,
        }
    }

    /// Reduce演算を作成
    pub fn reduce(input: InputRef, op: ReduceOp, axes: Vec<usize>, keepdim: bool) -> Self {
        Self::MapReduce {
            inputs: vec![input],
            expr: AstNode::Wildcard("0".to_string()),
            reduce_op: Some(op),
            axes,
            keepdim,
        }
    }

    /// 融合演算を作成
    pub fn fused(
        inputs: Vec<InputRef>,
        expr: AstNode,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Self {
        Self::MapReduce {
            inputs,
            expr,
            reduce_op: Some(reduce_op),
            axes,
            keepdim,
        }
    }

    /// この演算がElementwiseかどうか
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            TensorOp::MapReduce {
                reduce_op: None,
                axes,
                ..
            } if axes.is_empty()
        )
    }

    /// この演算がReduceかどうか
    pub fn is_reduce(&self) -> bool {
        matches!(
            self,
            TensorOp::MapReduce {
                reduce_op: Some(_),
                ..
            }
        )
    }

    /// この演算が実行済みかどうか
    pub fn is_executed(&self) -> bool {
        matches!(self, TensorOp::Executed)
    }

    /// この演算がView操作かどうか
    pub fn is_view(&self) -> bool {
        matches!(self, TensorOp::View { .. })
    }

    /// この演算がContiguousかどうか
    pub fn is_contiguous(&self) -> bool {
        matches!(self, TensorOp::Contiguous { .. })
    }

    /// 入力テンソルを取得
    pub fn inputs(&self) -> Vec<&InputRef> {
        match self {
            TensorOp::Buffer { .. }
            | TensorOp::Const(_)
            | TensorOp::ConstFill(_)
            | TensorOp::Rand
            | TensorOp::Arange
            | TensorOp::Executed => vec![],

            TensorOp::View { input }
            | TensorOp::Contiguous { input }
            | TensorOp::Clone { input } => vec![input],

            TensorOp::MapReduce { inputs, .. } | TensorOp::Concat { inputs, .. } => {
                inputs.iter().collect()
            }

            TensorOp::ScatterAdd {
                target, index, src, ..
            } => vec![target, index, src],
        }
    }

    /// 演算の名前を取得（デバッグ用）
    pub fn name(&self) -> &'static str {
        match self {
            TensorOp::Buffer { .. } => "Buffer",
            TensorOp::Const(_) => "Const",
            TensorOp::ConstFill(_) => "ConstFill",
            TensorOp::Rand => "Rand",
            TensorOp::Arange => "Arange",
            TensorOp::Executed => "Executed",
            TensorOp::View { .. } => "View",
            TensorOp::Contiguous { .. } => "Contiguous",
            TensorOp::Clone { .. } => "Clone",
            TensorOp::MapReduce {
                reduce_op: None, ..
            } => "MapReduce(Elementwise)",
            TensorOp::MapReduce {
                reduce_op: Some(_), ..
            } => "MapReduce(Reduce)",
            TensorOp::Concat { .. } => "Concat",
            TensorOp::ScatterAdd { .. } => "ScatterAdd",
        }
    }
}

impl std::fmt::Debug for TensorOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorOp::Buffer { name } => write!(f, "Buffer {{ name: {:?} }}", name),
            TensorOp::Const(lit) => write!(f, "Const({:?})", lit),
            TensorOp::ConstFill(lit) => write!(f, "ConstFill({:?})", lit),
            TensorOp::Rand => write!(f, "Rand"),
            TensorOp::Arange => write!(f, "Arange"),
            TensorOp::Executed => write!(f, "Executed"),
            TensorOp::View { .. } => write!(f, "View"),
            TensorOp::Contiguous { .. } => write!(f, "Contiguous"),
            TensorOp::Clone { .. } => write!(f, "Clone"),
            TensorOp::MapReduce {
                expr,
                reduce_op,
                axes,
                keepdim,
                ..
            } => {
                write!(
                    f,
                    "MapReduce {{ expr: {:?}, reduce_op: {:?}, axes: {:?}, keepdim: {} }}",
                    expr, reduce_op, axes, keepdim
                )
            }
            TensorOp::Concat { axis, .. } => write!(f, "Concat {{ axis: {} }}", axis),
            TensorOp::ScatterAdd { dim, .. } => write!(f, "ScatterAdd {{ dim: {} }}", dim),
        }
    }
}

impl ElementwiseOp {
    /// この演算が二項演算かどうか
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Mul
                | Self::Max
                | Self::Rem
                | Self::Idiv
                | Self::BitAnd
                | Self::BitOr
                | Self::BitXor
                | Self::Shl
                | Self::Shr
        )
    }

    /// この演算が単項演算かどうか
    pub fn is_unary(&self) -> bool {
        !self.is_binary()
    }

    /// 演算の名前を取得
    pub fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Max => "max",
            Self::Rem => "rem",
            Self::Idiv => "idiv",
            Self::Neg => "neg",
            Self::Recip => "recip",
            Self::Log2 => "log2",
            Self::Exp2 => "exp2",
            Self::Sin => "sin",
            Self::Sqrt => "sqrt",
            Self::Floor => "floor",
            Self::BitAnd => "bitand",
            Self::BitOr => "bitor",
            Self::BitXor => "bitxor",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::BitNot => "bitnot",
        }
    }

    /// ElementwiseOpからAstNodeを生成
    pub fn to_ast(&self, input_count: usize) -> AstNode {
        use crate::ast::helper::*;

        match self {
            // 単項演算
            ElementwiseOp::Neg => neg(wildcard("0")),
            ElementwiseOp::Recip => recip(wildcard("0")),
            ElementwiseOp::Log2 => log2(wildcard("0")),
            ElementwiseOp::Exp2 => exp2(wildcard("0")),
            ElementwiseOp::Sin => sin(wildcard("0")),
            ElementwiseOp::Sqrt => sqrt(wildcard("0")),
            ElementwiseOp::Floor => floor(wildcard("0")),
            ElementwiseOp::BitNot => bitnot(wildcard("0")),
            // 二項演算
            ElementwiseOp::Add if input_count == 2 => add(wildcard("0"), wildcard("1")),
            ElementwiseOp::Mul if input_count == 2 => mul(wildcard("0"), wildcard("1")),
            ElementwiseOp::Max if input_count == 2 => max(wildcard("0"), wildcard("1")),
            ElementwiseOp::Rem if input_count == 2 => rem(wildcard("0"), wildcard("1")),
            ElementwiseOp::Idiv if input_count == 2 => idiv(wildcard("0"), wildcard("1")),
            ElementwiseOp::BitAnd if input_count == 2 => bitand(wildcard("0"), wildcard("1")),
            ElementwiseOp::BitOr if input_count == 2 => bitor(wildcard("0"), wildcard("1")),
            ElementwiseOp::BitXor if input_count == 2 => bitxor(wildcard("0"), wildcard("1")),
            ElementwiseOp::Shl if input_count == 2 => shl(wildcard("0"), wildcard("1")),
            ElementwiseOp::Shr if input_count == 2 => shr(wildcard("0"), wildcard("1")),
            // Fallback for binary ops with wrong input count
            _ => panic!(
                "Binary op {:?} requires 2 inputs, got {}",
                self, input_count
            ),
        }
    }
}

impl ReduceOp {
    /// 演算の名前を取得
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sum => "sum",
            Self::Prod => "prod",
            Self::Max => "max",
        }
    }

    /// 単位元を取得
    pub fn identity(&self) -> f32 {
        match self {
            Self::Sum => 0.0,
            Self::Prod => 1.0,
            Self::Max => f32::NEG_INFINITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementwise_op_is_binary() {
        assert!(ElementwiseOp::Add.is_binary());
        assert!(ElementwiseOp::Mul.is_binary());
        assert!(ElementwiseOp::Max.is_binary());
        assert!(!ElementwiseOp::Neg.is_binary());
        assert!(!ElementwiseOp::Sqrt.is_binary());
    }

    #[test]
    fn test_elementwise_op_is_unary() {
        assert!(ElementwiseOp::Neg.is_unary());
        assert!(ElementwiseOp::Recip.is_unary());
        assert!(ElementwiseOp::Floor.is_unary());
        assert!(!ElementwiseOp::Add.is_unary());
    }

    #[test]
    fn test_reduce_op_identity() {
        assert_eq!(ReduceOp::Sum.identity(), 0.0);
        assert_eq!(ReduceOp::Prod.identity(), 1.0);
        assert!(ReduceOp::Max.identity().is_infinite());
    }
}

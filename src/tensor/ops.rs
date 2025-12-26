//! TensorOp - Tensor演算の定義
//!
//! 入力テンソルをTensorOp内に埋め込む設計。
//! Compute演算で全てのElementwise/Reduce演算を統一的に表現。

use std::sync::Arc;

use crate::ast::DType;
use crate::ast::{AstNode, Literal};
use crate::tensor::shape::Expr;
use crate::tensor::{DimDyn, Tensor};

/// テンソル参照型（共有可能、読み取り専用）
pub type TensorRef = Arc<Tensor<f32, DimDyn>>;

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
    View { input: TensorRef },

    /// Viewに従って要素を並べ直す（実体化）
    Contiguous { input: TensorRef },

    /// 型変換
    Cast {
        input: TensorRef,
        target_dtype: DType,
    },

    /// 分岐点を作成（バッファコピー）
    Clone { input: TensorRef },

    // ============================================================
    // 統一計算演算（Compute）
    // ============================================================
    /// 統一された計算演算
    ///
    /// - Elementwise: reduce_op = None, axes = []
    /// - Reduce: expr = Wildcard("0"), reduce_op = Some(...), axes = [...]
    /// - Fused: 任意のexpr + reduce_op
    Compute {
        /// 入力テンソル群（Wildcard("0"), Wildcard("1"), ... に対応）
        inputs: Vec<TensorRef>,
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
    /// パディング
    Pad {
        input: TensorRef,
        padding: Vec<(Expr, Expr)>,
        value: f32,
    },

    /// スライス
    Slice {
        input: TensorRef,
        ranges: Vec<(usize, usize)>,
    },

    /// 結合
    Concat { inputs: Vec<TensorRef>, axis: usize },
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
}

/// 縮約演算の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
}

impl TensorOp {
    /// Elementwise演算を作成
    pub fn elementwise(inputs: Vec<TensorRef>, expr: AstNode) -> Self {
        Self::Compute {
            inputs,
            expr,
            reduce_op: None,
            axes: vec![],
            keepdim: false,
        }
    }

    /// Reduce演算を作成
    pub fn reduce(input: TensorRef, op: ReduceOp, axes: Vec<usize>, keepdim: bool) -> Self {
        Self::Compute {
            inputs: vec![input],
            expr: AstNode::Wildcard("0".to_string()),
            reduce_op: Some(op),
            axes,
            keepdim,
        }
    }

    /// 融合演算を作成
    pub fn fused(
        inputs: Vec<TensorRef>,
        expr: AstNode,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
        keepdim: bool,
    ) -> Self {
        Self::Compute {
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
            TensorOp::Compute {
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
            TensorOp::Compute {
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
    pub fn inputs(&self) -> Vec<&TensorRef> {
        match self {
            TensorOp::Buffer { .. }
            | TensorOp::Const(_)
            | TensorOp::ConstFill(_)
            | TensorOp::Rand
            | TensorOp::Arange
            | TensorOp::Executed => vec![],

            TensorOp::View { input }
            | TensorOp::Contiguous { input }
            | TensorOp::Cast { input, .. }
            | TensorOp::Clone { input }
            | TensorOp::Pad { input, .. }
            | TensorOp::Slice { input, .. } => vec![input],

            TensorOp::Compute { inputs, .. } | TensorOp::Concat { inputs, .. } => {
                inputs.iter().collect()
            }
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
            TensorOp::Cast { target_dtype, .. } => {
                write!(f, "Cast {{ target_dtype: {:?} }}", target_dtype)
            }
            TensorOp::Clone { .. } => write!(f, "Clone"),
            TensorOp::Compute {
                expr,
                reduce_op,
                axes,
                keepdim,
                ..
            } => {
                write!(
                    f,
                    "Compute {{ expr: {:?}, reduce_op: {:?}, axes: {:?}, keepdim: {} }}",
                    expr, reduce_op, axes, keepdim
                )
            }
            TensorOp::Pad { padding, value, .. } => {
                write!(f, "Pad {{ padding: {:?}, value: {} }}", padding, value)
            }
            TensorOp::Slice { ranges, .. } => write!(f, "Slice {{ ranges: {:?} }}", ranges),
            TensorOp::Concat { axis, .. } => write!(f, "Concat {{ axis: {} }}", axis),
        }
    }
}

impl ElementwiseOp {
    /// この演算が二項演算かどうか
    pub fn is_binary(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Mul | Self::Max | Self::Rem | Self::Idiv
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
            // 二項演算
            ElementwiseOp::Add if input_count == 2 => add(wildcard("0"), wildcard("1")),
            ElementwiseOp::Mul if input_count == 2 => mul(wildcard("0"), wildcard("1")),
            ElementwiseOp::Max if input_count == 2 => max(wildcard("0"), wildcard("1")),
            ElementwiseOp::Rem if input_count == 2 => rem(wildcard("0"), wildcard("1")),
            ElementwiseOp::Idiv if input_count == 2 => idiv(wildcard("0"), wildcard("1")),
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

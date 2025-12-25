//! TensorOp - Tensor演算の定義
//!
//! TensorがGraphNodeの役割を直接担うための演算定義。
//! Eager Fusionにより、演算呼び出し時に融合判定が行われる。

use crate::ast::{AstNode, Literal};
use crate::core::DType;
use crate::core::shape::Expr;

/// Tensor演算の種類
///
/// 演算はselfを消費（move）する設計。分岐が必要な場合は明示的にclone()を使用。
#[derive(Debug, Clone)]
pub enum TensorOp {
    // ============================================================
    // 基本演算
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

    /// 型変換
    Cast { target_dtype: DType },

    /// 分岐点を作成（バッファコピー）
    ///
    /// 演算はselfを消費するため、同じテンソルを複数箇所で使用したい場合に使用。
    /// contiguous()時に実際のバッファコピーが実行される。
    Clone,

    // ============================================================
    // View操作
    // ============================================================
    /// Viewを変更（メモリコピーなし）
    View,

    /// Viewに従って要素を並べ直す（実体化）
    ///
    /// このノードに到達した時点でコンパイル・実行が行われる。
    Contiguous,

    // ============================================================
    // Elementwise演算（融合可能）
    // ============================================================
    /// 単一のElementwise演算
    Elementwise { op: ElementwiseOp },

    /// 融合されたElementwise演算
    ///
    /// 複数のElementwise演算が融合された場合に使用。
    /// expr内のWildcard("0"), Wildcard("1")等がsrc[0], src[1]に対応。
    FusedElementwise { expr: AstNode },

    // ============================================================
    // Reduce演算（融合可能）
    // ============================================================
    /// 縮約演算
    Reduce {
        op: ReduceOp,
        axes: Vec<usize>,
        keepdim: bool,
    },

    /// Elementwise → Reduce パターンの融合
    FusedElementwiseReduce {
        expr: AstNode,
        reduce_op: ReduceOp,
        axes: Vec<usize>,
        keepdim: bool,
    },

    // ============================================================
    // 構造操作
    // ============================================================
    /// パディング
    Pad {
        padding: Vec<(Expr, Expr)>,
        value: f32,
    },

    /// スライス
    Slice { ranges: Vec<(usize, usize)> },

    /// 結合
    Concat { axis: usize },

    // ============================================================
    // 実行済み
    // ============================================================
    /// contiguous()実行後の状態
    ///
    /// バッファにデータが格納されている。
    Executed,
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
    Floor, // 新規追加
}

/// 縮約演算の種類
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Prod,
    Max,
}

impl TensorOp {
    /// この演算がElementwiseかどうか
    pub fn is_elementwise(&self) -> bool {
        matches!(
            self,
            TensorOp::Elementwise { .. } | TensorOp::FusedElementwise { .. }
        )
    }

    /// この演算がReduceかどうか
    pub fn is_reduce(&self) -> bool {
        matches!(
            self,
            TensorOp::Reduce { .. } | TensorOp::FusedElementwiseReduce { .. }
        )
    }

    /// この演算が実行済みかどうか
    pub fn is_executed(&self) -> bool {
        matches!(self, TensorOp::Executed)
    }

    /// この演算がView操作かどうか
    pub fn is_view(&self) -> bool {
        matches!(self, TensorOp::View)
    }

    /// この演算がContiguousかどうか
    pub fn is_contiguous(&self) -> bool {
        matches!(self, TensorOp::Contiguous)
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

    #[test]
    fn test_tensor_op_is_elementwise() {
        assert!(
            TensorOp::Elementwise {
                op: ElementwiseOp::Add
            }
            .is_elementwise()
        );

        // Reduce is not elementwise
        let reduce_op = TensorOp::Reduce {
            op: ReduceOp::Sum,
            axes: vec![0],
            keepdim: false,
        };
        assert!(!reduce_op.is_elementwise());
        assert!(reduce_op.is_reduce());
    }
}

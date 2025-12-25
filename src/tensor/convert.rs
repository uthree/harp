//! TensorOp → GraphOp 変換
//!
//! Tensor内部のTensorNodeツリーをGraphNodeツリーに変換します。
//! これにより、既存のLowering/最適化パイプラインを再利用できます。

use std::collections::HashMap;

use crate::graph::shape::View;
use crate::graph::{DType, Graph, GraphNode, GraphOp};

use super::ops::{ElementwiseOp as TensorElementwiseOp, ReduceOp as TensorReduceOp, TensorOp};
use super::{DimDyn, Tensor};
use crate::graph::ops::ElementwiseOp as GraphElementwiseOp;
use crate::graph::ops::ReduceOp as GraphReduceOp;

/// TensorのTensorOpツリーをGraphに変換
///
/// Tensorの内部表現（TensorOp + src Tensors）をGraphノードに変換します。
/// realize()で使用され、既存の最適化・Loweringパイプラインに接続します。
///
/// # 引数
/// * `tensor` - 変換するテンソル
/// * `output_name` - 出力ノードの名前
///
/// # 戻り値
/// 変換されたGraph
pub fn tensor_to_graph(tensor: &Tensor<DimDyn>, output_name: &str) -> Graph {
    let mut graph = Graph::new();
    let mut cache: HashMap<usize, GraphNode> = HashMap::new();

    let node = convert_tensor_recursive(tensor, &mut graph, &mut cache);
    graph.output(output_name, node);

    graph
}

/// テンソルを再帰的にGraphNodeに変換
fn convert_tensor_recursive(
    tensor: &Tensor<DimDyn>,
    _graph: &mut Graph,
    _cache: &mut HashMap<usize, GraphNode>,
) -> GraphNode {
    // 現在はGraphNodeをそのまま使用（将来的にTensorNodeに置き換え）
    // TODO: Phase 6.2以降でTensorNodeベースに変更
    tensor.node.clone()
}

/// TensorOpをGraphOpに変換
///
/// # 注意
/// この関数は将来的にconvert_tensor_recursiveで使用されます。
/// 現在はTensorがGraphNodeを直接保持しているため未使用ですが、
/// TensorNode移行後に使用されます。
#[allow(dead_code)]
fn convert_tensor_op(
    op: &TensorOp,
    src_nodes: &[GraphNode],
    view: &View,
    _dtype: &DType,
) -> (GraphOp, Vec<GraphNode>) {
    match op {
        // 基本演算
        TensorOp::Buffer { name } => (GraphOp::Buffer { name: name.clone() }, vec![]),

        TensorOp::Const(lit) => (GraphOp::Const(lit.clone()), vec![]),

        TensorOp::ConstFill(lit) => (GraphOp::ConstFill(lit.clone()), vec![]),

        TensorOp::Rand => (GraphOp::Rand, vec![]),

        TensorOp::Arange => (GraphOp::Arange, vec![]),

        TensorOp::Cast { target_dtype } => (
            GraphOp::Cast {
                target_dtype: target_dtype.clone(),
            },
            src_nodes.to_vec(),
        ),

        TensorOp::Clone => (GraphOp::Clone, src_nodes.to_vec()),

        // View操作
        TensorOp::View => (GraphOp::View(view.clone()), src_nodes.to_vec()),

        TensorOp::Contiguous => (GraphOp::Contiguous, src_nodes.to_vec()),

        // Elementwise演算
        TensorOp::Elementwise { op } => (
            GraphOp::Elementwise {
                op: convert_elementwise_op(op),
            },
            src_nodes.to_vec(),
        ),

        TensorOp::FusedElementwise { expr } => (
            GraphOp::FusedElementwise { expr: expr.clone() },
            src_nodes.to_vec(),
        ),

        // Reduce演算
        // TensorOpのReduceは複数軸とkeepdimをサポート
        // GraphOpのReduceは単一軸のみなので、複数軸の場合は連鎖させる必要がある
        TensorOp::Reduce {
            op,
            axes,
            keepdim: _,
        } => {
            // 現在は単純化のため最初の軸のみを変換
            // TODO: 複数軸のReduce変換を実装
            // TODO: keepdim対応（View操作でunsqueezeを追加）
            let axis = axes.first().copied().unwrap_or(0);
            (
                GraphOp::Reduce {
                    op: convert_reduce_op(op),
                    axis,
                    reduce_strategy: None,
                },
                src_nodes.to_vec(),
            )
        }

        TensorOp::FusedElementwiseReduce {
            expr,
            reduce_op,
            axes,
            keepdim: _,
        } => (
            GraphOp::FusedElementwiseReduce {
                expr: expr.clone(),
                reduce_op: convert_reduce_op(reduce_op),
                axes: axes.clone(),
                reduce_strategy: None,
            },
            src_nodes.to_vec(),
        ),

        // 構造操作
        TensorOp::Pad { padding, value } => (
            GraphOp::Pad {
                padding: padding.clone(),
                value: *value,
            },
            src_nodes.to_vec(),
        ),

        TensorOp::Slice { ranges } => (
            GraphOp::Slice {
                ranges: ranges.clone(),
            },
            src_nodes.to_vec(),
        ),

        TensorOp::Concat { axis } => (GraphOp::Concat { axis: *axis }, src_nodes.to_vec()),

        // Executed: すでに実行済みなのでBufferとして扱う
        TensorOp::Executed => (
            GraphOp::Buffer {
                name: "executed".to_string(),
            },
            vec![],
        ),
    }
}

/// TensorElementwiseOp → GraphElementwiseOp変換
fn convert_elementwise_op(op: &TensorElementwiseOp) -> GraphElementwiseOp {
    match op {
        TensorElementwiseOp::Add => GraphElementwiseOp::Add,
        TensorElementwiseOp::Mul => GraphElementwiseOp::Mul,
        TensorElementwiseOp::Max => GraphElementwiseOp::Max,
        TensorElementwiseOp::Rem => GraphElementwiseOp::Rem,
        TensorElementwiseOp::Idiv => GraphElementwiseOp::Idiv,
        TensorElementwiseOp::Neg => GraphElementwiseOp::Neg,
        TensorElementwiseOp::Recip => GraphElementwiseOp::Recip,
        TensorElementwiseOp::Log2 => GraphElementwiseOp::Log2,
        TensorElementwiseOp::Exp2 => GraphElementwiseOp::Exp2,
        TensorElementwiseOp::Sin => GraphElementwiseOp::Sin,
        TensorElementwiseOp::Sqrt => GraphElementwiseOp::Sqrt,
        TensorElementwiseOp::Floor => GraphElementwiseOp::Floor,
    }
}

/// TensorReduceOp → GraphReduceOp変換
fn convert_reduce_op(op: &TensorReduceOp) -> GraphReduceOp {
    match op {
        TensorReduceOp::Sum => GraphReduceOp::Sum,
        TensorReduceOp::Prod => GraphReduceOp::Prod,
        TensorReduceOp::Max => GraphReduceOp::Max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    fn test_convert_elementwise_op() {
        assert!(matches!(
            convert_elementwise_op(&TensorElementwiseOp::Add),
            GraphElementwiseOp::Add
        ));
        assert!(matches!(
            convert_elementwise_op(&TensorElementwiseOp::Floor),
            GraphElementwiseOp::Floor
        ));
    }

    #[test]
    fn test_convert_reduce_op() {
        assert!(matches!(
            convert_reduce_op(&TensorReduceOp::Sum),
            GraphReduceOp::Sum
        ));
        assert!(matches!(
            convert_reduce_op(&TensorReduceOp::Max),
            GraphReduceOp::Max
        ));
    }

    #[test]
    fn test_convert_tensor_op_buffer() {
        use crate::graph::shape::Expr;
        let empty_shape: Vec<Expr> = vec![];
        let (op, srcs) = convert_tensor_op(
            &TensorOp::Buffer {
                name: "test".to_string(),
            },
            &[],
            &View::contiguous(empty_shape),
            &DType::F32,
        );
        assert!(matches!(op, GraphOp::Buffer { name } if name == "test"));
        assert!(srcs.is_empty());
    }

    #[test]
    fn test_convert_tensor_op_const() {
        use crate::graph::shape::Expr;
        let empty_shape: Vec<Expr> = vec![];
        let (op, _) = convert_tensor_op(
            &TensorOp::Const(Literal::F32(1.0)),
            &[],
            &View::contiguous(empty_shape),
            &DType::F32,
        );
        assert!(matches!(op, GraphOp::Const(Literal::F32(v)) if v == 1.0));
    }
}

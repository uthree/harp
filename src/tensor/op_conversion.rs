use crate::graph::{Graph, NodeId, NodeView};
use crate::ast::DType;
use crate::graph::shape::expr::Expr;
use super::{TensorOp};

pub fn op_to_graph_op(
    op: TensorOp,
    graph: &Graph,
    srcs: Vec<NodeId>,
    shape: Vec<Expr>,
    dtype: DType,
) -> NodeView<'_> {
    let src_views: Vec<NodeView> = srcs.iter().map(|&id| graph.get_view(id)).collect();

    match op {
        TensorOp::Rand => graph.rand(dtype, shape),
        TensorOp::Full(val) => graph.full(val, shape),
        TensorOp::Add => src_views[0].clone() + src_views[1].clone(),
        TensorOp::Sub => src_views[0].clone() - src_views[1].clone(),
        TensorOp::Mul => src_views[0].clone() * src_views[1].clone(),
        TensorOp::Neg => -src_views[0].clone(),
        TensorOp::Recip => src_views[0].clone().recip(),
        TensorOp::Sin => src_views[0].clone().sin(),
        TensorOp::Exp2 => src_views[0].clone().exp2(),
        TensorOp::Log2 => src_views[0].clone().log2(),
        TensorOp::Sqrt => src_views[0].clone().sqrt(),
        TensorOp::Permute(axes) => src_views[0].clone().permute(axes),
        TensorOp::Reshape(new_shape) => {
            src_views[0].clone().reshape(new_shape.iter().map(|&d| d.into()).collect())
        }
        TensorOp::Expand(new_shape) => {
            src_views[0].clone().expand(new_shape.iter().map(|&d| d.into()).collect())
        }
        TensorOp::Squeeze(dim) => src_views[0].clone().squeeze(dim),
        TensorOp::Unsqueeze(dim) => src_views[0].clone().unsqueeze(dim),
        TensorOp::Slice(args) => src_views[0]
            .clone()
            .slice(args.iter().map(|(s, e)| ((*s).into(), (*e).into())).collect()),
    }
}

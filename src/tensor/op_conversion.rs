use super::TensorOp;
use crate::ast::DType;
use crate::graph::shape::expr::Expr;
use crate::graph::{Graph, NodeId, NodeView};

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
        TensorOp::Add => src_views[0] + src_views[1],
        TensorOp::Sub => src_views[0] - src_views[1],
        TensorOp::Mul => src_views[0] * src_views[1],
        TensorOp::Neg => -src_views[0],
        TensorOp::Recip => src_views[0].recip(),
        TensorOp::Sin => src_views[0].sin(),
        TensorOp::Exp2 => src_views[0].exp2(),
        TensorOp::Log2 => src_views[0].log2(),
        TensorOp::Sqrt => src_views[0].sqrt(),
        TensorOp::Permute(axes) => src_views[0].clone().permute(axes),
        TensorOp::Reshape(new_shape) => src_views[0]
            .clone()
            .reshape(new_shape.iter().map(|&d| d.into()).collect()),
        TensorOp::Expand(new_shape) => src_views[0]
            .clone()
            .expand(new_shape.iter().map(|&d| d.into()).collect()),
        TensorOp::Squeeze(dim) => src_views[0].clone().squeeze(dim),
        TensorOp::Unsqueeze(dim) => src_views[0].clone().unsqueeze(dim),
        TensorOp::Slice(args) => src_views[0].clone().slice(
            args.iter()
                .map(|(s, e)| ((*s).into(), (*e).into()))
                .collect(),
        ),
        TensorOp::Reduce(op, axis) => match op {
            crate::ast::AstOp::Add => src_views[0].clone().sum(axis),
            crate::ast::AstOp::Max => src_views[0].clone().max(axis),
            crate::ast::AstOp::Mul => src_views[0].clone().prod(axis),
            _ => panic!("Unsupported reduce operation"),
        },
    }
}

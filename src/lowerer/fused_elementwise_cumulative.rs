use crate::ast::{AstNode, VariableDecl};
use crate::graph::GraphNode;

/// FusedElementwiseCumulative演算のコード生成を行う構造体
pub(super) struct FusedElementwiseCumulativeLowerer;

impl FusedElementwiseCumulativeLowerer {
    /// FusedElementwiseCumulative演算のコード生成
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower(
        _node: &GraphNode,
        _ast: &AstNode,
        _inputs: &[GraphNode],
        _op: &crate::graph::ops::CumulativeOp,
        _declarations: &mut Vec<VariableDecl>,
        mut _get_var: impl FnMut(&GraphNode) -> String,
    ) -> Option<AstNode> {
        // TODO: 実装
        todo!("FusedElementwiseCumulative not yet implemented in lowerer")
    }
}

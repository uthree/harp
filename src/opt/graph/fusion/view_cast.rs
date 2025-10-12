use super::GraphFusionOptimizer;
use crate::graph::{GraphNode, GraphOp};

impl GraphFusionOptimizer {
    /// View -> Viewチェーンの統合を試みる
    /// 連続したViewノードがあり、分岐がない場合に統合
    pub(crate) fn try_fuse_view_chain(&mut self, node: &GraphNode) -> Option<GraphNode> {
        // このノードがViewでない場合は統合しない
        let GraphOp::View(input) = &node.op else {
            return None;
        };

        // inputがViewでない、または分岐している場合は統合しない
        if !matches!(input.op, GraphOp::View(_)) || self.is_branching(input) {
            return None;
        }

        // Viewチェーンの最初のsourceを見つける
        let mut current = input;
        let mut source = input;
        let mut chain_length = 1; // 最初の1つをカウント

        while let GraphOp::View(input_node) = &current.op {
            source = input_node;
            chain_length += 1;
            // 分岐している場合はここで停止
            if self.is_branching(input_node) {
                break;
            }
            // inputがViewなら続ける
            if matches!(input_node.op, GraphOp::View(_)) {
                current = input_node;
            } else {
                // Viewでなければ終了
                break;
            }
        }

        // sourceをrebuild
        let rebuilt_source = self.rebuild_node(source);

        // 最終的なviewは現在のnodeのview
        let fused_node = GraphNode::new(
            GraphOp::View(rebuilt_source),
            node.dtype.clone(),
            node.view.clone(),
        );

        log::debug!(
            "Fused View chain of length {} into single View",
            chain_length
        );

        Some(fused_node)
    }

    /// Cast -> Castチェーンの統合を試みる
    /// 連続したCastノードがあり、分岐がない場合に統合
    pub(crate) fn try_fuse_cast_chain(&mut self, node: &GraphNode) -> Option<GraphNode> {
        // このノードがCastでない場合は統合しない
        let GraphOp::Cast(input, target_dtype) = &node.op else {
            return None;
        };

        // inputがCastでない、または分岐している場合は統合しない
        if !matches!(input.op, GraphOp::Cast(_, _)) || self.is_branching(input) {
            return None;
        }

        // Castチェーンの最初のsourceを見つける
        let mut current = input;
        let mut source = input;

        while let GraphOp::Cast(input_node, _) = &current.op {
            source = input_node;
            // 分岐している場合はここで停止
            if self.is_branching(input_node) {
                break;
            }
            // inputがCastなら続ける
            if matches!(input_node.op, GraphOp::Cast(_, _)) {
                current = input_node;
            } else {
                // Castでなければ終了
                break;
            }
        }

        // sourceをrebuild
        let rebuilt_source = self.rebuild_node(source);

        // 最終的なdtypeは現在のnodeのdtype
        let fused_node = GraphNode::new(
            GraphOp::Cast(rebuilt_source, target_dtype.clone()),
            node.dtype.clone(),
            node.view.clone(),
        );

        Some(fused_node)
    }
}

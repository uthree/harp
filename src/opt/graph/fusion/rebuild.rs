use super::GraphFusionOptimizer;
use crate::graph::ops::ElementwiseOp;
use crate::graph::{GraphNode, GraphOp};

impl GraphFusionOptimizer {
    /// ノードを再構築し、可能な場合は融合を適用
    pub(crate) fn rebuild_node(&mut self, node: &GraphNode) -> GraphNode {
        // すでに処理済みの場合はキャッシュから返す
        if let Some(rebuilt) = self.node_mapping.get(node) {
            return rebuilt.clone();
        }

        // まず、このノードが融合可能かチェック
        if let Some(fused) = self.try_fuse_elementwise_chain(node) {
            self.node_mapping.insert(node.clone(), fused.clone());
            return fused;
        }

        // 融合できない場合は、依存ノードを再構築してから新しいノードを作成
        let rebuilt = match &node.op {
            GraphOp::Input(_) | GraphOp::Const(_) => {
                // これらのノードは依存がないのでそのまま返す
                node.clone()
            }
            GraphOp::Contiguous(input) => {
                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Contiguous(rebuilt_input),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::Elementwise(op) => {
                // 入力を再構築
                let rebuilt_op = self.rebuild_elementwise_op(op);
                GraphNode::new(
                    GraphOp::Elementwise(rebuilt_op),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::Reduce(op, axis, input) => {
                // Reduce -> Reduceの融合を試みる
                if let Some((axes, final_input)) = self.try_fuse_reduce_chain(op, *axis, input) {
                    let rebuilt_input = self.rebuild_node(&final_input);
                    return GraphNode::new(
                        GraphOp::FusedReduce(op.clone(), axes, rebuilt_input),
                        node.dtype.clone(),
                        node.view.clone(),
                    );
                }

                // Elementwise -> Reduceの融合を試みる
                if let Some((ast, inputs, axes)) =
                    self.try_fuse_elementwise_reduce(op, *axis, input)
                {
                    // 融合されたinputsを再構築してView統合などを適用
                    let rebuilt_inputs: Vec<GraphNode> = inputs
                        .iter()
                        .map(|input| self.rebuild_node(input))
                        .collect();
                    return GraphNode::new(
                        GraphOp::FusedElementwiseReduce(ast, rebuilt_inputs, op.clone(), axes),
                        node.dtype.clone(),
                        node.view.clone(),
                    );
                }

                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Reduce(op.clone(), *axis, rebuilt_input),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::Cumulative(op, axis, input) => {
                // Elementwise -> Cumulativeの融合を試みる
                if let Some((ast, inputs, fused_axis)) =
                    self.try_fuse_elementwise_cumulative(op, *axis, input)
                {
                    // 融合されたinputsを再構築してView統合などを適用
                    let rebuilt_inputs: Vec<GraphNode> = inputs
                        .iter()
                        .map(|input| self.rebuild_node(input))
                        .collect();
                    return GraphNode::new(
                        GraphOp::FusedElementwiseCumulative(
                            ast,
                            rebuilt_inputs,
                            op.clone(),
                            fused_axis,
                        ),
                        node.dtype.clone(),
                        node.view.clone(),
                    );
                }

                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Cumulative(op.clone(), *axis, rebuilt_input),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::View(input) => {
                // View -> Viewチェーンの統合を試みる
                if let Some(fused) = self.try_fuse_view_chain(node) {
                    return fused;
                }

                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::View(rebuilt_input),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::Cast(input, target_dtype) => {
                // Cast -> Castチェーンの統合を試みる
                if let Some(fused) = self.try_fuse_cast_chain(node) {
                    return fused;
                }

                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Cast(rebuilt_input, target_dtype.clone()),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::Fold(dim, window_size, stride, dilation, input) => {
                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Fold(*dim, *window_size, *stride, *dilation, rebuilt_input),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::Pad(input, axis, amount) => {
                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Pad(rebuilt_input, *axis, amount.clone()),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::FusedElementwise(_, _)
            | GraphOp::FusedReduce(_, _, _)
            | GraphOp::FusedElementwiseReduce(_, _, _, _)
            | GraphOp::FusedElementwiseCumulative(_, _, _, _) => {
                // すでに融合済みのノードはそのまま
                node.clone()
            }
        };

        self.node_mapping.insert(node.clone(), rebuilt.clone());
        rebuilt
    }

    /// ElementwiseOpを再構築
    pub(crate) fn rebuild_elementwise_op(&mut self, op: &ElementwiseOp) -> ElementwiseOp {
        match op {
            ElementwiseOp::Add(lhs, rhs) => {
                ElementwiseOp::Add(self.rebuild_node(lhs), self.rebuild_node(rhs))
            }
            ElementwiseOp::Mul(lhs, rhs) => {
                ElementwiseOp::Mul(self.rebuild_node(lhs), self.rebuild_node(rhs))
            }
            ElementwiseOp::Max(lhs, rhs) => {
                ElementwiseOp::Max(self.rebuild_node(lhs), self.rebuild_node(rhs))
            }
            ElementwiseOp::Mod(lhs, rhs) => {
                ElementwiseOp::Mod(self.rebuild_node(lhs), self.rebuild_node(rhs))
            }
            ElementwiseOp::Neg(input) => ElementwiseOp::Neg(self.rebuild_node(input)),
            ElementwiseOp::Recip(input) => ElementwiseOp::Recip(self.rebuild_node(input)),
            ElementwiseOp::Sin(input) => ElementwiseOp::Sin(self.rebuild_node(input)),
            ElementwiseOp::Sqrt(input) => ElementwiseOp::Sqrt(self.rebuild_node(input)),
            ElementwiseOp::Log2(input) => ElementwiseOp::Log2(self.rebuild_node(input)),
            ElementwiseOp::Exp2(input) => ElementwiseOp::Exp2(self.rebuild_node(input)),
            ElementwiseOp::LessThan(lhs, rhs) => {
                ElementwiseOp::LessThan(self.rebuild_node(lhs), self.rebuild_node(rhs))
            }
            ElementwiseOp::Eq(lhs, rhs) => {
                ElementwiseOp::Eq(self.rebuild_node(lhs), self.rebuild_node(rhs))
            }
            ElementwiseOp::Select(cond, true_val, false_val) => ElementwiseOp::Select(
                self.rebuild_node(cond),
                self.rebuild_node(true_val),
                self.rebuild_node(false_val),
            ),
        }
    }
}

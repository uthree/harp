use super::GraphFusionOptimizer;
use crate::ast::AstNode;
use crate::graph::ops::ElementwiseOp;
use crate::graph::{GraphNode, GraphOp};
use std::collections::HashMap;

impl GraphFusionOptimizer {
    /// Elementwise -> Elementwiseの融合を試みる
    pub(crate) fn try_fuse_elementwise_chain(&mut self, node: &GraphNode) -> Option<GraphNode> {
        // このノードがElementwiseでない場合は融合しない
        let GraphOp::Elementwise(_) = node.op else {
            return None;
        };

        // 入力ノードのマッピングとリストを作成
        let mut input_mapping = HashMap::new();
        let mut inputs = Vec::new();

        // ノードをASTに変換（再帰的に入力も変換）
        let ast =
            self.elementwise_to_ast_with_branching_check(node, &mut input_mapping, &mut inputs)?;

        // 融合可能なノードが1つ以上ある場合のみ融合
        // (単一ノードの場合は融合する意味がない)
        if inputs.len() <= 1 {
            return None;
        }

        // 融合したノードを作成
        // inputsを再構築してView統合などを適用
        let rebuilt_inputs: Vec<GraphNode> = inputs
            .iter()
            .map(|input| self.rebuild_node(input))
            .collect();
        let fused_node = GraphNode::new(
            GraphOp::FusedElementwise(ast, rebuilt_inputs),
            node.dtype.clone(),
            node.view.clone(),
        );

        Some(fused_node)
    }

    /// 分岐チェック付きでElementwiseノードをAstNodeに変換
    /// 分岐があった場合はNoneを返す
    pub(crate) fn elementwise_to_ast_with_branching_check(
        &self,
        node: &GraphNode,
        input_mapping: &mut HashMap<GraphNode, usize>,
        inputs: &mut Vec<GraphNode>,
    ) -> Option<AstNode> {
        // すでに処理済みのノードの場合、対応するCaptureを返す
        if let Some(&idx) = input_mapping.get(node) {
            return Some(AstNode::Capture(idx));
        }

        match &node.op {
            GraphOp::Elementwise(op) => {
                // Elementwiseの入力をチェック
                let elementwise_inputs = self.get_elementwise_inputs(node);

                // 各入力について、分岐していないかチェック
                for input in &elementwise_inputs {
                    if matches!(input.op, GraphOp::Elementwise(_)) && self.is_branching(input) {
                        // 分岐しているElementwiseノードは融合できない
                        // このノードを入力として追加
                        let idx = inputs.len();
                        inputs.push(node.clone());
                        input_mapping.insert(node.clone(), idx);
                        return Some(AstNode::Capture(idx));
                    }
                }

                // 分岐していない場合は再帰的に変換
                Some(self.elementwise_op_to_ast_with_check(op, input_mapping, inputs)?)
            }
            _ => {
                // Elementwise以外のノードは新しい入力として追加
                let idx = inputs.len();
                inputs.push(node.clone());
                input_mapping.insert(node.clone(), idx);
                Some(AstNode::Capture(idx))
            }
        }
    }

    /// 分岐チェック付きでElementwiseOpをAstNodeに変換
    fn elementwise_op_to_ast_with_check(
        &self,
        op: &ElementwiseOp,
        input_mapping: &mut HashMap<GraphNode, usize>,
        inputs: &mut Vec<GraphNode>,
    ) -> Option<AstNode> {
        match op {
            ElementwiseOp::Add(lhs, rhs) => {
                let lhs_ast =
                    self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast =
                    self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(lhs_ast + rhs_ast)
            }
            ElementwiseOp::Mul(lhs, rhs) => {
                let lhs_ast =
                    self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast =
                    self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(lhs_ast * rhs_ast)
            }
            ElementwiseOp::Max(lhs, rhs) => {
                let lhs_ast =
                    self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast =
                    self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(AstNode::Max(Box::new(lhs_ast), Box::new(rhs_ast)))
            }
            ElementwiseOp::Mod(lhs, rhs) => {
                let lhs_ast =
                    self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast =
                    self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(lhs_ast % rhs_ast)
            }
            ElementwiseOp::Neg(input) => {
                let input_ast =
                    self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(-input_ast)
            }
            ElementwiseOp::Recip(input) => {
                let input_ast =
                    self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Recip(Box::new(input_ast)))
            }
            ElementwiseOp::Sin(input) => {
                let input_ast =
                    self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Sin(Box::new(input_ast)))
            }
            ElementwiseOp::Sqrt(input) => {
                let input_ast =
                    self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Sqrt(Box::new(input_ast)))
            }
            ElementwiseOp::Log2(input) => {
                let input_ast =
                    self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Log2(Box::new(input_ast)))
            }
            ElementwiseOp::Exp2(input) => {
                let input_ast =
                    self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Exp2(Box::new(input_ast)))
            }
            ElementwiseOp::LessThan(lhs, rhs) => {
                let lhs_ast =
                    self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast =
                    self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(AstNode::less_than(lhs_ast, rhs_ast))
            }
            ElementwiseOp::Eq(lhs, rhs) => {
                let lhs_ast =
                    self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast =
                    self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(AstNode::eq(lhs_ast, rhs_ast))
            }
            ElementwiseOp::Select(cond, true_val, false_val) => {
                let cond_ast =
                    self.elementwise_to_ast_with_branching_check(cond, input_mapping, inputs)?;
                let true_ast =
                    self.elementwise_to_ast_with_branching_check(true_val, input_mapping, inputs)?;
                let false_ast =
                    self.elementwise_to_ast_with_branching_check(false_val, input_mapping, inputs)?;
                Some(AstNode::select(cond_ast, true_ast, false_ast))
            }
        }
    }

    /// Elementwise演算の入力ノードを取得
    pub(crate) fn get_elementwise_inputs(&self, node: &GraphNode) -> Vec<GraphNode> {
        match &node.op {
            GraphOp::Elementwise(ElementwiseOp::Add(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::Mul(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::Max(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::Mod(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::LessThan(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::Eq(lhs, rhs)) => {
                vec![lhs.clone(), rhs.clone()]
            }
            GraphOp::Elementwise(ElementwiseOp::Neg(input))
            | GraphOp::Elementwise(ElementwiseOp::Recip(input))
            | GraphOp::Elementwise(ElementwiseOp::Sin(input))
            | GraphOp::Elementwise(ElementwiseOp::Sqrt(input))
            | GraphOp::Elementwise(ElementwiseOp::Log2(input))
            | GraphOp::Elementwise(ElementwiseOp::Exp2(input)) => {
                vec![input.clone()]
            }
            GraphOp::Elementwise(ElementwiseOp::Select(cond, true_val, false_val)) => {
                vec![cond.clone(), true_val.clone(), false_val.clone()]
            }
            _ => vec![],
        }
    }
}

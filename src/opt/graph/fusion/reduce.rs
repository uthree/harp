use super::GraphFusionOptimizer;
use crate::ast::AstNode;
use crate::graph::ops::{CumulativeOp, ReduceOp};
use crate::graph::{GraphNode, GraphOp};
use std::collections::HashMap;

impl GraphFusionOptimizer {
    /// Elementwise -> Reduceの融合を試みる
    pub(crate) fn try_fuse_elementwise_reduce(
        &mut self,
        _op: &ReduceOp,
        axis: usize,
        input: &GraphNode,
    ) -> Option<(AstNode, Vec<GraphNode>, Vec<usize>)> {
        // inputがElementwiseでない場合は融合しない
        let GraphOp::Elementwise(_) = input.op else {
            return None;
        };

        // inputが分岐している場合は融合しない
        if self.is_branching(input) {
            return None;
        }

        // 入力ノードのマッピングとリストを作成
        let mut input_mapping = HashMap::new();
        let mut inputs = Vec::new();

        // ノードをASTに変換（再帰的に入力も変換）
        let ast =
            self.elementwise_to_ast_with_branching_check(input, &mut input_mapping, &mut inputs)?;

        // 融合可能なノードが1つ以上ある場合のみ融合
        if inputs.is_empty() {
            return None;
        }

        // 単一軸のリストとして返す
        Some((ast, inputs, vec![axis]))
    }

    /// Elementwise -> Cumulativeの融合を試みる
    pub(crate) fn try_fuse_elementwise_cumulative(
        &mut self,
        _op: &CumulativeOp,
        _axis: usize,
        input: &GraphNode,
    ) -> Option<(AstNode, Vec<GraphNode>)> {
        // inputがElementwiseでない場合は融合しない
        let GraphOp::Elementwise(_) = input.op else {
            return None;
        };

        // inputが分岐している場合は融合しない
        if self.is_branching(input) {
            return None;
        }

        // 入力ノードのマッピングとリストを作成
        let mut input_mapping = HashMap::new();
        let mut inputs = Vec::new();

        // ノードをASTに変換（再帰的に入力も変換）
        let ast =
            self.elementwise_to_ast_with_branching_check(input, &mut input_mapping, &mut inputs)?;

        // 融合可能なノードが1つ以上ある場合のみ融合
        if inputs.is_empty() {
            return None;
        }

        Some((ast, inputs))
    }

    /// Reduce -> Reduceの融合を試みる
    pub(crate) fn try_fuse_reduce_chain(
        &mut self,
        op: &ReduceOp,
        axis: usize,
        input: &GraphNode,
    ) -> Option<(Vec<usize>, GraphNode)> {
        // inputがReduceでない、または異なる演算子の場合は融合しない
        let GraphOp::Reduce(input_op, input_axis, inner_input) = &input.op else {
            return None;
        };

        if input_op != op {
            return None;
        }

        // inputが分岐している場合は融合しない
        if self.is_branching(input) {
            return None;
        }

        // 軸のリストを計算
        // 最初のreduceのaxisと、2番目のaxisを元のテンソルの軸に変換
        let first_axis = *input_axis;
        let second_axis = if axis >= first_axis {
            axis + 1 // first_axisで縮約されたので、それ以降の軸は+1
        } else {
            axis
        };

        // 再帰的にさらに融合可能かチェック
        if let Some((mut axes, final_input)) =
            self.try_fuse_reduce_chain(input_op, first_axis, inner_input)
        {
            // さらに融合できる場合
            axes.push(second_axis);
            Some((axes, final_input))
        } else {
            // これ以上融合できない場合
            Some((vec![first_axis, second_axis], inner_input.clone()))
        }
    }
}

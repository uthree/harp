use crate::ast::AstNode;
use crate::graph::ops::{CumulativeOp, ElementwiseOp, ReduceOp};
use crate::graph::{Graph, GraphNode, GraphOp};
use crate::opt::graph::GraphOptimizer;
use std::collections::HashMap;

pub struct GraphFusionOptimizer {
    // 融合したノードのマッピング: 古いノード -> 新しいノード
    node_mapping: HashMap<GraphNode, GraphNode>,
    // 最適化の各ステップでのグラフのスナップショット
    pub snapshots: Vec<OptimizationSnapshot>,
    // ログ記録を有効にするかどうか
    pub enable_logging: bool,
    // 最適化完了後のコールバック（VIZ=1の時にビジュアライザーを起動）
    pub on_complete: Option<Box<dyn FnOnce(Vec<OptimizationSnapshot>)>>,
}

#[derive(Debug, Clone)]
pub struct OptimizationSnapshot {
    pub description: String,
    pub graph: Graph,
}

impl GraphFusionOptimizer {
    pub fn new() -> Self {
        // VIZ環境変数が"1"なら自動的にログを有効化
        #[cfg(feature = "visualizer")]
        let enable_logging = std::env::var("VIZ").map(|v| v == "1").unwrap_or(false);
        #[cfg(not(feature = "visualizer"))]
        let enable_logging = false;

        Self {
            node_mapping: HashMap::new(),
            snapshots: Vec::new(),
            enable_logging,
            on_complete: None,
        }
    }

    pub fn with_logging(mut self) -> Self {
        self.enable_logging = true;
        self
    }

    pub fn with_visualizer<F>(mut self, callback: F) -> Self
    where
        F: FnOnce(Vec<OptimizationSnapshot>) + 'static,
    {
        self.on_complete = Some(Box::new(callback));
        self
    }

    /// VIZ=1が設定されている場合、最適化完了時にコールバックを設定
    /// この関数は環境変数をチェックして、VIZ=1の場合のみコールバックを追加
    pub fn auto_visualize<F>(self, callback: F) -> Self
    where
        F: FnOnce(Vec<OptimizationSnapshot>) + 'static,
    {
        if crate::opt::graph::is_viz_enabled() {
            self.with_visualizer(callback)
        } else {
            self
        }
    }

    fn log_snapshot(&mut self, description: String, graph: &Graph) {
        if self.enable_logging {
            let snapshot = OptimizationSnapshot {
                description,
                graph: graph.clone(),
            };
            self.snapshots.push(snapshot);
        }
    }

    /// ノードが分岐しているか（複数の場所から参照されているか）を判定
    /// 分岐している場合は融合しない
    fn is_branching(&self, node: &GraphNode) -> bool {
        // strong_countで参照数を取得
        // 1つはnode自身、もう1つは親ノードからの参照
        // それ以上あれば分岐している
        node.strong_count() > 2
    }

    /// View -> Viewチェーンの統合を試みる
    /// 連続したViewノードがあり、分岐がない場合に統合
    fn try_fuse_view_chain(&mut self, node: &GraphNode) -> Option<GraphNode> {
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

        loop {
            match &current.op {
                GraphOp::View(input_node) => {
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
                _ => break,
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
    fn try_fuse_cast_chain(&mut self, node: &GraphNode) -> Option<GraphNode> {
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

        loop {
            match &current.op {
                GraphOp::Cast(input_node, _) => {
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
                _ => break,
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

    /// Elementwise -> Elementwiseの融合を試みる
    fn try_fuse_elementwise_chain(&mut self, node: &GraphNode) -> Option<GraphNode> {
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
    fn elementwise_to_ast_with_branching_check(
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
    fn get_elementwise_inputs(&self, node: &GraphNode) -> Vec<GraphNode> {
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

    /// Elementwise -> Reduceの融合を試みる
    fn try_fuse_elementwise_reduce(
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
    fn try_fuse_elementwise_cumulative(
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
    fn try_fuse_reduce_chain(
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

impl Default for GraphFusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphOptimizer for GraphFusionOptimizer {
    fn optimize(&mut self, graph: &mut Graph) {
        // 最適化前のスナップショット
        self.log_snapshot("Initial graph".to_string(), graph);

        // 出力ノードから再帰的にグラフを再構築
        let new_outputs: Vec<GraphNode> = graph
            .outputs
            .iter()
            .map(|output| self.rebuild_node(output))
            .collect();

        graph.outputs = new_outputs;

        // 最適化後のスナップショット
        self.log_snapshot("After fusion optimization".to_string(), graph);

        // コールバックがあれば実行（ビジュアライザー起動）
        if let Some(callback) = self.on_complete.take() {
            callback(self.snapshots.clone());
        }
    }
}

impl GraphFusionOptimizer {
    /// ノードを再構築し、可能な場合は融合を適用
    fn rebuild_node(&mut self, node: &GraphNode) -> GraphNode {
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
                if let Some((ast, inputs)) = self.try_fuse_elementwise_cumulative(op, *axis, input)
                {
                    // 融合されたinputsを再構築してView統合などを適用
                    let rebuilt_inputs: Vec<GraphNode> = inputs
                        .iter()
                        .map(|input| self.rebuild_node(input))
                        .collect();
                    return GraphNode::new(
                        GraphOp::FusedElementwiseCumulative(ast, rebuilt_inputs, op.clone()),
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
            GraphOp::FusedElementwise(_, _)
            | GraphOp::FusedReduce(_, _, _)
            | GraphOp::FusedElementwiseReduce(_, _, _, _)
            | GraphOp::FusedElementwiseCumulative(_, _, _) => {
                // すでに融合済みのノードはそのまま
                node.clone()
            }
        };

        self.node_mapping.insert(node.clone(), rebuilt.clone());
        rebuilt
    }

    /// ElementwiseOpを再構築
    fn rebuild_elementwise_op(&mut self, op: &ElementwiseOp) -> ElementwiseOp {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::Expr;

    #[test]
    fn test_is_branching() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![Expr::from(10)]);

        let optimizer = GraphFusionOptimizer::new();

        // 単一参照の場合は分岐していない
        assert!(!optimizer.is_branching(&input));

        // 複数参照がある場合は分岐している
        let _add1 = input.clone() + input.clone();
        assert!(optimizer.is_branching(&input));
    }

    #[test]
    fn test_view_chain_fusion() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![Expr::from(2), Expr::from(3)]);

        // View変換のチェーン: unsqueeze -> expand -> permute
        let unsqueezed = input.unsqueeze(2); // [2, 3] -> [2, 3, 1]
        let expanded = unsqueezed.expand(vec![2.into(), 3.into(), 4.into()]); // -> [2, 3, 4]
        let permuted = expanded.permute(vec![2, 0, 1]); // -> [4, 2, 3]

        graph.output(permuted.clone());

        let mut optimizer = GraphFusionOptimizer::new();
        optimizer.optimize(&mut graph);

        // 最適化後、outputはViewノードであるべき
        let output = &graph.outputs[0];
        assert!(matches!(output.op, GraphOp::View(_)));

        // Viewのsourceは直接inputであるべき（中間のViewノードが統合された）
        if let GraphOp::View(source) = &output.op {
            assert!(matches!(source.op, GraphOp::Input(_)));
        }
    }

    #[test]
    fn test_fuse_simple_chain() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(10)]);

        // (a + b) * a を作成
        let add = a.clone() + b.clone();
        let mul = add * a.clone();

        graph.output(mul.clone());

        let mut optimizer = GraphFusionOptimizer::new();

        // 融合を試みる
        if let Some(fused) = optimizer.try_fuse_elementwise_chain(&mul) {
            // 融合されたノードがFusedElementwiseであることを確認
            assert!(matches!(fused.op, GraphOp::FusedElementwise(_, _)));

            if let GraphOp::FusedElementwise(_, ref inputs) = fused.op {
                // 入力は2つ (a, b)
                assert_eq!(inputs.len(), 2);
            }
        } else {
            panic!("Fusion should succeed");
        }
    }

    #[test]
    fn test_no_fusion_with_branching() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(10)]);

        // a + bを作成し、2箇所で使用（分岐）
        let add = a.clone() + b.clone();
        let mul1 = add.clone() * a.clone();
        let mul2 = add * b.clone();

        graph.output(mul1.clone());
        graph.output(mul2);

        let mut optimizer = GraphFusionOptimizer::new();

        // mul1の融合を試みる - addが分岐しているので、addは融合されない
        if let Some(fused) = optimizer.try_fuse_elementwise_chain(&mul1) {
            if let GraphOp::FusedElementwise(_, ref inputs) = fused.op {
                // 入力は (a+b) と a の2つのはず
                assert_eq!(inputs.len(), 2);
            }
        }
    }

    #[test]
    fn test_graph_optimization() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![Expr::from(10)]);
        let b = graph.input(DType::F32, vec![Expr::from(10)]);

        // (a + b) * (a + b) を作成
        let add1 = a.clone() + b.clone();
        let add2 = a + b;
        let mul = add1 * add2;

        graph.output(mul);

        let mut optimizer = GraphFusionOptimizer::new();
        optimizer.optimize(&mut graph);

        // 出力が更新されていることを確認
        assert_eq!(graph.outputs.len(), 1);
    }

    #[test]
    fn test_cast_chain_fusion() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![Expr::from(10)]);

        // F32 -> Isize -> Usize のCastチェーン
        let cast1 = input.cast(DType::Isize);
        let cast2 = cast1.cast(DType::Usize);

        graph.output(cast2.clone());

        let mut optimizer = GraphFusionOptimizer::new();

        // 融合を試みる
        if let Some(fused) = optimizer.try_fuse_cast_chain(&cast2) {
            // 融合されたノードがCastであることを確認
            assert!(matches!(fused.op, GraphOp::Cast(_, _)));

            if let GraphOp::Cast(ref source, ref dtype) = fused.op {
                // sourceは直接inputであるべき（中間のCastノードが統合された）
                assert!(matches!(source.op, GraphOp::Input(_)));
                // dtypeはUsizeであるべき
                assert_eq!(dtype, &DType::Usize);
            }
        } else {
            panic!("Fusion should succeed");
        }
    }

    #[test]
    fn test_no_cast_fusion_with_branching() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![Expr::from(10)]);

        // F32 -> IsizeのCastを作成し、2箇所で使用（分岐）
        let cast1 = input.cast(DType::Isize);
        let cast2 = cast1.clone().cast(DType::Usize);
        let cast3 = cast1.clone().cast(DType::F32);

        graph.output(cast2.clone());
        graph.output(cast3.clone());

        let mut optimizer = GraphFusionOptimizer::new();
        optimizer.optimize(&mut graph);

        // 最適化後も、両方の出力でcast1が残っているべき（分岐のため融合されない）
        // cast2の入力はCastノードであるべき
        let output1 = &graph.outputs[0];
        if let GraphOp::Cast(source, _) = &output1.op {
            // sourceはCast(input, Isize)であるべき
            assert!(matches!(source.op, GraphOp::Cast(_, _)));
        } else {
            panic!("Output should be a Cast node");
        }
    }

    #[test]
    fn test_cast_chain_optimization() {
        let mut graph = Graph::new();
        let input = graph.input(DType::F32, vec![Expr::from(10)]);

        // F32 -> Isize -> Usize -> F32 のCastチェーン
        let cast1 = input.cast(DType::Isize);
        let cast2 = cast1.cast(DType::Usize);
        let cast3 = cast2.cast(DType::F32);

        graph.output(cast3);

        let mut optimizer = GraphFusionOptimizer::new();
        optimizer.optimize(&mut graph);

        // 最適化後、outputはCastノードであるべき
        let output = &graph.outputs[0];
        assert!(matches!(output.op, GraphOp::Cast(_, _)));

        // Castのsourceは直接inputであるべき（中間のCastノードが統合された）
        if let GraphOp::Cast(source, dtype) = &output.op {
            assert!(matches!(source.op, GraphOp::Input(_)));
            assert_eq!(dtype, &DType::F32);
        }
    }
}

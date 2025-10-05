use crate::ast::AstNode;
use crate::graph::ops::ElementwiseOp;
use crate::graph::{Graph, GraphNode, GraphOp};
use crate::opt::graph::GraphOptimizer;
use std::collections::HashMap;

pub struct GraphFusionOptimizer {
    // 融合したノードのマッピング: 古いノード -> 新しいノード
    node_mapping: HashMap<GraphNode, GraphNode>,
}

impl GraphFusionOptimizer {
    pub fn new() -> Self {
        Self {
            node_mapping: HashMap::new(),
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
        let ast = self.elementwise_to_ast_with_branching_check(node, &mut input_mapping, &mut inputs)?;

        // 融合可能なノードが1つ以上ある場合のみ融合
        // (単一ノードの場合は融合する意味がない)
        if inputs.len() <= 1 {
            return None;
        }

        // 融合したノードを作成
        let fused_node = GraphNode::new(
            GraphOp::FusedElementwise(ast, inputs),
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
                let lhs_ast = self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast = self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(lhs_ast + rhs_ast)
            }
            ElementwiseOp::Mul(lhs, rhs) => {
                let lhs_ast = self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast = self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(lhs_ast * rhs_ast)
            }
            ElementwiseOp::Max(lhs, rhs) => {
                let lhs_ast = self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast = self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(AstNode::Max(Box::new(lhs_ast), Box::new(rhs_ast)))
            }
            ElementwiseOp::Mod(lhs, rhs) => {
                let lhs_ast = self.elementwise_to_ast_with_branching_check(lhs, input_mapping, inputs)?;
                let rhs_ast = self.elementwise_to_ast_with_branching_check(rhs, input_mapping, inputs)?;
                Some(lhs_ast % rhs_ast)
            }
            ElementwiseOp::Neg(input) => {
                let input_ast = self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(-input_ast)
            }
            ElementwiseOp::Recip(input) => {
                let input_ast = self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Recip(Box::new(input_ast)))
            }
            ElementwiseOp::Sin(input) => {
                let input_ast = self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Sin(Box::new(input_ast)))
            }
            ElementwiseOp::Sqrt(input) => {
                let input_ast = self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Sqrt(Box::new(input_ast)))
            }
            ElementwiseOp::Log2(input) => {
                let input_ast = self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Log2(Box::new(input_ast)))
            }
            ElementwiseOp::Exp2(input) => {
                let input_ast = self.elementwise_to_ast_with_branching_check(input, input_mapping, inputs)?;
                Some(AstNode::Exp2(Box::new(input_ast)))
            }
        }
    }

    /// Elementwise演算の入力ノードを取得
    fn get_elementwise_inputs(&self, node: &GraphNode) -> Vec<GraphNode> {
        match &node.op {
            GraphOp::Elementwise(ElementwiseOp::Add(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::Mul(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::Max(lhs, rhs))
            | GraphOp::Elementwise(ElementwiseOp::Mod(lhs, rhs)) => {
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
            _ => vec![],
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
        // 出力ノードから再帰的にグラフを再構築
        let new_outputs: Vec<GraphNode> = graph
            .outputs
            .iter()
            .map(|output| self.rebuild_node(output))
            .collect();

        graph.outputs = new_outputs;
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
            GraphOp::Input | GraphOp::Const(_) | GraphOp::Contiguous => {
                // これらのノードは依存がないのでそのまま返す
                node.clone()
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
                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Reduce(op.clone(), *axis, rebuilt_input),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::Cumulative(op, axis, input) => {
                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::Cumulative(op.clone(), *axis, rebuilt_input),
                    node.dtype.clone(),
                    node.view.clone(),
                )
            }
            GraphOp::View(input) => {
                let rebuilt_input = self.rebuild_node(input);
                GraphNode::new(
                    GraphOp::View(rebuilt_input),
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
}

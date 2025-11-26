use crate::ast::Literal;
use crate::graph::ops::ElementwiseOp;
use crate::graph::{Graph, GraphNode, GraphOp};
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// 定数伝播を提案するSuggester
///
/// Viewで拡張された定数をElementwise演算に融合することで、
/// より効率的なカーネルの生成を可能にします。
///
/// # 対象パターン
///
/// 以下のようなパターンを検出して最適化します：
/// ```ignore
/// const_value = Const(6.0)          // スカラー定数
/// expanded = View(const_value)      // Viewで拡張 (unsqueeze + expand)
/// result = matmul_result + expanded // Elementwise演算
/// ```
///
/// これを以下のように変換します：
/// ```ignore
/// result = matmul_result + Const(6.0)  // 定数を直接使用
/// // または
/// result = FusedElementwise([matmul_result], [AddScalar(6.0)])
/// ```
pub struct ConstPropagationSuggester;

impl ConstPropagationSuggester {
    /// 新しいConstPropagationSuggesterを作成
    pub fn new() -> Self {
        Self
    }

    /// グラフ内の全ノードを収集（トポロジカル順）
    fn collect_all_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const crate::graph::GraphNodeData>,
            nodes: &mut Vec<GraphNode>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in &node.src {
                visit(src, visited, nodes);
            }

            nodes.push(node.clone());
        }

        for output in graph.outputs().values() {
            visit(output, &mut visited, &mut nodes);
        }

        nodes
    }

    /// ノードが定数値を持つか確認し、定数値を返す
    ///
    /// 以下のパターンを検出：
    /// - Const(value): 直接定数
    /// - View(Const(value)): Viewで拡張された定数
    /// - View(View(Const(value))): 複数のViewを経由した定数
    /// - Elementwise(Const, Const): 全入力が定数のElementwise演算（定数畳み込み）
    fn extract_const_value(&self, node: &GraphNode) -> Option<Literal> {
        match &node.op {
            GraphOp::Const(lit) => Some(lit.clone()),
            GraphOp::View(_) => {
                // Viewの入力を再帰的にチェック
                if node.src.len() == 1 {
                    self.extract_const_value(&node.src[0])
                } else {
                    None
                }
            }
            GraphOp::Elementwise { op, .. } => {
                // 全入力が定数の場合、定数畳み込みを試みる
                let mut const_inputs = Vec::new();
                for src in &node.src {
                    if let Some(lit) = self.extract_const_value(src) {
                        const_inputs.push(lit);
                    } else {
                        // 1つでも非定数があれば、このノードは定数でない
                        return None;
                    }
                }

                // すべての入力が定数の場合、演算を評価
                if const_inputs.len() == node.src.len() {
                    self.evaluate_const_op(op, &const_inputs)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// 定数演算を評価（簡易的な定数畳み込み）
    fn evaluate_const_op(&self, op: &ElementwiseOp, inputs: &[Literal]) -> Option<Literal> {
        use crate::graph::ops::ElementwiseOp::*;

        // 単項演算
        if inputs.len() == 1 {
            return match op {
                Neg => Some(self.eval_unary(inputs[0].clone(), |x| -x)),
                Recip => Some(self.eval_unary(inputs[0].clone(), |x| 1.0 / x)),
                Sqrt => Some(self.eval_unary(inputs[0].clone(), |x| x.sqrt())),
                Log2 => Some(self.eval_unary(inputs[0].clone(), |x| x.log2())),
                Exp2 => Some(self.eval_unary(inputs[0].clone(), |x| x.exp2())),
                Sin => Some(self.eval_unary(inputs[0].clone(), |x| x.sin())),
                _ => None,
            };
        }

        // 二項演算
        if inputs.len() == 2 {
            return match op {
                Add => Some(self.eval_binary(inputs[0].clone(), inputs[1].clone(), |a, b| a + b)),
                Mul => Some(self.eval_binary(inputs[0].clone(), inputs[1].clone(), |a, b| a * b)),
                Max => {
                    Some(self.eval_binary(inputs[0].clone(), inputs[1].clone(), |a, b| a.max(b)))
                }
                Rem => Some(self.eval_binary(inputs[0].clone(), inputs[1].clone(), |a, b| a % b)),
                Idiv => Some(
                    self.eval_binary(inputs[0].clone(), inputs[1].clone(), |a, b| (a / b).floor()),
                ),
                _ => None,
            };
        }

        None
    }

    /// 単項演算を評価
    fn eval_unary<F>(&self, lit: Literal, f: F) -> Literal
    where
        F: Fn(f64) -> f64,
    {
        match lit {
            Literal::Bool(x) => Literal::Bool(f(if x { 1.0 } else { 0.0 }) != 0.0),
            Literal::F32(x) => Literal::F32(f(x as f64) as f32),
            Literal::Int(x) => Literal::Int(f(x as f64) as isize),
        }
    }

    /// 二項演算を評価
    fn eval_binary<F>(&self, a: Literal, b: Literal, f: F) -> Literal
    where
        F: Fn(f64, f64) -> f64,
    {
        match (a, b) {
            (Literal::Bool(x), Literal::Bool(y)) => {
                let xf = if x { 1.0 } else { 0.0 };
                let yf = if y { 1.0 } else { 0.0 };
                Literal::Bool(f(xf, yf) != 0.0)
            }
            (Literal::F32(x), Literal::F32(y)) => Literal::F32(f(x as f64, y as f64) as f32),
            (Literal::Int(x), Literal::Int(y)) => Literal::Int(f(x as f64, y as f64) as isize),
            // 型が一致しない場合は、F32に合わせる
            (Literal::F32(x), Literal::Int(y)) => Literal::F32(f(x as f64, y as f64) as f32),
            (Literal::Int(x), Literal::F32(y)) => Literal::F32(f(x as f64, y as f64) as f32),
            // Bool と他の型の混合はF32に合わせる
            (Literal::Bool(x), _) | (_, Literal::Bool(x)) => {
                let xf = if x { 1.0 } else { 0.0 };
                // このケースは通常発生しないが、安全のため
                Literal::F32(xf as f32)
            }
        }
    }

    /// Elementwise演算で定数入力を持つパターンを検出
    ///
    /// 戻り値: (元のノード, 定数でない入力のリスト, 定数値, 演算)
    fn detect_elementwise_with_const(
        &self,
        node: &GraphNode,
    ) -> Option<(Vec<GraphNode>, Literal, ElementwiseOp)> {
        // このノードがElementwiseでない場合はNone
        let op = match &node.op {
            GraphOp::Elementwise { op, .. } => op.clone(),
            _ => return None,
        };

        // 入力の中に定数があるか確認
        let mut non_const_inputs = Vec::new();
        let mut const_value = None;

        for src in &node.src {
            if let Some(lit) = self.extract_const_value(src) {
                // 既に定数が見つかっている場合は、複数の定数があるのでスキップ
                if const_value.is_some() {
                    return None;
                }
                const_value = Some(lit);
            } else {
                non_const_inputs.push(src.clone());
            }
        }

        // 定数が1つだけ見つかった場合
        if let Some(lit) = const_value
            && !non_const_inputs.is_empty()
        {
            return Some((non_const_inputs, lit, op));
        }

        None
    }

    /// 定数を使用する演算を最適化した新しいノードを作成
    ///
    /// 定数を直接FusedElementwiseノードに埋め込むことで、
    /// ConstノードとViewノードを完全に削除します。
    fn create_optimized_node(
        &self,
        non_const_inputs: Vec<GraphNode>,
        const_value: Literal,
        op: ElementwiseOp,
        original_view: crate::graph::View,
        original_dtype: crate::graph::DType,
    ) -> GraphNode {
        use crate::ast::{AstNode, Literal as AstLiteral, helper::wildcard};

        // 非定数入力をWildcardとして参照
        let non_const_args: Vec<AstNode> = (0..non_const_inputs.len())
            .map(|i| wildcard(i.to_string()))
            .collect();

        // 定数値をAstNodeに変換
        let const_ast = match const_value {
            Literal::Bool(v) => AstNode::Const(AstLiteral::Bool(v)),
            Literal::F32(v) => AstNode::Const(AstLiteral::F32(v)),
            Literal::Int(v) => AstNode::Const(AstLiteral::Int(v)),
        };

        // ElementwiseOpに基づいてAstNode式を構築
        let expr = match op {
            ElementwiseOp::Add => {
                assert_eq!(non_const_args.len(), 1);
                non_const_args[0].clone() + const_ast
            }
            ElementwiseOp::Mul => {
                assert_eq!(non_const_args.len(), 1);
                non_const_args[0].clone() * const_ast
            }
            ElementwiseOp::Max => {
                assert_eq!(non_const_args.len(), 1);
                AstNode::Max(Box::new(non_const_args[0].clone()), Box::new(const_ast))
            }
            ElementwiseOp::Rem => {
                assert_eq!(non_const_args.len(), 1);
                non_const_args[0].clone() % const_ast
            }
            ElementwiseOp::Idiv => {
                assert_eq!(non_const_args.len(), 1);
                non_const_args[0].clone() / const_ast
            }
            ElementwiseOp::Neg => {
                // Negは単項演算なので定数は使用されないはず
                assert_eq!(non_const_args.len(), 1);
                -non_const_args[0].clone()
            }
            ElementwiseOp::Recip => {
                assert_eq!(non_const_args.len(), 1);
                AstNode::Recip(Box::new(non_const_args[0].clone()))
            }
            ElementwiseOp::Log2 => {
                assert_eq!(non_const_args.len(), 1);
                AstNode::Log2(Box::new(non_const_args[0].clone()))
            }
            ElementwiseOp::Exp2 => {
                assert_eq!(non_const_args.len(), 1);
                AstNode::Exp2(Box::new(non_const_args[0].clone()))
            }
            ElementwiseOp::Sin => {
                assert_eq!(non_const_args.len(), 1);
                AstNode::Sin(Box::new(non_const_args[0].clone()))
            }
            ElementwiseOp::Sqrt => {
                assert_eq!(non_const_args.len(), 1);
                AstNode::Sqrt(Box::new(non_const_args[0].clone()))
            }
        };

        // FusedElementwiseノードを作成
        GraphNode::new(
            original_dtype,
            GraphOp::FusedElementwise {
                expr,
                elementwise_strategies: None,
            },
            non_const_inputs,
            original_view,
        )
    }

    /// グラフ内の特定ノードを置き換えた新しいグラフを作成
    fn replace_node_in_graph(
        &self,
        graph: &Graph,
        old_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut node_map: HashMap<*const crate::graph::GraphNodeData, GraphNode> = HashMap::new();
        node_map.insert(old_node.as_ptr(), new_node.clone());

        let mut visited = HashSet::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const crate::graph::GraphNodeData, GraphNode>,
            visited: &mut HashSet<*const crate::graph::GraphNodeData>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // Inputノードは常に元のノードをそのまま返す
            if matches!(node.op, GraphOp::Input) {
                return node.clone();
            }

            if let Some(new_node) = node_map.get(&ptr) {
                return new_node.clone();
            }

            if visited.contains(&ptr) {
                return node.clone();
            }
            visited.insert(ptr);

            let new_src: Vec<GraphNode> = node
                .src
                .iter()
                .map(|src| rebuild_node(src, node_map, visited))
                .collect();

            let src_changed = new_src
                .iter()
                .zip(&node.src)
                .any(|(a, b)| a.as_ptr() != b.as_ptr());

            if !src_changed {
                return node.clone();
            }

            GraphNode::with_elementwise_strategies(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
                node.elementwise_strategies.clone(),
            )
        }

        let mut new_graph = Graph::new();

        // 入力ノードを保持
        for (name, weak_input) in graph.inputs() {
            if let Some(rc_node) = weak_input.upgrade() {
                let input_node = GraphNode::from_rc(rc_node);
                new_graph.register_input(name.clone(), input_node);
            }
        }

        // 出力ノードを名前順でソートして再構築
        let mut outputs: Vec<_> = graph.outputs().iter().collect();
        outputs.sort_by_key(|(name, _)| name.as_str());

        for (name, output_node) in outputs {
            let rebuilt = rebuild_node(output_node, &node_map, &mut visited);
            new_graph.output(name, rebuilt);
        }

        new_graph
    }
}

impl Default for ConstPropagationSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphSuggester for ConstPropagationSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let nodes = self.collect_all_nodes(graph);

        for node in &nodes {
            // Elementwise演算で定数入力を持つパターンを検出
            if let Some((non_const_inputs, const_value, op)) =
                self.detect_elementwise_with_const(node)
            {
                let optimized_node = self.create_optimized_node(
                    non_const_inputs,
                    const_value,
                    op,
                    node.view.clone(),
                    node.dtype.clone(),
                );

                let new_graph = self.replace_node_in_graph(graph, node, optimized_node);
                suggestions.push(new_graph);
            }
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    #[test]
    fn test_const_propagation_basic() {
        let suggester = ConstPropagationSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // スカラー定数は自動的にブロードキャストされる
        // a + const
        let result = a + 5.0f32;
        graph.output("result", result);

        let suggestions = suggester.suggest(&graph);

        // 定数伝播が検出され、最適化候補が生成されるはず
        assert!(
            !suggestions.is_empty(),
            "Expected const propagation suggestions"
        );
    }

    #[test]
    fn test_const_propagation_no_pattern() {
        let suggester = ConstPropagationSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        let b = graph
            .input("b")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // 定数がない場合
        let result = a + b;
        graph.output("result", result);

        let suggestions = suggester.suggest(&graph);

        // 定数がないので、候補は生成されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_const_propagation_multiple_ops() {
        let suggester = ConstPropagationSuggester::new();

        let mut graph = Graph::new();
        let a = graph
            .input("a")
            .with_dtype(DType::F32)
            .with_shape(vec![10, 20])
            .build();

        // 定数演算（スカラー定数は自動的にブロードキャストされる）
        // 2.0 * 3.0 = 6.0
        let const1: GraphNode = 2.0f32.into();
        let const2: GraphNode = 3.0f32.into();
        let scale = const1 * const2;

        // a + scale
        let result = a + scale;
        graph.output("result", result);

        let suggestions = suggester.suggest(&graph);

        // 定数伝播が検出されるはず
        assert!(
            !suggestions.is_empty(),
            "Expected const propagation suggestions for folded constant"
        );
    }
}

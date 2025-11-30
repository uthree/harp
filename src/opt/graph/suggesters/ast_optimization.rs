//! AST Optimization Suggester
//!
//! CustomノードのASTに対してAstSuggesterを適用し、
//! グラフ最適化の枠組みでAST最適化を行うラッパーです。
//!
//! これにより、グラフ変換とAST変換を単一のビームサーチで探索でき、
//! 相互作用を発見できます。

use crate::ast::AstNode;
use crate::graph::{Graph, GraphNode, GraphNodeData, GraphOp};
use crate::opt::ast::Suggester as AstSuggester;
use crate::opt::graph::GraphSuggester;
use std::collections::{HashMap, HashSet};

/// CustomノードのASTに対してAstSuggesterを適用するGraphSuggester
///
/// グラフ最適化の枠組みでAST最適化を行うラッパーです。
/// lowering後のCustomノードに対してのみ有効です。
pub struct AstOptimizationSuggester {
    /// 使用するAstSuggesterのリスト
    ast_suggesters: Vec<Box<dyn AstSuggester>>,
    /// 各Customノードあたりの最大提案数
    max_suggestions_per_node: usize,
}

impl AstOptimizationSuggester {
    /// 新しいAstOptimizationSuggesterを作成
    pub fn new(ast_suggesters: Vec<Box<dyn AstSuggester>>) -> Self {
        Self {
            ast_suggesters,
            max_suggestions_per_node: 3, // デフォルト: ノードあたり3つまで
        }
    }

    /// 各Customノードあたりの最大提案数を設定
    pub fn with_max_suggestions_per_node(mut self, max: usize) -> Self {
        self.max_suggestions_per_node = max;
        self
    }

    /// グラフ内の全Customノードを収集
    fn collect_custom_nodes(&self, graph: &Graph) -> Vec<GraphNode> {
        let mut visited = HashSet::new();
        let mut custom_nodes = Vec::new();

        fn visit(
            node: &GraphNode,
            visited: &mut HashSet<*const GraphNodeData>,
            custom_nodes: &mut Vec<GraphNode>,
        ) {
            let ptr = node.as_ptr();
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            for src in &node.src {
                visit(src, visited, custom_nodes);
            }

            if matches!(node.op, GraphOp::Custom { .. }) {
                custom_nodes.push(node.clone());
            }
        }

        for output in graph.outputs().values() {
            visit(output, &mut visited, &mut custom_nodes);
        }

        custom_nodes
    }

    /// CustomノードのASTにSuggesterを適用
    fn apply_suggesters_to_ast(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut suggestions = Vec::new();

        for suggester in &self.ast_suggesters {
            let new_suggestions = suggester.suggest(ast);
            for suggestion in new_suggestions {
                // 元のASTと異なる場合のみ追加
                if suggestion != *ast {
                    suggestions.push(suggestion);
                }
            }
        }

        // 提案数を制限
        suggestions.truncate(self.max_suggestions_per_node);
        suggestions
    }

    /// グラフ内のノードを置き換えた新しいグラフを作成
    fn replace_node_in_graph(
        &self,
        graph: &Graph,
        old_node: &GraphNode,
        new_node: GraphNode,
    ) -> Graph {
        let mut node_map: HashMap<*const GraphNodeData, GraphNode> = HashMap::new();
        node_map.insert(old_node.as_ptr(), new_node);

        let mut visited = HashSet::new();

        fn rebuild_node(
            node: &GraphNode,
            node_map: &HashMap<*const GraphNodeData, GraphNode>,
            visited: &mut HashSet<*const GraphNodeData>,
        ) -> GraphNode {
            let ptr = node.as_ptr();

            // Inputノードは常に元のノードをそのまま返す
            if matches!(node.op, GraphOp::Buffer { .. }) {
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

            GraphNode::new(
                node.dtype.clone(),
                node.op.clone(),
                new_src,
                node.view.clone(),
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

        // 出力ノードを再構築
        for (name, output) in graph.outputs() {
            let new_output = rebuild_node(&output, &node_map, &mut visited);
            new_graph.output(&name, new_output);
        }

        // shape変数のデフォルト値をコピー
        for (var_name, default_value) in graph.shape_var_defaults() {
            new_graph.set_shape_var_default(var_name.clone(), *default_value);
        }

        new_graph
    }
}

impl Default for AstOptimizationSuggester {
    fn default() -> Self {
        Self::new(vec![])
    }
}

impl GraphSuggester for AstOptimizationSuggester {
    fn name(&self) -> &'static str {
        "AstOptimization"
    }

    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut suggestions = Vec::new();
        let custom_nodes = self.collect_custom_nodes(graph);

        if custom_nodes.is_empty() {
            return suggestions;
        }

        log::debug!(
            "AstOptimizationSuggester: found {} Custom nodes",
            custom_nodes.len()
        );

        for node in &custom_nodes {
            if let GraphOp::Custom { ast } = &node.op {
                // AstSuggesterを適用
                let ast_suggestions = self.apply_suggesters_to_ast(ast);

                for new_ast in ast_suggestions {
                    // 新しいCustomノードを作成
                    let new_node = GraphNode::new(
                        node.dtype.clone(),
                        GraphOp::Custom { ast: new_ast },
                        node.src.clone(),
                        node.view.clone(),
                    );

                    // グラフを更新
                    let new_graph = self.replace_node_in_graph(graph, node, new_node);
                    suggestions.push(new_graph);
                }
            }
        }

        if !suggestions.is_empty() {
            log::debug!(
                "AstOptimizationSuggester: generated {} suggestions",
                suggestions.len()
            );
        }

        suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::wildcard;
    use crate::graph::DType;
    use crate::opt::ast::RuleBaseSuggester;
    use crate::opt::ast::rules::all_rules_with_search;

    #[test]
    fn test_ast_optimization_suggester_basic() {
        let _ = env_logger::try_init();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // Customノードを作成（0 + x -> x に簡約可能）
        // 0 + wildcard("0") という式を持つCustomノード
        use crate::ast::helper::const_f32;
        let expr = const_f32(0.0) + wildcard("0");
        let custom = a.custom_elementwise_binary(b, expr);
        graph.output("result", custom);

        // AstOptimizationSuggesterを作成
        let rules = all_rules_with_search();
        let ast_suggester = RuleBaseSuggester::new(rules);
        let suggester = AstOptimizationSuggester::new(vec![Box::new(ast_suggester)]);

        // Customノードが見つかることを確認
        let custom_nodes = suggester.collect_custom_nodes(&graph);
        assert_eq!(custom_nodes.len(), 1, "Should find 1 Custom node");

        // 提案を生成
        let suggestions = suggester.suggest(&graph);
        println!("Generated {} suggestions", suggestions.len());

        // 0 + x -> x の簡約が提案されることを期待
        // （ただしRuleBaseSuggesterが適切なルールを持っている場合）
    }

    #[test]
    fn test_ast_optimization_with_beam_search() {
        use crate::backend::pipeline::{SuggesterFlags, create_graph_suggester};
        use crate::opt::graph::{
            BeamSearchGraphOptimizer, CompositeSuggester, GraphOptimizer, SimpleCostEstimator,
        };

        let _ = env_logger::try_init();

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // 簡単な演算
        let result = &a + &b;
        graph.output("result", result);

        // AstOptimizationSuggesterを含むSuggesterを作成
        let base_suggester = create_graph_suggester(SuggesterFlags::single_stage());

        // RuleBaseSuggesterを作成
        let rules = all_rules_with_search();
        let ast_suggester = RuleBaseSuggester::new(rules);
        let ast_opt_suggester = AstOptimizationSuggester::new(vec![Box::new(ast_suggester)]);

        // CompositeSuggesterに追加
        let combined_suggester =
            CompositeSuggester::new(vec![Box::new(base_suggester), Box::new(ast_opt_suggester)]);

        let estimator = SimpleCostEstimator::new();
        let optimizer = BeamSearchGraphOptimizer::new(combined_suggester, estimator)
            .with_beam_width(4)
            .with_max_steps(50);

        let optimized = optimizer.optimize(graph);

        // 最適化が完了することを確認
        assert!(
            !optimized.outputs().is_empty(),
            "Optimized graph should have outputs"
        );
    }

    #[test]
    fn test_no_suggestions_without_custom_nodes() {
        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        let b = graph.input("b", DType::F32, vec![10]);

        // Customノードなし（通常のAdd）
        let result = &a + &b;
        graph.output("result", result);

        let rules = all_rules_with_search();
        let ast_suggester = RuleBaseSuggester::new(rules);
        let suggester = AstOptimizationSuggester::new(vec![Box::new(ast_suggester)]);

        let suggestions = suggester.suggest(&graph);
        assert!(
            suggestions.is_empty(),
            "Should not suggest anything without Custom nodes"
        );
    }
}

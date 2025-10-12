use crate::graph::{Graph, GraphNode};
use crate::opt::graph::GraphOptimizer;
use std::collections::HashMap;

mod elementwise;
mod rebuild;
mod reduce;
mod view_cast;

pub struct GraphFusionOptimizer {
    // 融合したノードのマッピング: 古いノード -> 新しいノード
    pub(crate) node_mapping: HashMap<GraphNode, GraphNode>,
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

    pub(crate) fn log_snapshot(&mut self, description: String, graph: &Graph) {
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
    pub(crate) fn is_branching(&self, node: &GraphNode) -> bool {
        // strong_countで参照数を取得
        // 1つはnode自身、もう1つは親ノードからの参照
        // それ以上あれば分岐している
        node.strong_count() > 2
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::Expr;
    use crate::graph::GraphOp;

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

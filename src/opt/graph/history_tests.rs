//! 最適化履歴管理のテスト

#[cfg(test)]
mod tests {
    use crate::graph::{DType, Graph, GraphNode};
    use crate::opt::graph::history::OptimizationHistory;

    #[test]
    fn test_history_creation() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();

        let history = OptimizationHistory::new(graph.clone());
        assert_eq!(history.steps().len(), 0);
    }

    #[test]
    fn test_history_add_step() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        graph.output("y", x);

        let mut history = OptimizationHistory::new(graph.clone());

        // ステップを追加
        history.add_step("Test optimization", graph.clone(), 100.0);
        assert_eq!(history.steps().len(), 1);

        // 2つ目のステップを追加
        history.add_step("Second optimization", graph.clone(), 90.0);
        assert_eq!(history.steps().len(), 2);
    }

    #[test]
    fn test_history_best_graph() {
        let mut graph = Graph::new();
        let x = graph
            .input("x")
            .with_dtype(DType::F32)
            .with_shape(vec![10])
            .build();
        graph.output("y", x);

        let mut history = OptimizationHistory::new(graph.clone());

        // 最初のステップ（コスト100）
        history.add_step("First", graph.clone(), 100.0);

        // より良いステップ（コスト50）
        history.add_step("Better", graph.clone(), 50.0);

        // 悪いステップ（コスト150）
        history.add_step("Worse", graph.clone(), 150.0);

        // 最良のグラフを取得
        let best = history.best_graph();
        assert!(best.is_some());
    }
}

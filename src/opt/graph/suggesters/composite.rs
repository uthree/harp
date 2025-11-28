use crate::graph::Graph;
use crate::opt::graph::GraphSuggester;

/// 複数のSuggesterを組み合わせるSuggester
pub struct CompositeSuggester {
    suggesters: Vec<Box<dyn GraphSuggester>>,
}

impl CompositeSuggester {
    /// 新しいCompositeSuggesterを作成
    pub fn new(suggesters: Vec<Box<dyn GraphSuggester>>) -> Self {
        Self { suggesters }
    }

    /// Suggesterを追加
    pub fn add_suggester(&mut self, suggester: Box<dyn GraphSuggester>) {
        self.suggesters.push(suggester);
    }
}

impl GraphSuggester for CompositeSuggester {
    fn suggest(&self, graph: &Graph) -> Vec<Graph> {
        let mut all_suggestions = Vec::new();

        // 各Suggesterから候補を収集
        for suggester in &self.suggesters {
            let suggestions = suggester.suggest(graph);
            all_suggestions.extend(suggestions);
        }

        all_suggestions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{DType, Graph};

    // テスト用のダミーSuggester
    struct DummySuggester1;
    struct DummySuggester2;

    impl GraphSuggester for DummySuggester1 {
        fn suggest(&self, _graph: &Graph) -> Vec<Graph> {
            vec![Graph::new()] // 1つの候補を返す
        }
    }

    impl GraphSuggester for DummySuggester2 {
        fn suggest(&self, _graph: &Graph) -> Vec<Graph> {
            vec![Graph::new(), Graph::new()] // 2つの候補を返す
        }
    }

    #[test]
    fn test_composite_suggester() {
        let composite =
            CompositeSuggester::new(vec![Box::new(DummySuggester1), Box::new(DummySuggester2)]);

        let mut graph = Graph::new();
        let a = graph.input("a", DType::F32, vec![10]);
        graph.output("a", a);

        let suggestions = composite.suggest(&graph);
        // 1 + 2 = 3つの候補が得られるはず
        assert_eq!(suggestions.len(), 3);
    }
}

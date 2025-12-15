use crate::graph::Graph;
use crate::opt::graph::{GraphSuggester, SuggestResult};

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
    fn name(&self) -> &'static str {
        "Composite"
    }

    fn suggest(&self, graph: &Graph) -> Vec<SuggestResult> {
        let mut all_results = Vec::new();

        // 各Suggesterから候補を収集（それぞれのSuggester名を保持）
        for suggester in &self.suggesters {
            log::trace!(
                "CompositeSuggester: calling suggest for '{}'",
                suggester.name()
            );
            let results = suggester.suggest(graph);
            log::trace!(
                "CompositeSuggester: suggester '{}' returned {} results",
                suggester.name(),
                results.len()
            );
            all_results.extend(results);
        }

        all_results
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
        fn name(&self) -> &'static str {
            "Dummy1"
        }

        fn suggest(&self, _graph: &Graph) -> Vec<SuggestResult> {
            vec![SuggestResult::with_description(
                Graph::new(),
                "Dummy1",
                "dummy suggestion 1",
            )]
        }
    }

    impl GraphSuggester for DummySuggester2 {
        fn name(&self) -> &'static str {
            "Dummy2"
        }

        fn suggest(&self, _graph: &Graph) -> Vec<SuggestResult> {
            vec![
                SuggestResult::with_description(Graph::new(), "Dummy2", "dummy suggestion 2a"),
                SuggestResult::with_description(Graph::new(), "Dummy2", "dummy suggestion 2b"),
            ]
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

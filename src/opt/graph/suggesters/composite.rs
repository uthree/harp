//! Composite suggester that combines multiple suggesters

use super::super::{GraphSuggestResult, GraphSuggester};
use crate::graph::GraphNode;

/// A suggester that combines multiple suggesters
///
/// Returns suggestions from all contained suggesters.
pub struct CompositeSuggester {
    suggesters: Vec<Box<dyn GraphSuggester>>,
}

impl CompositeSuggester {
    /// Create a new composite suggester
    pub fn new(suggesters: Vec<Box<dyn GraphSuggester>>) -> Self {
        Self { suggesters }
    }
}

impl GraphSuggester for CompositeSuggester {
    fn name(&self) -> &str {
        "composite"
    }

    fn suggest(&self, roots: &[GraphNode]) -> Vec<GraphSuggestResult> {
        let mut results = Vec::new();
        for suggester in &self.suggesters {
            results.extend(suggester.suggest(roots));
        }
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummySuggester {
        name: String,
    }

    impl GraphSuggester for DummySuggester {
        fn name(&self) -> &str {
            &self.name
        }

        fn suggest(&self, _roots: &[GraphNode]) -> Vec<GraphSuggestResult> {
            vec![]
        }
    }

    #[test]
    fn test_composite_empty() {
        let suggester = CompositeSuggester::new(vec![]);
        assert!(suggester.suggest(&[]).is_empty());
    }

    #[test]
    fn test_composite_combines() {
        let suggester = CompositeSuggester::new(vec![
            Box::new(DummySuggester {
                name: "a".to_string(),
            }),
            Box::new(DummySuggester {
                name: "b".to_string(),
            }),
        ]);
        // Both suggesters return empty, so composite returns empty
        assert!(suggester.suggest(&[]).is_empty());
    }
}

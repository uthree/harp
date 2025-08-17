use crate::graph::pattern::{RewriteRule, Rewriter, Suggester};
use crate::graph::{Graph, GraphNode};

pub fn rules() -> Vec<Box<dyn Suggester>> {
    vec![
        Box::new(RewriteRule::new(
            "double_negation",
            -(-GraphNode::capture(0)),
            GraphNode::capture(0),
        )),
    ]
}

pub fn optimize(graph: Graph) -> Graph {
    let mut rewriter = Rewriter::new();
    for rule in rules() {
        rewriter.add_suggester(rule);
    }
    rewriter.rewrite(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::shape::expr::Expr as ShapeExpr;
    use crate::ast::DType;

    #[test]
    fn test_double_negation_optimization() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = -(-a.clone());
        graph.outputs.push(b);

        let optimized_graph = optimize(graph);

        assert_eq!(optimized_graph.outputs.len(), 1);
        // The output should be the original input node 'a'
        assert_eq!(optimized_graph.outputs[0], a);
    }
}
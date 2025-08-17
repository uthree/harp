use crate::graph::Graph;
use crate::graph::pattern::GraphRewriter;
use crate::{graph_rewriter, graphpat};

pub fn algebraic_simplification() -> GraphRewriter {
    graph_rewriter!("algebraic_simplification", graphpat!(|a| -(-a) => a))
}

pub fn optimize(graph: Graph) -> Graph {
    let rewriter = algebraic_simplification();
    rewriter.rewrite(graph)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::shape::expr::Expr as ShapeExpr;

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

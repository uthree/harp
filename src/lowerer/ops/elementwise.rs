use crate::{
    ast::{AstNode, AstOp},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_elementwise(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
) -> Vec<AstNode> {
    let mut stmts = vec![];
    let mut lowered_srcs = vec![];
    for src in &node.src {
        let mut lowered = lowerer.lower_node_rec(src, indices, inputs, None);
        let val = lowered.pop().unwrap();
        stmts.extend(lowered);
        lowered_srcs.push(val);
    }
    stmts.push(AstNode::_new(op.clone(), lowered_srcs, node.dtype.clone()));
    stmts
}

#[cfg(test)]
mod tests {
    use crate::{
        ast::{AstNode, AstOp, DType},
        graph::{Graph, shape::expr::Expr as ShapeExpr},
        lowerer::lower_graph,
    };

    #[test]
    fn test_lower_simple_add() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = a + b;
        graph.outputs.push(c);
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        });

        let ast = lower_graph(&graph);

        // Expect a Program node with two functions: kernel_impl and kernel_main
        if let AstNode {
            op: AstOp::Program,
            src,
            ..
        } = &ast
        {
            assert_eq!(src.len(), 2);

            // Check kernel_impl
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(0)
            {
                assert_eq!(name, "kernel_impl");
                assert_eq!(args.len(), 3); // 1 output, 2 inputs
                assert_eq!(args[0].0, "output0");
                assert_eq!(args[1].0, "input0");
                assert_eq!(args[2].0, "input1");
            } else {
                panic!("Expected kernel_impl function");
            }

            // Check kernel_main
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(1)
            {
                assert_eq!(name, "kernel_main");
                assert_eq!(args.len(), 2);
                assert_eq!(args[0].0, "bufs");
                assert_eq!(args[1].0, "shape_vars");
            } else {
                panic!("Expected kernel_main function");
            }
        } else {
            panic!("Expected a program AST node, got {:?}", ast);
        }
    }

    #[test]
    fn test_lower_chained_elementwise() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = graph.add_input(shape.clone(), &dtype);
        let c = graph.add_input(shape.clone(), &dtype);
        let d = (a + b) * c;
        graph.outputs.push(d);
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype: dtype.clone(),
            shape: shape.clone(),
        });

        use crate::opt::graph::GraphOptimizer;
        use crate::opt::graph::fusion::ElementwiseFusion;
        let optimizer = ElementwiseFusion;
        let graph = optimizer.optimize(&graph);

        let ast = lower_graph(&graph);

        // Expect a Program node with two functions: kernel_impl and kernel_main
        if let AstNode {
            op: AstOp::Program,
            src,
            ..
        } = &ast
        {
            assert_eq!(src.len(), 2);

            // Check kernel_impl
            if let Some(AstNode {
                op: AstOp::Func { name, args, .. },
                ..
            }) = src.get(0)
            {
                assert_eq!(name, "kernel_impl");
                assert_eq!(args.len(), 4); // 1 output, 3 inputs
                assert_eq!(args[0].0, "output0");
                assert_eq!(args[1].0, "input0");
                assert_eq!(args[2].0, "input1");
                assert_eq!(args[3].0, "input2");
            } else {
                panic!("Expected kernel_impl function");
            }
        } else {
            panic!("Expected a program AST node, got {:?}", ast);
        }
    }
}

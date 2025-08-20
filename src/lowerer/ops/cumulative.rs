use crate::{
    ast::{AstNode, AstOp, DType},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_cumulative(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
    axis: &usize,
) -> Vec<AstNode> {
    // Create an accumulator variable, initialized to the identity of the op.
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());

    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported cumulative operation: {:?}", op),
    };
    let declare_acc = AstNode::declare(&acc_name, node.dtype.clone(), Some(init_val));

    // Create an inner loop to accumulate values from the source.
    // The loop runs from 0 up to the current index along the cumulative axis.
    let cum_limit = indices[*axis].clone() + AstNode::from(1isize);
    let cidx_name = format!("cidx{}", lowerer.ridx_counter);
    lowerer.ridx_counter += 1;
    let cidx_var = AstNode::var(&cidx_name, DType::Isize);

    let mut inner_indices = indices.to_vec();
    inner_indices[*axis] = cidx_var;

    let mut lowered_src = lowerer.lower_node_rec(&node.src[0], &mut inner_indices, inputs, None);
    let value_to_accumulate = lowered_src.pop().unwrap();

    let update_op = AstNode::_new(
        op.clone(),
        vec![acc_var.clone(), value_to_accumulate],
        node.dtype.clone(),
    );
    let assign_op = AstNode::assign(acc_var.clone(), update_op);
    let loop_body = AstNode::block(
        lowered_src
            .into_iter()
            .chain(std::iter::once(assign_op))
            .collect(),
    );

    let loop_node = AstNode::range(&cidx_name, 1, AstNode::from(0isize), cum_limit, loop_body);

    let mut stmts = vec![declare_acc];
    stmts.push(loop_node);
    stmts.push(acc_var);
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
    fn test_lower_cumulative_sum() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.cumulative(AstOp::Add, 0);
        graph.outputs.push(b.clone());
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype,
            shape: b.shape().to_vec(),
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
                assert_eq!(args.len(), 2); // 1 output, 1 input
                assert_eq!(args[0].0, "output0");
                assert_eq!(args[1].0, "input0");
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
            } else {
                panic!("Expected kernel_main function");
            }
        } else {
            panic!("Expected a program AST node, got {:?}", ast);
        }
    }

    #[test]
    fn test_lower_cumulative_max() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.cumulative(AstOp::Max, 0);
        graph.outputs.push(b.clone());
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype,
            shape: b.shape().to_vec(),
        });

        let ast = lower_graph(&graph);
        assert!(matches!(ast.op, AstOp::Program));
    }
}

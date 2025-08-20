use crate::{
    ast::{AstNode, AstOp, DType},
    graph::GraphNode,
    lowerer::Lowerer,
};

pub fn lower_reduce(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
    axis: &usize,
) -> Vec<AstNode> {
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());
    lowerer
        .declarations
        .push(AstNode::declare(&acc_name, node.dtype.clone()));

    // Initialize accumulator.
    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported reduction operation: {:?}", op),
    };
    let init_acc = AstNode::assign(acc_var.clone(), init_val);

    // Create reduction loop.
    let reduce_dim = node.src[0].shape()[*axis].clone();
    let ridx_name = format!("ridx{}", lowerer.ridx_counter);
    lowerer.ridx_counter += 1;
    let ridx_var = AstNode::var(&ridx_name, DType::Isize);

    let mut inner_indices = indices.to_vec();
    // Insert the reduction loop variable at the reduction axis.
    inner_indices.insert(*axis, ridx_var);

    let mut lowered_src = lowerer.lower_node_rec(&node.src[0], &mut inner_indices, inputs, None);
    let value_to_reduce = lowered_src.pop().unwrap();

    let update_op = AstNode::_new(
        op.clone(),
        vec![acc_var.clone(), value_to_reduce],
        node.dtype.clone(),
    );
    let assign_op = AstNode::assign(acc_var.clone(), update_op);
    let loop_body = AstNode::block(
        lowered_src
            .into_iter()
            .chain(std::iter::once(assign_op))
            .collect(),
    );

    let loop_node = AstNode::range(
        &ridx_name,
        1,
        AstNode::from(0isize),
        reduce_dim.into(),
        loop_body,
    );

    let mut stmts = vec![init_acc];
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
    fn test_lower_reduce_sum() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10), ShapeExpr::from(20)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.reduce(AstOp::Add, 1); // Reduce along axis 1
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
    fn test_lower_reduce_max() {
        let mut graph = Graph::new();
        let shape = vec![ShapeExpr::from(10)];
        let dtype = DType::F32;

        let a = graph.add_input(shape.clone(), &dtype);
        let b = a.reduce(AstOp::Max, 0);
        graph.outputs.push(b.clone());
        graph.signature.outputs.push(crate::graph::TensorSignature {
            dtype,
            shape: b.shape().to_vec(),
        });

        let ast = lower_graph(&graph);
        assert!(matches!(ast.op, AstOp::Program));
    }
}

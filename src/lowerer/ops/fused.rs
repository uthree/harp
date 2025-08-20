use crate::{
    ast::{AstNode, AstOp},
    graph::{GraphNode, shape::view::View},
    lowerer::Lowerer,
};

pub fn lower_fused_elementwise(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    fused_ast: &AstNode,
) -> Vec<AstNode> {
    replace_captures_rec(lowerer, node, indices, inputs, fused_ast)
}

pub fn lower_fused_elementwise_reduce(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    fused_ast: &AstNode,
    op: &AstOp,
    axes: &[usize],
    output_ptr: AstNode,
) -> Vec<AstNode> {
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());
    lowerer
        .declarations
        .push(AstNode::declare(&acc_name, node.dtype.clone()));

    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported reduction operation: {:?}", op),
    };
    let init_acc = AstNode::assign(acc_var.clone(), init_val);

    let mut reduce_vars = vec![];
    let mut sorted_axes = axes.to_vec();
    sorted_axes.sort();

    let mut body_indices = indices.to_vec();
    for axis in &sorted_axes {
        body_indices.insert(*axis, AstNode::var("dummy", node.dtype.clone()));
    }

    for (i, axis) in sorted_axes.iter().enumerate() {
        let ridx_name = format!("ridx{}_{}", lowerer.ridx_counter, i);
        let ridx_var = AstNode::var(&ridx_name, node.dtype.clone());
        reduce_vars.push((ridx_name, node.src[0].shape()[*axis].clone()));
        body_indices[*axis] = ridx_var;
    }
    lowerer.ridx_counter += 1;

    let mut lowered_fused =
        replace_captures_rec(lowerer, node, &mut body_indices, inputs, fused_ast);
    let value_to_reduce = lowered_fused.pop().unwrap();
    let update_op = AstNode::_new(
        op.clone(),
        vec![acc_var.clone(), value_to_reduce],
        node.dtype.clone(),
    );
    let mut body = AstNode::assign(acc_var.clone(), update_op);
    body = AstNode::block(
        lowered_fused
            .into_iter()
            .chain(std::iter::once(body))
            .collect(),
    );

    for (ridx_name, reduce_dim) in reduce_vars.into_iter().rev() {
        body = AstNode::range(
            &ridx_name,
            1,
            AstNode::from(0isize),
            reduce_dim.into(),
            body,
        );
    }

    let mut stmts = vec![init_acc];
    stmts.push(body);

    let output_view = View::new_contiguous(node.shape().to_vec());
    let physical_index = output_view.to_physical_index_ast(indices);
    let store_ptr = AstNode::index(output_ptr, physical_index);
    stmts.push(AstNode::store(store_ptr, acc_var));

    stmts
}

pub fn lower_fused_reduce(
    lowerer: &mut Lowerer,
    node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    op: &AstOp,
    axes: &[usize],
    output_ptr: AstNode,
) -> Vec<AstNode> {
    let acc_name = format!("acc{}", lowerer.acc_counter);
    lowerer.acc_counter += 1;
    let acc_var = AstNode::var(&acc_name, node.dtype.clone());
    lowerer
        .declarations
        .push(AstNode::declare(&acc_name, node.dtype.clone()));

    let init_val = match op {
        AstOp::Add => node.dtype.zero(),
        AstOp::Mul => node.dtype.one(),
        AstOp::Max => node.dtype.min_value(),
        _ => unimplemented!("Unsupported reduction operation: {:?}", op),
    };
    let init_acc = AstNode::assign(acc_var.clone(), init_val);

    let mut inner_indices = indices.to_vec();
    let mut loops = vec![];

    let mut lowered_src = lowerer.lower_node_rec(&node.src[0], &mut inner_indices, inputs, None);
    let value_to_reduce = lowered_src.pop().unwrap();
    let update_op = AstNode::_new(
        op.clone(),
        vec![acc_var.clone(), value_to_reduce],
        node.dtype.clone(),
    );
    let mut body = AstNode::assign(acc_var.clone(), update_op);
    body = AstNode::block(
        lowered_src
            .into_iter()
            .chain(std::iter::once(body))
            .collect(),
    );

    for (i, axis) in axes.iter().enumerate().rev() {
        let reduce_dim = node.src[0].shape()[*axis].clone();
        let ridx_name = format!("ridx{}_{}", lowerer.ridx_counter, i);
        let ridx_var = AstNode::var(&ridx_name, node.dtype.clone());
        inner_indices.insert(*axis, ridx_var);

        body = AstNode::range(
            &ridx_name,
            1,
            AstNode::from(0isize),
            reduce_dim.into(),
            body,
        );
    }
    loops.push(body);
    lowerer.ridx_counter += 1;

    let mut stmts = vec![init_acc];
    stmts.extend(loops);

    let output_view = View::new_contiguous(node.shape().to_vec());
    let physical_index = output_view.to_physical_index_ast(indices);
    let store_ptr = AstNode::index(output_ptr, physical_index);
    stmts.push(AstNode::store(store_ptr, acc_var));

    stmts
}

fn replace_captures_rec(
    lowerer: &mut Lowerer,
    graph_node: &GraphNode,
    indices: &mut [AstNode],
    inputs: &[GraphNode],
    fused_ast: &AstNode,
) -> Vec<AstNode> {
    if let AstOp::Capture(n) = fused_ast.op {
        return lowerer.lower_node_rec(&graph_node.src[n], indices, inputs, None);
    }

    let mut stmts = vec![];
    let mut lowered_srcs = vec![];
    for src in &fused_ast.src {
        let mut lowered = replace_captures_rec(lowerer, graph_node, indices, inputs, src);
        let val = lowered.pop().unwrap();
        stmts.extend(lowered);
        lowered_srcs.push(val);
    }

    stmts.push(AstNode::_new(
        fused_ast.op.clone(),
        lowered_srcs,
        fused_ast.dtype.clone(),
    ));
    stmts
}

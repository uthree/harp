use crate::{
    node::Node,
    operator::{self, BinaryOp, Operator, ReduceOp, UnaryOp},
    tensor::TensorData,
};
use ndarray::{ArrayD, Axis};
use petgraph::{graph::NodeIndex, visit::EdgeRef};
use std::collections::HashMap;

pub struct Interpreter {
    cache: HashMap<NodeIndex, TensorData>,
}

impl Interpreter {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn evaluate(
        &mut self,
        node_index: NodeIndex,
        graph: &petgraph::graph::DiGraph<Node, usize>,
        inputs: &HashMap<NodeIndex, TensorData>,
    ) -> Result<TensorData, String> {
        if let Some(data) = self.cache.get(&node_index) {
            return Ok(data.clone());
        }

        let node = graph.node_weight(node_index).ok_or("Node not found")?;
        let op = node.op();

        let result = if let Some(input_data) = inputs.get(&node_index) {
            // If it's an input node, use the provided input data
            input_data.clone()
        } else if let Some(const_op) = op.as_any().downcast_ref::<operator::Const>() {
            // If it's a constant node, use its internal data
            const_op.data.clone()
        } else {
            // Evaluate based on operator type
            let parents: Vec<(NodeIndex, usize)> = graph
                .edges_directed(node_index, petgraph::Direction::Incoming)
                .map(|edge| (edge.source(), *edge.weight()))
                .collect();

            let mut parent_data = HashMap::new();
            for (parent_idx, arg_idx) in parents {
                let data = self.evaluate(parent_idx, graph, inputs)?;
                parent_data.insert(arg_idx, data);
            }

            if let Some(unary_op) = op.as_any().downcast_ref::<dyn UnaryOp>() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Unary op missing input")?
                    .0
                    .clone();
                let output = if unary_op.as_any().downcast_ref::<operator::Exp2>().is_some() {
                    input.mapv(|x| 2.0f32.powf(x))
                } else if unary_op.as_any().downcast_ref::<operator::Log2>().is_some() {
                    input.mapv(|x| x.log2())
                } else if unary_op.as_any().downcast_ref::<operator::Sin>().is_some() {
                    input.mapv(|x| x.sin())
                } else if unary_op.as_any().downcast_ref::<operator::Sqrt>().is_some() {
                    input.mapv(|x| x.sqrt())
                } else if unary_op
                    .as_any()
                    .downcast_ref::<operator::Recip>()
                    .is_some()
                {
                    input.mapv(|x| 1.0 / x)
                } else {
                    return Err(format!("Unsupported unary op: {:?}", unary_op));
                };
                TensorData(output)
            } else if let Some(binary_op) = op.as_any().downcast_ref::<dyn BinaryOp>() {
                let lhs = parent_data
                    .get(&0)
                    .ok_or("Binary op missing lhs")?
                    .0
                    .clone();
                let rhs = parent_data
                    .get(&1)
                    .ok_or("Binary op missing rhs")?
                    .0
                    .clone();
                let output = if binary_op.as_any().downcast_ref::<operator::Add>().is_some() {
                    lhs + rhs
                } else if binary_op.as_any().downcast_ref::<operator::Mul>().is_some() {
                    lhs * rhs
                } else if binary_op.as_any().downcast_ref::<operator::Rem>().is_some() {
                    lhs % rhs
                } else if binary_op
                    .as_any()
                    .downcast_ref::<operator::LessThan>()
                    .is_some()
                {
                    lhs.mapv(|a| if a < rhs[[0]] { 1.0 } else { 0.0 })
                } else {
                    return Err(format!("Unsupported binary op: {:?}", binary_op));
                };
                TensorData(output)
            } else if let Some(reduce_op) = op.as_any().downcast_ref::<dyn ReduceOp>() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Reduce op missing input")?
                    .0
                    .clone();
                let dim = reduce_op.dim();
                let output = if reduce_op
                    .as_any()
                    .downcast_ref::<operator::SumReduce>()
                    .is_some()
                {
                    input.sum_axis(Axis(dim))
                } else if reduce_op
                    .as_any()
                    .downcast_ref::<operator::MaxReduce>()
                    .is_some()
                {
                    input.fold_axis(Axis(dim), f32::MIN, |&acc, &x| acc.max(x))
                } else {
                    return Err(format!("Unsupported reduce op: {:?}", reduce_op));
                };
                TensorData(output.into_dyn())
            } else if op.as_any().downcast_ref::<operator::Contiguous>().is_some() {
                // Contiguous: For now, assume it's already contiguous or handle simple cases.
                // This is a movement op, actual data manipulation depends on ShapeTracker.
                // For interpreter, we might just return the input data as is if shapes match.
                parent_data
                    .get(&0)
                    .ok_or("Contiguous op missing input")?
                    .clone()
            } else {
                return Err(format!("Unsupported operator for interpretation: {:?}", op));
            }
        };

        self.cache.insert(node_index, result.clone());
        Ok(result)
    }
}

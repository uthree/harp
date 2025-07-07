use crate::{
    node::Node,
    operator::{self, Operator},
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

            // Handle Unary Operators
            if op.as_any().downcast_ref::<operator::Exp2>().is_some() {
                let input = parent_data.get(&0).ok_or("Exp2 op missing input")?.0.clone();
                TensorData(input.mapv(|x| 2.0f32.powf(x)))
            } else if op.as_any().downcast_ref::<operator::Log2>().is_some() {
                let input = parent_data.get(&0).ok_or("Log2 op missing input")?.0.clone();
                TensorData(input.mapv(|x| x.log2()))
            } else if op.as_any().downcast_ref::<operator::Sin>().is_some() {
                let input = parent_data.get(&0).ok_or("Sin op missing input")?.0.clone();
                TensorData(input.mapv(|x| x.sin()))
            } else if op.as_any().downcast_ref::<operator::Sqrt>().is_some() {
                let input = parent_data.get(&0).ok_or("Sqrt op missing input")?.0.clone();
                TensorData(input.mapv(|x| x.sqrt()))
            } else if op.as_any().downcast_ref::<operator::Recip>().is_some() {
                let input = parent_data.get(&0).ok_or("Recip op missing input")?.0.clone();
                TensorData(input.mapv(|x| 1.0 / x))
            }
            // Handle Binary Operators
            else if op.as_any().downcast_ref::<operator::Add>().is_some() {
                let lhs = parent_data.get(&0).ok_or("Add op missing lhs")?.0.clone();
                let rhs = parent_data.get(&1).ok_or("Add op missing rhs")?.0.clone();
                TensorData(lhs + rhs)
            } else if op.as_any().downcast_ref::<operator::Mul>().is_some() {
                let lhs = parent_data.get(&0).ok_or("Mul op missing lhs")?.0.clone();
                let rhs = parent_data.get(&1).ok_or("Mul op missing rhs")?.0.clone();
                TensorData(lhs * rhs)
            } else if op.as_any().downcast_ref::<operator::Rem>().is_some() {
                let lhs = parent_data.get(&0).ok_or("Rem op missing lhs")?.0.clone();
                let rhs = parent_data.get(&1).ok_or("Rem op missing rhs")?.0.clone();
                TensorData(lhs % rhs)
            } else if op.as_any().downcast_ref::<operator::LessThan>().is_some() {
                let lhs = parent_data.get(&0).ok_or("LessThan op missing lhs")?.0.clone();
                let rhs = parent_data.get(&1).ok_or("LessThan op missing rhs")?.0.clone();
                TensorData(lhs.mapv(|a| if a < rhs[[0]] { 1.0 } else { 0.0 }))
            }
            // Handle Reduce Operators
            else if let Some(sum_reduce_op) = op.as_any().downcast_ref::<operator::SumReduce>() {
                let input = parent_data.get(&0).ok_or("SumReduce op missing input")?.0.clone();
                let dim = sum_reduce_op.dim; // Access dim directly from concrete type
                TensorData(input.sum_axis(Axis(dim)).into_dyn())
            } else if let Some(max_reduce_op) = op.as_any().downcast_ref::<operator::MaxReduce>() {
                let input = parent_data.get(&0).ok_or("MaxReduce op missing input")?.0.clone();
                let dim = max_reduce_op.dim; // Access dim directly from concrete type
                TensorData(input.fold_axis(Axis(dim), f32::MIN, |&acc, &x| acc.max(x)).into_dyn())
            }
            // Handle Movement Operators
            else if op.as_any().downcast_ref::<operator::Contiguous>().is_some() {
                parent_data.get(&0).ok_or("Contiguous op missing input")?.clone()
            } else {
                return Err(format!("Unsupported operator for interpretation: {:?}", op));
            }
        };

        self.cache.insert(node_index, result.clone());
        Ok(result)
    }
}
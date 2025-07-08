use crate::{dtype::DType, graph::EdgeMetadata, node::Node, operator, tensor::TensorData};
use ndarray::Axis;
use petgraph::{graph::NodeIndex, visit::EdgeRef};
use std::collections::HashMap;

/// Interprets a computation graph and evaluates nodes.
///
/// The `Interpreter` traverses the graph, evaluating each node's operation
/// and caching the results to avoid redundant computations.
pub struct Interpreter {
    /// A cache to store the computed `TensorData` for each `NodeIndex`.
    cache: HashMap<NodeIndex, TensorData>,
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpreter {
    /// Creates a new `Interpreter` with an empty cache.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Evaluates the tensor data for a given node in the computation graph.
    ///
    /// This method recursively evaluates the node's dependencies (parents)
    /// and applies the corresponding operator to produce the result.
    /// Results are cached to optimize subsequent evaluations of the same node.
    ///
    /// # Arguments
    ///
    /// * `node_index` - The `NodeIndex` of the node to evaluate.
    /// * `graph` - A reference to the `petgraph::graph::DiGraph` representing the computation graph.
    /// * `inputs` - A `HashMap` containing initial `TensorData` for input nodes.
    ///
    /// # Returns
    ///
    /// A `Result` which is `Ok(TensorData)` if the evaluation is successful,
    /// or `Err(String)` if an error occurs (e.g., node not found, missing input,
    /// or unsupported operator).
    pub fn evaluate(
        &mut self,
        node_index: NodeIndex,
        graph: &petgraph::graph::DiGraph<Node, EdgeMetadata>,
        global_inputs: &HashMap<NodeIndex, TensorData>,
        local_inputs: &HashMap<NodeIndex, TensorData>,
    ) -> Result<TensorData, String> {
        // If the result is already in the cache, return it directly.
        if let Some(data) = self.cache.get(&node_index) {
            println!("Evaluating node: {:?} (Cache Hit)", node_index);
            return Ok(data.clone());
        }

        let node = graph.node_weight(node_index).ok_or("Node not found")?;
        let op = node.op();
        println!("Evaluating node: {node_index:?}");
        println!("Operator: {op:?}");

        let result = if let Some(input_data) = global_inputs.get(&node_index) {
            // If it's a global input node, use the provided input data.
            input_data.clone()
        } else if let Some(const_op) = op.as_any().downcast_ref::<operator::Const>() {
            // If it's a constant node, use its internal data.
            const_op.data.clone()
        } else {
            // Evaluate based on operator type by recursively evaluating parent nodes.
            let parents: Vec<(NodeIndex, usize)> = graph
                .edges_directed(node_index, petgraph::Direction::Incoming)
                .map(|edge| (edge.source(), edge.weight().arg_index))
                .collect();

            let mut parent_data = HashMap::<usize, TensorData>::new();
            for (parent_idx, arg_idx) in parents {
                let data = if let Some(data) = local_inputs.get(&parent_idx) {
                    data.clone()
                } else {
                    self.evaluate(parent_idx, graph, global_inputs, local_inputs)?
                };
                parent_data.insert(arg_idx, data);
            }

            // Handle Unary Operators
            if op.as_any().downcast_ref::<operator::Exp2>().is_some() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Exp2 op missing input")?
                    .data
                    .clone();
                TensorData {
                    data: input.mapv(|x| 2.0f32.powf(x)),
                    dtype: DType::F32,
                }
            } else if op.as_any().downcast_ref::<operator::Log2>().is_some() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Log2 op missing input")?
                    .data
                    .clone();
                TensorData {
                    data: input.mapv(|x| x.log2()),
                    dtype: DType::F32,
                }
            } else if op.as_any().downcast_ref::<operator::Sin>().is_some() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Sin op missing input")?
                    .data
                    .clone();
                TensorData {
                    data: input.mapv(|x| x.sin()),
                    dtype: DType::F32,
                }
            } else if op.as_any().downcast_ref::<operator::Sqrt>().is_some() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Sqrt op missing input")?
                    .data
                    .clone();
                TensorData {
                    data: input.mapv(|x| x.sqrt()),
                    dtype: DType::F32,
                }
            } else if op.as_any().downcast_ref::<operator::Recip>().is_some() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Recip op missing input")?
                    .data
                    .clone();
                TensorData {
                    data: input.mapv(|x| 1.0 / x),
                    dtype: DType::F32,
                }
            } else if let Some(cast_op) = op.as_any().downcast_ref::<operator::Cast>() {
                let input_tensor_data = parent_data.get(&0).ok_or("Cast op missing input")?.clone();
                // For now, just return the input data with the new dtype
                TensorData {
                    data: input_tensor_data.data,
                    dtype: cast_op.dtype,
                }
            }
            // Handle Binary Operators
            else if op.as_any().downcast_ref::<operator::Add>().is_some() {
                let lhs = parent_data
                    .get(&0)
                    .ok_or("Add op missing lhs")?
                    .data
                    .clone();
                let rhs = parent_data
                    .get(&1)
                    .ok_or("Add op missing rhs")?
                    .data
                    .clone();
                TensorData {
                    data: lhs + rhs,
                    dtype: DType::F32,
                }
            } else if op.as_any().downcast_ref::<operator::Mul>().is_some() {
                let lhs = parent_data
                    .get(&0)
                    .ok_or("Mul op missing lhs")?
                    .data
                    .clone();
                let rhs = parent_data
                    .get(&1)
                    .ok_or("Mul op missing rhs")?
                    .data
                    .clone();
                TensorData {
                    data: lhs * rhs,
                    dtype: DType::F32,
                }
            } else if op.as_any().downcast_ref::<operator::Rem>().is_some() {
                let lhs = parent_data
                    .get(&0)
                    .ok_or("Rem op missing lhs")?
                    .data
                    .clone();
                let rhs = parent_data
                    .get(&1)
                    .ok_or("Rem op missing rhs")?
                    .data
                    .clone();
                TensorData {
                    data: lhs % rhs,
                    dtype: DType::F32,
                }
            } else if op.as_any().downcast_ref::<operator::LessThan>().is_some() {
                let lhs = parent_data
                    .get(&0)
                    .ok_or("LessThan op missing lhs")?
                    .data
                    .clone();
                let rhs = parent_data
                    .get(&1)
                    .ok_or("LessThan op missing rhs")?
                    .data
                    .clone();
                TensorData {
                    data: lhs.mapv(|a| {
                        if a < *rhs.iter().next().unwrap() {
                            1.0
                        } else {
                            0.0
                        }
                    }),
                    dtype: DType::F32,
                }
            }
            // Handle Reduce Operators
            else if let Some(sum_reduce_op) = op.as_any().downcast_ref::<operator::SumReduce>() {
                let input = parent_data
                    .get(&0)
                    .ok_or("SumReduce op missing input")?
                    .data
                    .clone();
                let dim = sum_reduce_op.dim; // Access dim directly from concrete type
                TensorData {
                    data: input.sum_axis(Axis(dim)).into_dyn(),
                    dtype: DType::F32,
                }
            } else if let Some(max_reduce_op) = op.as_any().downcast_ref::<operator::MaxReduce>() {
                let input = parent_data
                    .get(&0)
                    .ok_or("MaxReduce op missing input")?
                    .data
                    .clone();
                let dim = max_reduce_op.dim; // Access dim directly from concrete type
                TensorData {
                    data: input
                        .fold_axis(Axis(dim), f32::MIN, |&acc, &x| acc.max(x))
                        .into_dyn(),
                    dtype: DType::F32,
                }
            }
            // Handle Movement Operators
            else if op.as_any().downcast_ref::<operator::Contiguous>().is_some() {
                let input = parent_data
                    .get(&0)
                    .ok_or("Contiguous op missing input")?
                    .clone();
                input
            } else {
                return Err(format!("Unsupported operator for interpretation: {op:?}"));
            }
        };

        self.cache.insert(node_index, result.clone());
        Ok(result)
    }
}

use crate::dot::ToDot;
use crate::dtype::DType;
use crate::node::{self, constant, Node};
use crate::op::{
    Const, Expand, HasIdentityElement, Load, OpAdd, OpDiv, OpMul, OpRandn, OpSub, OpUniform,
    Operator, Permute, Reduce, Reshape, Slice,
};
use crate::simplify::simplify;
use dyn_clone::clone_box;
use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;
use std::sync::Arc;

/// A multi-dimensional array.
///
/// `Tensor` is a lightweight, reference-counted wrapper around the core
/// computation graph (`TensorData`). Cloning a `Tensor` is cheap.
#[derive(Clone)]
pub struct Tensor {
    pub data: Arc<TensorData>,
}

impl Tensor {
    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &Vec<usize> {
        &self.data.shape
    }
}

impl ToDot for Tensor {
    fn to_dot(&self) -> String {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut visited = HashMap::new();
        let mut counter = 0;
        Self::build_dot_recursive(self, &mut nodes, &mut edges, &mut visited, &mut counter);

        let mut dot = String::from("digraph G {\n");
        dot.push_str("  rankdir=TB;\n\n");
        dot.push_str("  // Nodes\n");
        for node_def in nodes {
            dot.push_str(&format!("  {node_def}\n"));
        }
        dot.push_str("\n  // Edges\n");
        for edge_def in edges {
            dot.push_str(&format!("  {edge_def}\n"));
        }
        dot.push_str("}\n");
        dot
    }
}

impl Tensor {
    fn build_dot_recursive(
        tensor: &Tensor,
        nodes: &mut Vec<String>,
        edges: &mut Vec<String>,
        visited: &mut HashMap<*const TensorData, String>,
        counter: &mut usize,
    ) {
        let tensor_ptr = Arc::as_ptr(&tensor.data);
        if visited.contains_key(&tensor_ptr) {
            return;
        }

        let node_id = format!("node{}", *counter);
        *counter += 1;
        visited.insert(tensor_ptr, node_id.clone());

        let label = format!("{}\nshape: {:?}", tensor.data.op.name(), tensor.shape());
        let shape_style = "box";
        nodes.push(format!(
            "{node_id} [label=\"{label}\", shape=\"{shape_style}\"];"
        ));

        for src_tensor in &tensor.data.src {
            Self::build_dot_recursive(src_tensor, nodes, edges, visited, counter);
            let src_id = visited.get(&Arc::as_ptr(&src_tensor.data)).unwrap();
            edges.push(format!("{src_id} -> {node_id};"));
        }
    }
}

/// The internal representation of a `Tensor`'s computation.
///
/// It holds the operator that produces the tensor's value and a list
/// of source (input) tensors.
pub struct TensorData {
    pub op: Box<dyn Operator>,
    pub src: Vec<Tensor>,
    pub shape: Vec<usize>,
}

/// Tracks the shape and indexing of a `Tensor`.
///
/// This is crucial for compiling the high-level tensor graph into a
/// scalar `Node` graph. It resolves multi-dimensional indexing into
/// linear memory offsets.
#[derive(Clone)]
pub struct ShapeTracker {
    /// The size of each dimension (e.g., `[4, 3]` for a 4x3 matrix).
    pub dims: Vec<Rc<Node>>,
    /// The mathematical expression to convert a multi-dimensional index
    /// into a linear memory offset.
    pub index_expr: Vec<Rc<Node>>,
}

impl Tensor {
    /// Creates a new tensor filled with a specific scalar value.
    ///
    /// This operation is achieved by creating a scalar constant tensor and
    /// then expanding it to the desired shape, which is memory-efficient.
    pub fn full<T: DType + 'static>(shape: Vec<usize>, value: T) -> Self {
        // Create a scalar tensor (0-dimensional)
        let scalar = Self {
            data: Arc::new(TensorData {
                op: Box::new(Const(Box::new(value))),
                src: vec![],
                shape: vec![], // Scalar shape
            }),
        };
        // Expand the scalar to the target shape
        scalar.expand(shape)
    }

    /// Creates a new tensor with values to be sampled from a uniform distribution [0, 1).
    ///
    /// The random numbers are not generated immediately but are represented by
    /// the `OpUniform` operator in the computation graph.
    pub fn uniform(shape: Vec<usize>) -> Self {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpUniform),
                src: vec![],
                shape,
            }),
        }
    }

    /// Creates a new tensor with values to be sampled from a standard normal distribution.
    ///
    /// The random numbers are not generated immediately but are represented by
    /// the `OpRandn` operator in the computation graph.
    pub fn randn(shape: Vec<usize>) -> Self {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpRandn),
                src: vec![],
                shape,
            }),
        }
    }

    /// Creates a new tensor of zeros with the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        Self::full(shape, 0.0)
    }

    /// Creates a new tensor of ones with the given shape.
    pub fn ones(shape: Vec<usize>) -> Self {
        Self::full(shape, 1.0)
    }

    /// Creates a new tensor of zeros with the same shape as the given tensor.
    pub fn zeros_like(other: &Self) -> Self {
        Self::full(other.shape().clone(), 0.0)
    }

    /// Creates a new tensor of ones with the same shape as the given tensor.
    pub fn ones_like(other: &Self) -> Self {
        Self::full(other.shape().clone(), 1.0)
    }

    /// Creates a new "leaf" tensor that represents loading data from a source.
    pub fn new_load(shape: Vec<usize>) -> Self {
        Self {
            data: Arc::new(TensorData {
                op: Box::new(Load),
                src: vec![],
                shape,
            }),
        }
    }

    /// Changes the shape of the tensor without changing its data.
    pub fn reshape(self, new_shape: Vec<usize>) -> Self {
        let original_size: u64 = self.shape().iter().map(|&d| d as u64).product();
        let new_size: u64 = new_shape.iter().map(|&d| d as u64).product();
        assert_eq!(
            original_size, new_size,
            "Cannot reshape tensor of size {original_size} to shape {new_shape:?} with size {new_size}"
        );

        Self {
            data: Arc::new(TensorData {
                op: Box::new(Reshape),
                src: vec![self],
                shape: new_shape,
            }),
        }
    }

    /// Permutes the dimensions of the tensor.
    pub fn permute(self, order: Vec<usize>) -> Self {
        assert_eq!(
            self.shape().len(),
            order.len(),
            "The new order must have the same number of dimensions"
        );
        let new_shape = order.iter().map(|&i| self.shape()[i]).collect();
        Self {
            data: Arc::new(TensorData {
                op: Box::new(Permute { order }),
                src: vec![self],
                shape: new_shape,
            }),
        }
    }

    /// Expands the tensor to a new shape by adding new dimensions or stretching existing ones of size 1.
    pub fn expand(self, new_shape: Vec<usize>) -> Self {
        assert!(
            self.shape().len() <= new_shape.len(),
            "New shape must have at least as many dimensions as the original shape"
        );
        // Add checks to ensure broadcast compatibility
        // ...

        Self {
            data: Arc::new(TensorData {
                op: Box::new(Expand {
                    shape: new_shape.clone(),
                }),
                src: vec![self],
                shape: new_shape,
            }),
        }
    }

    /// Slices the tensor along each dimension.
    pub fn slice(self, args: Vec<(usize, usize)>) -> Self {
        assert_eq!(
            self.shape().len(),
            args.len(),
            "Number of slice arguments must match number of dimensions"
        );
        let new_shape = args.iter().map(|(start, end)| end - start).collect();
        Self {
            data: Arc::new(TensorData {
                op: Box::new(Slice { args }),
                src: vec![self],
                shape: new_shape,
            }),
        }
    }

    /// Creates a new `Reduce` tensor.
    pub fn reduce(self, op: impl Operator + 'static, axis: usize) -> Self {
        // Calculate the new shape after reduction
        let mut new_shape = self.shape().clone();
        if axis < new_shape.len() {
            new_shape.remove(axis);
        }

        Self {
            data: Arc::new(TensorData {
                op: Box::new(Reduce {
                    op: Box::new(op),
                    axis,
                }),
                src: vec![self],
                shape: new_shape,
            }),
        }
    }

    /// Performs a sum reduction along a specified axis.
    pub fn sum(self, axis: usize) -> Self {
        self.reduce(OpAdd, axis)
    }

    /// Rearranges dimensions of a tensor based on a pattern.
    ///
    /// Supports permutation and composition, e.g., "b h w c -> b (h w) c".
    pub fn rearrange(self, pattern: &str) -> Self {
        let parts: Vec<&str> = pattern.split("->").collect();
        assert_eq!(parts.len(), 2, "Invalid einops pattern: must contain '->'");
        let left_str = parts[0].trim();
        let right_str = parts[1].trim();

        let left_dims: Vec<&str> = left_str.split_whitespace().collect();
        assert_eq!(
            left_dims.len(),
            self.shape().len(),
            "Number of dimensions on left side of pattern must match tensor shape"
        );

        let mut current_tensor = self;
        let mut current_dims = left_dims.clone();

        // --- Composition Step ---
        let mut next_right_dims = vec![];
        let mut composition_groups: Vec<Vec<&str>> = vec![];
        let mut in_group = false;
        let mut current_group = vec![];

        for dim in right_str.split_whitespace() {
            if dim.starts_with('(') {
                in_group = true;
                current_group.push(dim.strip_prefix('(').unwrap());
            } else if dim.ends_with(')') {
                in_group = false;
                current_group.push(dim.strip_suffix(')').unwrap());
                let group_name = format!("({})", current_group.join(" "));
                next_right_dims.push(group_name);
                composition_groups.push(current_group.clone());
                current_group.clear();
            } else if in_group {
                current_group.push(dim);
            } else {
                next_right_dims.push(dim.to_string());
            }
        }

        if !composition_groups.is_empty() {
            // This is a simplified implementation. A full implementation would be more robust.
            let group = &composition_groups[0]; // Assuming one group for now
            let group_indices: Vec<usize> = group
                .iter()
                .map(|d| current_dims.iter().position(|&cd| cd == *d).unwrap())
                .collect();

            // Permute to make the group contiguous
            let mut permute_order: Vec<usize> = (0..current_dims.len()).collect();
            let _non_group_dims: Vec<usize> = (0..current_dims.len())
                .filter(|i| !group_indices.contains(i))
                .collect();

            let first_group_idx = *group_indices.iter().min().unwrap();
            permute_order.splice(first_group_idx..first_group_idx, group_indices.clone());

            let mut final_order = vec![];
            let mut group_added = false;
            for i in 0..current_dims.len() {
                if !group_indices.contains(&i) {
                    final_order.push(i);
                } else if !group_added {
                    final_order.extend(group_indices.clone());
                    group_added = true;
                }
            }

            current_tensor = current_tensor.permute(final_order.clone());
            current_dims = final_order.iter().map(|&i| current_dims[i]).collect();

            // Reshape to compose the group
            let mut new_shape = vec![];
            let mut composed_dim = 1;
            let mut in_composition = false;
            for (i, &dim_name) in current_dims.iter().enumerate() {
                if group.contains(&dim_name) {
                    composed_dim *= current_tensor.shape()[i];
                    if !in_composition {
                        in_composition = true;
                    }
                } else {
                    if in_composition {
                        new_shape.push(composed_dim);
                        composed_dim = 1;
                        in_composition = false;
                    }
                    new_shape.push(current_tensor.shape()[i]);
                }
            }
            if in_composition {
                new_shape.push(composed_dim);
            }

            current_tensor = current_tensor.reshape(new_shape);
            current_dims = next_right_dims.iter().map(|s| s.as_str()).collect();
        }

        // --- Permutation Step ---
        let right_map: HashMap<&str, usize> = current_dims
            .iter()
            .enumerate()
            .map(|(i, &s)| (s, i))
            .collect();

        let order: Vec<usize> = right_str
            .split_whitespace()
            .filter(|s| !s.starts_with('(') && !s.ends_with(')'))
            .map(|s| {
                *right_map
                    .get(s)
                    .unwrap_or_else(|| panic!("Dimension '{s}' not found on right side of pattern"))
            })
            .collect();

        if !order.is_empty() && (0..order.len()).all(|i| order.contains(&i)) {
            current_tensor.permute(order)
        } else {
            current_tensor
        }
    }

    /// Compiles the tensor's computation graph into a traditional Node graph.
    /// This process resolves tensor indexing into scalar operations.
    pub fn compile(&self, shape_tracker: &ShapeTracker) -> Rc<Node> {
        let op = &self.data.op;
        let src = &self.data.src;

        let compiled_node = match op.name() {
            "Load" => {
                // Calculate the flat, 1D memory offset from the multi-dimensional index.
                let mut offset = constant(0.0);
                let mut stride = constant(1.0);
                for (i, (_dim, idx)) in shape_tracker
                    .dims
                    .iter()
                    .zip(shape_tracker.index_expr.iter())
                    .enumerate()
                    .rev()
                {
                    if i < shape_tracker.dims.len() - 1 {
                        let prev_dim = &shape_tracker.dims[i + 1];
                        stride *= (**prev_dim).clone();
                    }
                    offset += (**idx).clone() * stride.clone();
                }

                Rc::new(node::Node::from(Arc::new(node::NodeData {
                    op: Box::new(Load),
                    src: vec![offset],
                })))
            }
            "Reshape" => {
                let source_tensor = &src[0];
                let source_shape_dims = source_tensor
                    .shape()
                    .iter()
                    .map(|&d| Rc::new(constant(d)))
                    .collect();
                let new_tracker = ShapeTracker {
                    dims: source_shape_dims,
                    index_expr: shape_tracker.index_expr.clone(),
                };
                return source_tensor.compile(&new_tracker); // Early return to avoid double simplification
            }
            "Permute" => {
                let permute_op = op.as_any().downcast_ref::<Permute>().unwrap();
                let source_tensor = &src[0];
                let new_index_expr = permute_op
                    .order
                    .iter()
                    .map(|&i| shape_tracker.index_expr[i].clone())
                    .collect();

                let new_tracker = ShapeTracker {
                    dims: source_tensor
                        .shape()
                        .iter()
                        .map(|&d| Rc::new(constant(d)))
                        .collect(),
                    index_expr: new_index_expr,
                };
                return source_tensor.compile(&new_tracker); // Early return to avoid double simplification
            }
            "Expand" => {
                let source_tensor = &src[0];
                let mut new_index_expr = shape_tracker.index_expr.clone();
                let diff = self.shape().len() - source_tensor.shape().len();
                for i in 0..diff {
                    new_index_expr.remove(i);
                }

                let new_tracker = ShapeTracker {
                    dims: source_tensor
                        .shape()
                        .iter()
                        .map(|&d| Rc::new(constant(d)))
                        .collect(),
                    index_expr: new_index_expr,
                };
                return source_tensor.compile(&new_tracker);
            }
            "Slice" => {
                let slice_op = op.as_any().downcast_ref::<Slice>().unwrap();
                let source_tensor = &src[0];
                let new_index_expr = shape_tracker
                    .index_expr
                    .iter()
                    .zip(slice_op.args.iter())
                    .map(|(idx, (start, _end))| Rc::new((**idx).clone() + constant(*start as f64)))
                    .collect();

                let new_tracker = ShapeTracker {
                    dims: source_tensor
                        .shape()
                        .iter()
                        .map(|&d| Rc::new(constant(d)))
                        .collect(),
                    index_expr: new_index_expr,
                };
                return source_tensor.compile(&new_tracker);
            }
            "Reduce" => {
                let reduce_op = op.as_any().downcast_ref::<Reduce>().unwrap();
                let source_tensor = &src[0];
                let axis = reduce_op.axis;
                let dim_size = source_tensor.shape()[axis];

                let identity_node = if reduce_op.op.as_any().downcast_ref::<OpAdd>().is_some() {
                    OpAdd::identity_element()
                } else if reduce_op.op.as_any().downcast_ref::<OpMul>().is_some() {
                    OpMul::identity_element()
                } else {
                    panic!("Reduce compilation not implemented for this op");
                };

                let mut accumulator = identity_node;

                for i in 0..dim_size {
                    let mut source_index_expr = shape_tracker.index_expr.clone();
                    source_index_expr.insert(axis, Rc::new(constant(i)));

                    let source_tracker = ShapeTracker {
                        dims: source_tensor
                            .shape()
                            .iter()
                            .map(|&d| Rc::new(constant(d)))
                            .collect(),
                        index_expr: source_index_expr,
                    };
                    let term = source_tensor.compile(&source_tracker);
                    accumulator = node::Node::from(Arc::new(node::NodeData {
                        op: clone_box(&*reduce_op.op),
                        src: vec![accumulator, (*term).clone()],
                    }));
                }
                Rc::new(accumulator)
            }
            "OpAdd" | "OpSub" | "OpMul" | "OpDiv" => {
                let left = src[0].compile(shape_tracker);
                let right = src[1].compile(shape_tracker);
                let new_op = clone_box(&**op);
                Rc::new(node::Node::from(Arc::new(node::NodeData {
                    op: new_op,
                    src: vec![(*left).clone(), (*right).clone()],
                })))
            }
            _ => todo!("Compile not implemented for op: {}", op.name()),
        };
        Rc::new(simplify((*compiled_node).clone()))
    }
}

// --- Operator Overloads ---

impl Add for Tensor {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Tensor shapes must match for Add"
        );
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpAdd),
                src: vec![self.clone(), rhs],
                shape: self.shape().clone(),
            }),
        }
    }
}

impl Sub for Tensor {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Tensor shapes must match for Sub"
        );
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpSub),
                src: vec![self.clone(), rhs],
                shape: self.shape().clone(),
            }),
        }
    }
}

impl Mul for Tensor {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Tensor shapes must match for Mul"
        );
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpMul),
                src: vec![self.clone(), rhs],
                shape: self.shape().clone(),
            }),
        }
    }
}

impl Div for Tensor {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Tensor shapes must match for Div"
        );
        Self {
            data: Arc::new(TensorData {
                op: Box::new(OpDiv),
                src: vec![self.clone(), rhs],
                shape: self.shape().clone(),
            }),
        }
    }
}

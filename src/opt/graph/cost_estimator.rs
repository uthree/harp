//! Cost estimation for graph nodes
//!
//! This module provides cost estimation for graph operations without
//! requiring full lowering to AST. The cost model considers:
//! - Memory access patterns (read/write bytes, stride penalties)
//! - Computational complexity
//! - Kernel launch overhead
//! - Cache efficiency based on loop ordering

use crate::graph::{GraphNode, GraphOp};

/// Estimate the computational cost of a graph node
///
/// Returns a rough estimate of the execution cost, where:
/// - Lower values indicate faster execution
/// - Cost units are arbitrary but consistent for comparison
pub fn estimate_node_cost(node: &GraphNode) -> usize {
    let memory_cost = estimate_memory_cost(node);
    let compute_cost = estimate_compute_cost(node);

    memory_cost + compute_cost
}

/// Estimate memory access cost for a node
///
/// Considers:
/// - Number of elements read/written
/// - Stride pattern (contiguous vs non-contiguous)
fn estimate_memory_cost(node: &GraphNode) -> usize {
    let num_elements = estimate_num_elements(node);
    let stride_penalty = estimate_stride_penalty(node);

    // Base cost: read + write operations
    // Multiply by data type size (approximate)
    let dtype_size = match &node.dtype {
        crate::ast::DType::F32 | crate::ast::DType::Isize | crate::ast::DType::Usize => 4,
        crate::ast::DType::Bool => 1,
        crate::ast::DType::Void => 0,
        crate::ast::DType::Ptr(_) => 8,    // Pointer size
        crate::ast::DType::Vec(_, _) => 4, // Use base type size for simplicity
    };

    (num_elements * dtype_size) + stride_penalty
}

/// Estimate computational cost for a node
fn estimate_compute_cost(node: &GraphNode) -> usize {
    let num_elements = estimate_num_elements(node);

    match &node.op {
        GraphOp::Input(_) | GraphOp::Const(_) => 0,

        // View operations have no compute cost (metadata only)
        GraphOp::View(_) => 0,

        // Memory operations
        GraphOp::Contiguous(_) => num_elements, // Just copy cost
        GraphOp::Cast(_, _) => num_elements,    // Type conversion
        GraphOp::Pad(_, _, _) => num_elements,  // Padding

        // Arithmetic operations
        GraphOp::Elementwise(op) => {
            use crate::graph::ops::ElementwiseOp;
            let op_cost = match op {
                ElementwiseOp::Add(_, _) | ElementwiseOp::Mul(_, _) => 1,
                ElementwiseOp::Recip(_) => 5, // Division is expensive
                ElementwiseOp::Sin(_) => 20,  // Trig functions are very expensive
                ElementwiseOp::Sqrt(_) => 10,
                ElementwiseOp::Log2(_) | ElementwiseOp::Exp2(_) => 15,
                ElementwiseOp::Neg(_) => 1,
                ElementwiseOp::Max(_, _) => 2,
                ElementwiseOp::Mod(_, _) => 3,
                ElementwiseOp::LessThan(_, _) | ElementwiseOp::Eq(_, _) => 1,
                ElementwiseOp::Select(_, _, _) => 2,
            };
            num_elements * op_cost
        }

        // Reduction operations (more expensive due to synchronization)
        GraphOp::Reduce(_, _, _) => num_elements * 3,
        GraphOp::FusedReduce(_, _, _) => num_elements * 3,

        // Cumulative operations
        GraphOp::Cumulative(_, _, _) => num_elements * 2,

        // Fused operations
        GraphOp::FusedElementwise(_, _) => num_elements * 2,
        GraphOp::FusedElementwiseReduce(_, _, _, _) => num_elements * 4,
        GraphOp::FusedElementwiseCumulative(_, _, _, _) => num_elements * 3,

        // Fold operation (col2im)
        GraphOp::Fold(_, _, _, _, _) => num_elements * 4,
    }
}

/// Estimate the number of elements in the output
fn estimate_num_elements(node: &GraphNode) -> usize {
    let shape = node.view.shape();

    // Calculate product of all dimensions
    // For symbolic expressions, use a heuristic default
    let mut total = 1usize;
    for dim_expr in shape {
        total = total.saturating_mul(estimate_expr_value(dim_expr));
    }

    total
}

/// Estimate the value of a shape expression
///
/// For constants, return the actual value.
/// For variables and complex expressions, return a heuristic default.
fn estimate_expr_value(expr: &crate::graph::shape::Expr) -> usize {
    use crate::graph::shape::Expr;

    match expr {
        Expr::Const(val) => (*val).max(1) as usize,
        Expr::Var(_) => 100, // Heuristic: assume moderate size
        Expr::Add(a, b) => estimate_expr_value(a).saturating_add(estimate_expr_value(b)),
        Expr::Mul(a, b) => estimate_expr_value(a).saturating_mul(estimate_expr_value(b)),
        Expr::Div(a, b) => {
            let b_val = estimate_expr_value(b).max(1);
            estimate_expr_value(a) / b_val
        }
        Expr::Rem(a, b) => {
            let b_val = estimate_expr_value(b).max(1);
            estimate_expr_value(a) % b_val
        }
        Expr::Sub(a, b) => estimate_expr_value(a).saturating_sub(estimate_expr_value(b)),
    }
}

/// Estimate stride penalty for non-contiguous access
fn estimate_stride_penalty(node: &GraphNode) -> usize {
    if node.view.is_contiguous() {
        0
    } else {
        // Non-contiguous access incurs cache miss penalty
        let num_elements = estimate_num_elements(node);
        num_elements / 10 // 10% penalty for non-contiguous access
    }
}

/// Estimate kernel launch overhead for unfused operations
pub fn estimate_kernel_launch_cost(num_kernels: usize) -> usize {
    // Each kernel launch has significant overhead on GPU
    // Use a fixed cost per launch
    num_kernels * 1000
}

/// Estimate total cost of a graph
pub fn estimate_graph_cost(nodes: &[GraphNode]) -> usize {
    let mut total_cost = 0;

    for node in nodes {
        total_cost += estimate_node_cost(node);
    }

    // Add kernel launch overhead (assuming one kernel per node for now)
    total_cost += estimate_kernel_launch_cost(nodes.len());

    total_cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::Graph;

    #[test]
    fn test_estimate_simple_add() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![100.into()]);
        let b = graph.input(DType::F32, vec![100.into()]);
        let c = a + b;

        let cost = estimate_node_cost(&c);
        assert!(cost > 0, "Cost should be positive");
    }

    #[test]
    fn test_expensive_operations_cost_more() {
        let mut graph = Graph::new();
        let a = graph.input(DType::F32, vec![100.into()]);

        let add_result = a.clone() + a.clone();
        let sin_result = a.sin();

        let add_cost = estimate_compute_cost(&add_result);
        let sin_cost = estimate_compute_cost(&sin_result);

        assert!(sin_cost > add_cost, "Sin should cost more than add");
    }

    #[test]
    fn test_larger_tensors_cost_more() {
        let mut graph = Graph::new();
        let small = graph.input(DType::F32, vec![10.into()]);
        let large = graph.input(DType::F32, vec![1000.into()]);

        // Create constant nodes for addition
        use crate::ast::ConstLiteral;
        use crate::graph::shape::Expr;
        use crate::graph::GraphNode;
        let one_small = GraphNode::new(
            crate::graph::GraphOp::Const(ConstLiteral::F32(1.0)),
            DType::F32,
            crate::graph::shape::view::View::new_contiguous(Vec::<Expr>::new()),
        );
        let one_large = GraphNode::new(
            crate::graph::GraphOp::Const(ConstLiteral::F32(1.0)),
            DType::F32,
            crate::graph::shape::view::View::new_contiguous(Vec::<Expr>::new()),
        );

        let small_result = small + one_small;
        let large_result = large + one_large;

        let small_cost = estimate_node_cost(&small_result);
        let large_cost = estimate_node_cost(&large_result);

        assert!(large_cost > small_cost, "Larger tensor should cost more");
    }
}

//! Graph cost estimation
//!
//! This module provides cost estimation for computation graphs,
//! used by the beam search optimizer to evaluate transformations.

use super::GraphCostEstimator;
use crate::graph::{GraphNode, GraphOp};
use crate::opt::cost_utils::log_sum_exp;
use std::collections::HashSet;

/// Simple cost estimator for computation graphs
///
/// Estimates cost based on:
/// - FLOPs (floating point operations)
/// - Memory access patterns
/// - Special operations (MatMul with WMMA)
#[derive(Clone, Debug)]
pub struct SimpleGraphCostEstimator {
    /// Cost per floating point operation
    pub flop_cost: f32,
    /// Cost per memory read
    pub memory_read_cost: f32,
    /// Cost per memory write
    pub memory_write_cost: f32,
    /// Efficiency factor for MatMul operations (lower = more efficient)
    pub matmul_efficiency: f32,
    /// Efficiency factor for WMMA-eligible MatMul (F16, aligned)
    pub wmma_efficiency: f32,
}

impl Default for SimpleGraphCostEstimator {
    fn default() -> Self {
        Self {
            flop_cost: 1.0,
            memory_read_cost: 4.0,
            memory_write_cost: 4.0,
            matmul_efficiency: 0.5, // MatMul is relatively efficient
            wmma_efficiency: 0.05,  // WMMA is very efficient
        }
    }
}

impl SimpleGraphCostEstimator {
    /// Create a new estimator with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set FLOP cost
    pub fn with_flop_cost(mut self, cost: f32) -> Self {
        self.flop_cost = cost;
        self
    }

    /// Set memory read cost
    pub fn with_memory_read_cost(mut self, cost: f32) -> Self {
        self.memory_read_cost = cost;
        self
    }

    /// Set memory write cost
    pub fn with_memory_write_cost(mut self, cost: f32) -> Self {
        self.memory_write_cost = cost;
        self
    }

    /// Set MatMul efficiency factor
    pub fn with_matmul_efficiency(mut self, efficiency: f32) -> Self {
        self.matmul_efficiency = efficiency;
        self
    }

    /// Set WMMA efficiency factor
    pub fn with_wmma_efficiency(mut self, efficiency: f32) -> Self {
        self.wmma_efficiency = efficiency;
        self
    }

    /// Estimate the number of elements in a shape
    fn estimate_elements(shape: &[crate::graph::Expr]) -> f32 {
        let mut total = 1.0_f32;
        for dim in shape {
            if let Some(c) = dim.as_const() {
                total *= c as f32;
            } else {
                // For symbolic dimensions, use a reasonable default
                total *= 256.0;
            }
        }
        total
    }

    /// Estimate cost for a single node
    fn estimate_node(&self, node: &GraphNode) -> f32 {
        let output_elements = Self::estimate_elements(&node.shape());

        match node.op() {
            GraphOp::View(_) => {
                // Views are essentially free (no computation)
                // Only charge for potential memory access if materialized
                (self.memory_read_cost * output_elements * 0.1).ln()
            }

            GraphOp::MapReduce { reduce, .. } => {
                let src_elements = if !node.sources().is_empty() {
                    Self::estimate_elements(&node.sources()[0].shape())
                } else {
                    output_elements
                };

                let flops = if reduce.is_some() {
                    // Reduce operations: process all source elements
                    src_elements * self.flop_cost
                } else {
                    // Elementwise: process output elements
                    output_elements * self.flop_cost
                };

                let memory =
                    src_elements * self.memory_read_cost + output_elements * self.memory_write_cost;

                (flops + memory).ln()
            }

            GraphOp::MatMul { .. } => {
                // MatMul: C[M, N] = A[M, K] @ B[K, N]
                // FLOPs = 2 * M * K * N (multiply-add)
                let shape = node.shape();
                let m = shape.first().and_then(|e| e.as_const()).unwrap_or(256) as f32;
                let n = shape.last().and_then(|e| e.as_const()).unwrap_or(256) as f32;

                // Get K from source shape
                let k = if !node.sources().is_empty() {
                    let src_shape = node.sources()[0].shape();
                    src_shape.last().and_then(|e| e.as_const()).unwrap_or(256) as f32
                } else {
                    256.0
                };

                let flops = 2.0 * m * k * n * self.flop_cost;
                let memory =
                    (m * k + k * n) * self.memory_read_cost + m * n * self.memory_write_cost;

                // Apply efficiency factor
                let efficiency = if self.is_wmma_eligible(node) {
                    self.wmma_efficiency
                } else {
                    self.matmul_efficiency
                };

                ((flops + memory) * efficiency).ln()
            }

            GraphOp::Unfold { .. } => {
                // Unfold is a gather operation
                let memory = output_elements * (self.memory_read_cost + self.memory_write_cost);
                memory.ln()
            }

            GraphOp::Scatter { .. } => {
                // Scatter-add: read input, atomic-add to output
                let src_elements = if !node.sources().is_empty() {
                    Self::estimate_elements(&node.sources()[0].shape())
                } else {
                    output_elements
                };
                let memory = src_elements * self.memory_read_cost
                    + output_elements * self.memory_write_cost * 2.0; // atomic add is slower
                memory.ln()
            }

            GraphOp::Scan { .. } => {
                // Sequential scan along an axis
                let src_elements = if !node.sources().is_empty() {
                    Self::estimate_elements(&node.sources()[0].shape())
                } else {
                    output_elements
                };
                let flops = src_elements * self.flop_cost;
                let memory =
                    src_elements * self.memory_read_cost + output_elements * self.memory_write_cost;
                (flops + memory).ln()
            }
        }
    }

    /// Check if a MatMul node is eligible for WMMA optimization
    fn is_wmma_eligible(&self, node: &GraphNode) -> bool {
        use crate::ast::DType;

        match node.op() {
            GraphOp::MatMul { .. } => {
                // Check dtype
                if !matches!(node.dtype(), DType::F16) {
                    return false;
                }

                // Check dimensions are multiples of 16
                let shape = node.shape();
                let m = shape.first().and_then(|e| e.as_const()).unwrap_or(0);
                let n = shape.last().and_then(|e| e.as_const()).unwrap_or(0);

                let k = if !node.sources().is_empty() {
                    node.sources()[0]
                        .shape()
                        .last()
                        .and_then(|e| e.as_const())
                        .unwrap_or(0)
                } else {
                    0
                };

                m > 0 && m % 16 == 0 && n > 0 && n % 16 == 0 && k > 0 && k % 16 == 0
            }
            _ => false,
        }
    }
}

impl GraphCostEstimator for SimpleGraphCostEstimator {
    fn estimate(&self, roots: &[GraphNode]) -> f32 {
        use std::rc::Rc;

        // Traverse all nodes and sum their costs
        let mut visited: HashSet<*const crate::graph::GraphInner> = HashSet::new();
        let mut costs: Vec<f32> = Vec::new();

        fn traverse(
            node: &GraphNode,
            estimator: &SimpleGraphCostEstimator,
            visited: &mut HashSet<*const crate::graph::GraphInner>,
            costs: &mut Vec<f32>,
        ) {
            let ptr = Rc::as_ptr(&node.0);
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);

            // Process sources first
            for src in node.sources() {
                traverse(src, estimator, visited, costs);
            }

            // Add this node's cost (skip external inputs)
            if !node.is_external() {
                costs.push(estimator.estimate_node(node));
            }
        }

        for root in roots {
            traverse(root, self, &mut visited, &mut costs);
        }

        // Sum costs in log space
        if costs.is_empty() {
            0.0
        } else {
            costs.into_iter().fold(f32::NEG_INFINITY, log_sum_exp)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::DType;
    use crate::graph::{Expr, input};

    #[test]
    fn test_estimate_elementwise() {
        let estimator = SimpleGraphCostEstimator::new();

        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = &x + &x;

        let cost = estimator.estimate(&[y]);
        assert!(cost.is_finite());
        assert!(cost > 0.0);
    }

    #[test]
    fn test_estimate_reduction() {
        let estimator = SimpleGraphCostEstimator::new();

        let x = input(vec![Expr::Const(32), Expr::Const(64)], DType::F32);
        let y = x.sum(1);

        let cost = estimator.estimate(&[y]);
        assert!(cost.is_finite());
        assert!(cost > 0.0);
    }

    #[test]
    fn test_estimate_elements() {
        // Constant dimensions
        let shape = vec![Expr::Const(32), Expr::Const(64)];
        let elements = SimpleGraphCostEstimator::estimate_elements(&shape);
        assert_eq!(elements, 32.0 * 64.0);

        // Symbolic dimension uses default
        let shape = vec![Expr::Sym("batch".to_string()), Expr::Const(64)];
        let elements = SimpleGraphCostEstimator::estimate_elements(&shape);
        assert_eq!(elements, 256.0 * 64.0);
    }
}

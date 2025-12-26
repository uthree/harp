//! Linear algebra high-level operations
//!
//! - MatMul = Unsqueeze + Mul + ReduceSum
//! - BatchMatMul = MatMul with batch dimensions
//! - Dot = Sum(a * b)

use std::marker::PhantomData;
use std::sync::Arc;

use crate::tensor::{Dim1, Dim2, DimDyn, Tensor, TensorInner};

// ============================================================================
// Type-safe operations for Dim1
// ============================================================================

impl Tensor<Dim1> {
    /// Type-safe dot product for 1D tensors
    ///
    /// a[K] · b[K] -> scalar
    ///
    /// Returns a scalar tensor (0-dimensional).
    pub fn dot1(&self, other: &Tensor<Dim1>) -> Tensor<DimDyn> {
        assert_eq!(
            self.shape()[0],
            other.shape()[0],
            "dot product dimension mismatch: {} vs {}",
            self.shape()[0],
            other.shape()[0]
        );

        let product = &self.clone().into_dyn() * &other.clone().into_dyn();
        product.reduce_sum(&[0], false)
    }

    /// Type-safe outer product for 1D tensors
    ///
    /// a[M] ⊗ b[N] -> C[M, N]
    ///
    /// Returns a 2D tensor.
    pub fn outer1(&self, other: &Tensor<Dim1>) -> Tensor<Dim2> {
        let m = self.shape()[0];
        let n = other.shape()[0];

        // a[M] -> a[M, 1]
        let a_col = self.unsqueeze(1);
        // b[N] -> b[1, N]
        let b_row = other.unsqueeze(0);

        // Broadcast multiply: [M, 1] * [1, N] -> [M, N]
        let result_dyn = a_col.expand(&[m, n]) * b_row.expand(&[m, n]);

        // Convert to Dim2
        Tensor {
            inner: Arc::new(TensorInner {
                op: result_dyn.inner.op.clone(),
                view: result_dyn.inner.view.clone(),
                shape: result_dyn.inner.shape.clone(),
                dtype: result_dyn.inner.dtype.clone(),
                name: result_dyn.inner.name.clone(),
                autograd: None,
                buffer: std::sync::RwLock::new(None),
            }),
            _dim: PhantomData,
        }
    }
}

// ============================================================================
// Type-safe matmul for Dim2
// ============================================================================

impl Tensor<Dim2> {
    /// Type-safe matrix multiplication for 2D tensors
    ///
    /// A[M, K] @ B[K, N] -> C[M, N]
    ///
    /// This is a type-safe version that preserves the Dim2 type.
    pub fn matmul2(&self, other: &Tensor<Dim2>) -> Tensor<Dim2> {
        let a_shape = self.shape();
        let b_shape = other.shape();

        let m = a_shape[0];
        let k_a = a_shape[1];
        let k_b = b_shape[0];
        let n = b_shape[1];

        assert_eq!(
            k_a, k_b,
            "matmul dimension mismatch: A[{}, {}] @ B[{}, {}] - inner dimensions must match",
            m, k_a, k_b, n
        );

        let k = k_a;

        // A[M, K] -> A[M, K, 1]
        let a_expanded = self.unsqueeze(2);
        // B[K, N] -> B[1, K, N]
        let b_expanded = other.unsqueeze(0);

        // Broadcast multiply: [M, K, 1] * [1, K, N] -> [M, K, N]
        let product = a_expanded.expand(&[m, k, n]) * b_expanded.expand(&[m, k, n]);

        // Sum over K dimension: [M, K, N] -> [M, N]
        let result_dyn = product.reduce_sum(&[1], false);

        // Convert back to Dim2
        Tensor {
            inner: Arc::new(TensorInner {
                op: result_dyn.inner.op.clone(),
                view: result_dyn.inner.view.clone(),
                shape: result_dyn.inner.shape.clone(),
                dtype: result_dyn.inner.dtype.clone(),
                name: result_dyn.inner.name.clone(),
                autograd: None,
                buffer: std::sync::RwLock::new(None),
            }),
            _dim: PhantomData,
        }
    }
}

impl Tensor<DimDyn> {
    /// Matrix multiplication for dynamic tensors (hlop)
    ///
    /// For 2D tensors A[M, K] @ B[K, N] -> C[M, N]
    /// Supports batched matmul and vector-matrix operations.
    ///
    /// For type-safe 2D matmul, use `Tensor<Dim2>::matmul2` instead.
    ///
    /// Implemented as:
    /// 1. Unsqueeze(A, -1)     → [M, K, 1]
    /// 2. Unsqueeze(B, 0)      → [1, K, N]
    /// 3. Mul                  → [M, K, N]  (broadcast)
    /// 4. ReduceSum(axis=1)    → [M, N]
    pub fn matmul(&self, other: &Tensor<DimDyn>) -> Tensor<DimDyn> {
        assert!(
            self.ndim() >= 1 && other.ndim() >= 1,
            "matmul requires at least 1D tensors"
        );

        // Get the contraction dimension
        let a_shape = self.shape();
        let b_shape = other.shape();

        if self.ndim() == 1 && other.ndim() == 1 {
            // Dot product: [K] @ [K] -> scalar
            return self.dot(other);
        }

        if self.ndim() == 1 {
            // [K] @ [K, N] -> [N]
            let k = a_shape[0];
            assert_eq!(
                k,
                b_shape[b_shape.len() - 2],
                "matmul dimension mismatch: {} vs {}",
                k,
                b_shape[b_shape.len() - 2]
            );

            // Reshape to [1, K], matmul, then squeeze
            let a_2d = self.unsqueeze(0);
            let result = a_2d.matmul(other);
            return result.squeeze_dim(0);
        }

        if other.ndim() == 1 {
            // [M, K] @ [K] -> [M]
            let k = a_shape[a_shape.len() - 1];
            assert_eq!(
                k, b_shape[0],
                "matmul dimension mismatch: {} vs {}",
                k, b_shape[0]
            );

            // Reshape to [K, 1], matmul, then squeeze
            let b_2d = other.unsqueeze(1);
            let result = self.matmul(&b_2d);
            return result.squeeze_dim(result.ndim() - 1);
        }

        // General case: [..., M, K] @ [..., K, N] -> [..., M, N]
        let k_a = a_shape[a_shape.len() - 1];
        let k_b = b_shape[b_shape.len() - 2];
        assert_eq!(k_a, k_b, "matmul dimension mismatch: {} vs {}", k_a, k_b);

        let m = a_shape[a_shape.len() - 2];
        let n = b_shape[b_shape.len() - 1];
        let k = k_a;

        // Handle batch dimensions
        let a_batch: Vec<usize> = a_shape[..a_shape.len() - 2].to_vec();
        let b_batch: Vec<usize> = b_shape[..b_shape.len() - 2].to_vec();

        // Compute broadcast batch shape
        let batch_shape = broadcast_batch_shapes(&a_batch, &b_batch);

        // For 2D case: simple implementation
        if self.ndim() == 2 && other.ndim() == 2 {
            // A[M, K] -> A[M, K, 1]
            let a_expanded = self.unsqueeze(2);
            // B[K, N] -> B[1, K, N]
            let b_expanded = other.unsqueeze(0);

            // Broadcast multiply: [M, K, 1] * [1, K, N] -> [M, K, N]
            let product = a_expanded.expand(&[m, k, n]) * b_expanded.expand(&[m, k, n]);

            // Sum over K dimension: [M, K, N] -> [M, N]
            return product.reduce_sum(&[1], false);
        }

        // For batched case, we need to handle batch dimensions
        // This is a simplified implementation
        let mut result_shape = batch_shape.clone();
        result_shape.push(m);
        result_shape.push(n);

        // General batched matmul implementation
        // Reshape to 3D, compute, reshape back
        let a_flat = if a_batch.is_empty() {
            self.unsqueeze(0)
        } else {
            let batch_size: usize = a_batch.iter().product();
            self.reshape_dyn(&[batch_size, m, k])
        };

        let b_flat = if b_batch.is_empty() {
            other.unsqueeze(0)
        } else {
            let batch_size: usize = b_batch.iter().product();
            other.reshape_dyn(&[batch_size, k, n])
        };

        // Batched matmul via einsum-like expansion
        // a_flat: [B, M, K] -> [B, M, K, 1]
        // b_flat: [B, K, N] -> [B, 1, K, N]
        let batch_a = a_flat.shape()[0];
        let batch_b = b_flat.shape()[0];
        let batch_size = batch_a.max(batch_b);

        let a_4d = a_flat.unsqueeze(3); // [B, M, K, 1]
        let b_4d = b_flat.unsqueeze(1); // [B, 1, K, N]

        // Broadcast and multiply
        let expanded_shape = vec![batch_size, m, k, n];
        let product = a_4d.expand(&expanded_shape) * b_4d.expand(&expanded_shape);

        // Sum over K (axis 2)
        let result_3d = product.reduce_sum(&[2], false); // [B, M, N]

        // Reshape to final batch shape
        result_3d.reshape_dyn(&result_shape)
    }

    /// Dot product (hlop)
    ///
    /// For 1D tensors: sum(a * b)
    pub fn dot(&self, other: &Tensor<DimDyn>) -> Tensor<DimDyn> {
        assert_eq!(self.ndim(), 1, "dot product requires 1D tensors");
        assert_eq!(other.ndim(), 1, "dot product requires 1D tensors");
        assert_eq!(
            self.shape()[0],
            other.shape()[0],
            "dot product requires same length"
        );

        let product = self * other;
        product.reduce_sum(&[0], false)
    }

    /// Outer product (hlop)
    ///
    /// For 1D tensors a[M], b[N] -> result[M, N]
    pub fn outer(&self, other: &Tensor<DimDyn>) -> Tensor<DimDyn> {
        assert_eq!(self.ndim(), 1, "outer product requires 1D tensors");
        assert_eq!(other.ndim(), 1, "outer product requires 1D tensors");

        let m = self.shape()[0];
        let n = other.shape()[0];

        // a[M] -> a[M, 1]
        let a_col = self.unsqueeze(1);
        // b[N] -> b[1, N]
        let b_row = other.unsqueeze(0);

        // Broadcast multiply
        a_col.expand(&[m, n]) * b_row.expand(&[m, n])
    }
}

/// Helper to broadcast batch dimensions
fn broadcast_batch_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let a_dim = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let b_dim = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            panic!("Cannot broadcast batch shapes {:?} and {:?}", a, b);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Dim1, Dim2};

    #[test]
    fn test_matmul2_type_safe() {
        let a: Tensor<Dim2> = Tensor::ones([2, 3]);
        let b: Tensor<Dim2> = Tensor::ones([3, 4]);
        // matmul2 returns Tensor<Dim2>, not Tensor<DimDyn>
        let c: Tensor<Dim2> = a.matmul2(&b);
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_dot1_type_safe() {
        let a: Tensor<Dim1> = Tensor::ones([3]);
        let b: Tensor<Dim1> = Tensor::ones([3]);
        // dot1 returns scalar (0-dim tensor)
        let c = a.dot1(&b);
        assert_eq!(c.shape(), &[]);
    }

    #[test]
    fn test_outer1_type_safe() {
        let a: Tensor<Dim1> = Tensor::ones([3]);
        let b: Tensor<Dim1> = Tensor::ones([4]);
        // outer1 returns Tensor<Dim2>
        let c: Tensor<Dim2> = a.outer1(&b);
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_matmul_dyn() {
        let a = Tensor::<Dim2>::ones([2, 3]).into_dyn();
        let b = Tensor::<Dim2>::ones([3, 4]).into_dyn();
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 4]);
    }

    #[test]
    fn test_dot_product() {
        let a = Tensor::<Dim1>::ones([3]).into_dyn();
        let b = Tensor::<Dim1>::ones([3]).into_dyn();
        let c = a.dot(&b);
        assert_eq!(c.shape(), &[]);
    }

    #[test]
    fn test_outer_product() {
        let a = Tensor::<Dim1>::ones([3]).into_dyn();
        let b = Tensor::<Dim1>::ones([4]).into_dyn();
        let c = a.outer(&b);
        assert_eq!(c.shape(), &[3, 4]);
    }

    #[test]
    fn test_matvec() {
        let a = Tensor::<Dim2>::ones([2, 3]).into_dyn();
        let b = Tensor::<Dim1>::ones([3]).into_dyn();
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2]);
    }

    #[test]
    fn test_vecmat() {
        let a = Tensor::<Dim1>::ones([2]).into_dyn();
        let b = Tensor::<Dim2>::ones([2, 3]).into_dyn();
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[3]);
    }
}

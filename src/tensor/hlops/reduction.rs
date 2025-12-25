//! Reduction high-level operations
//!
//! - Mean(x) = Sum(x) / count
//! - Var(x) = Mean((x - mean)^2)
//! - Std(x) = Sqrt(Var(x))
//! - Softmax(x) = Exp(x - max) / Sum(Exp(x - max))
//! - LogSoftmax(x) = x - max - Log(Sum(Exp(x - max)))

use crate::tensor::{DimDyn, Dimension, Recip, Sqrt, Tensor};

impl<D: Dimension> Tensor<D> {
    /// Alias for reduce_sum (for convenience)
    pub fn sum(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
        self.reduce_sum(axes, keepdim)
    }

    /// Mean reduction along specified axes (hlop)
    ///
    /// Implemented as: Sum(x, axes) / count
    pub fn mean(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
        let mut count: usize = 1;
        for &axis in axes {
            count *= self.shape()[axis];
        }
        let sum = self.reduce_sum(axes, keepdim);
        sum / (count as f32)
    }

    /// Variance along specified axes (hlop)
    ///
    /// Var(x) = Mean((x - mean)^2)
    pub fn var(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
        // Compute mean
        let mean = self.mean(axes, true);
        // Expand mean back to original shape for broadcasting
        let diff = &self.clone().into_dyn() - &mean.expand(self.shape());
        let sq_diff = &diff * &diff;
        sq_diff.mean(axes, keepdim)
    }

    /// Standard deviation along specified axes (hlop)
    ///
    /// Std(x) = Sqrt(Var(x))
    pub fn std(&self, axes: &[usize], keepdim: bool) -> Tensor<DimDyn> {
        self.var(axes, keepdim).sqrt()
    }

    /// Softmax along specified axis (hlop)
    ///
    /// Softmax(x) = Exp(x - max) / Sum(Exp(x - max))
    pub fn softmax(&self, axis: usize) -> Tensor<DimDyn> {
        // Numerical stability: subtract max before exp
        let max_val = self.reduce_max(&[axis], true);
        let max_expanded = max_val.expand(self.shape());
        let shifted = &self.clone().into_dyn() - &max_expanded;
        let exp_shifted = shifted.exp();
        let sum_exp = exp_shifted.reduce_sum(&[axis], true);
        let sum_expanded = sum_exp.expand(self.shape());
        exp_shifted / sum_expanded
    }

    /// Log-softmax along specified axis (hlop)
    ///
    /// LogSoftmax(x) = x - max - Log(Sum(Exp(x - max)))
    pub fn log_softmax(&self, axis: usize) -> Tensor<DimDyn> {
        // Numerical stability version
        let max_val = self.reduce_max(&[axis], true);
        let max_expanded = max_val.expand(self.shape());
        let shifted = &self.clone().into_dyn() - &max_expanded;
        let exp_shifted = shifted.exp();
        let sum_exp = exp_shifted.reduce_sum(&[axis], true);
        let log_sum_exp = sum_exp.ln();
        let log_sum_expanded = log_sum_exp.expand(self.shape());
        shifted - log_sum_expanded
    }

    /// Layer normalization (hlop)
    ///
    /// LayerNorm(x) = (x - mean) / sqrt(var + eps)
    pub fn layer_norm(&self, normalized_axes: &[usize], eps: f32) -> Tensor<DimDyn> {
        let mean = self.mean(normalized_axes, true);
        let var = self.var(normalized_axes, true);
        let mean_expanded = mean.expand(self.shape());
        let var_expanded = var.expand(self.shape());

        let centered = &self.clone().into_dyn() - &mean_expanded;
        let std_inv = (var_expanded + eps).sqrt().recip();
        centered * std_inv
    }

    /// Sum all elements (hlop)
    pub fn sum_all(&self) -> Tensor<DimDyn> {
        let axes: Vec<usize> = (0..self.ndim()).collect();
        self.reduce_sum(&axes, false)
    }

    /// Mean of all elements (hlop)
    pub fn mean_all(&self) -> Tensor<DimDyn> {
        let axes: Vec<usize> = (0..self.ndim()).collect();
        self.mean(&axes, false)
    }

    /// Max of all elements (hlop)
    pub fn max_all(&self) -> Tensor<DimDyn> {
        let axes: Vec<usize> = (0..self.ndim()).collect();
        self.reduce_max(&axes, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dim2;

    #[test]
    fn test_mean() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let m = a.mean(&[1], false);
        assert_eq!(m.shape(), &[2]);
    }

    #[test]
    fn test_mean_keepdim() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let m = a.mean(&[1], true);
        assert_eq!(m.shape(), &[2, 1]);
    }

    #[test]
    fn test_var() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let v = a.var(&[1], false);
        assert_eq!(v.shape(), &[2]);
    }

    #[test]
    fn test_std() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.std(&[1], false);
        assert_eq!(s.shape(), &[2]);
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.softmax(1);
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    fn test_log_softmax() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.log_softmax(1);
        assert_eq!(s.shape(), &[2, 3]);
    }

    #[test]
    fn test_layer_norm() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let n = a.layer_norm(&[1], 1e-5);
        assert_eq!(n.shape(), &[2, 3]);
    }

    #[test]
    fn test_sum_all() {
        let a = Tensor::<Dim2>::ones([2, 3]);
        let s = a.sum_all();
        assert_eq!(s.shape(), &[]);
    }
}

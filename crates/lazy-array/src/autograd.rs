//! Autograd Primops トレイト実装
//!
//! LazyArray に autograd クレートの primops トレイトを実装し、
//! Variable<LazyArray> で自動微分を可能にします。

use crate::backend::{ArrayElement, ArrayError, LazyArray};
use crate::dim::Dimension;

use autograd::Variable;
use autograd::primops::{Exp2, Log2, Sin, Sqrt};
use autograd::primops::{Expand, Ones, Permute, Reshape, Shape, Squeeze, Sum, Unsqueeze, Zeros};
use autograd::traits::GradRoot;

// ============================================================================
// GradRoot 実装
// ============================================================================

impl<D: Dimension> GradRoot for LazyArray<f32, D> {
    fn unit_grad() -> Self {
        // スカラー1.0を返す（f32用のfullを明示的に呼び出す）
        <LazyArray<f32, D>>::full([1], 1.0f32)
    }
}

// ============================================================================
// Shape 実装
// ============================================================================

impl<T: ArrayElement, D: Dimension> Shape for LazyArray<T, D> {
    fn shape(&self) -> &[usize] {
        LazyArray::shape(self)
    }
}

// ============================================================================
// Zeros / Ones 実装
// ============================================================================

impl<D: Dimension> Zeros for LazyArray<f32, D> {
    fn zeros(shape: &[usize]) -> Self {
        <LazyArray<f32, D>>::zeros(shape.to_vec())
    }
}

impl<D: Dimension> Ones for LazyArray<f32, D> {
    fn ones(shape: &[usize]) -> Self {
        <LazyArray<f32, D>>::ones(shape.to_vec())
    }
}

// ============================================================================
// 要素単位演算 Primops
// ============================================================================

impl<D: Dimension> Sin for LazyArray<f32, D> {
    fn sin(&self) -> Self {
        LazyArray::sin(self)
    }
}

impl<D: Dimension> Exp2 for LazyArray<f32, D> {
    fn exp2(&self) -> Self {
        LazyArray::exp2(self)
    }
}

impl<D: Dimension> Log2 for LazyArray<f32, D> {
    fn log2(&self) -> Self {
        LazyArray::log2(self)
    }
}

impl<D: Dimension> Sqrt for LazyArray<f32, D> {
    fn sqrt(&self) -> Self {
        LazyArray::sqrt(self)
    }
}

// ============================================================================
// 構造変更演算 Primops
// ============================================================================

impl<D: Dimension> Sum for LazyArray<f32, D> {
    type Output = Self;

    fn sum(&self, axis: usize) -> Self::Output {
        LazyArray::sum(self, axis)
    }
}

impl<D: Dimension> Expand for LazyArray<f32, D> {
    type Output = Self;

    fn expand(&self, axis: usize, size: usize) -> Self::Output {
        LazyArray::expand(self, axis, size)
    }
}

impl<D: Dimension> Permute for LazyArray<f32, D> {
    fn permute(&self, axes: &[usize]) -> Self {
        LazyArray::permute(self, axes)
    }
}

impl<D: Dimension> Reshape for LazyArray<f32, D> {
    fn reshape(&self, new_shape: &[usize]) -> Self {
        LazyArray::reshape(self, new_shape.to_vec())
    }
}

impl<D: Dimension> Squeeze for LazyArray<f32, D> {
    type Output = Self;

    fn squeeze(&self, axis: usize) -> Self::Output {
        LazyArray::squeeze(self, axis)
    }
}

impl<D: Dimension> Unsqueeze for LazyArray<f32, D> {
    type Output = Self;

    fn unsqueeze(&self, axis: usize) -> Self::Output {
        LazyArray::unsqueeze(self, axis)
    }
}

// ============================================================================
// Variable<LazyArray> 拡張トレイト
// ============================================================================

/// Variable<LazyArray> に forward() と to_vec() を追加する拡張トレイト
///
/// # Example
/// ```ignore
/// use harp_lazy_array::autograd::VariableLazyArrayExt;
///
/// let a = Variable::new(LazyArray::<f32, Dim1>::ones([4]));
/// let b = Variable::new(LazyArray::<f32, Dim1>::full([4], 2.0));
/// let c = &a * &b;
/// c.forward()?;  // 計算を実行
/// ```
pub trait VariableLazyArrayExt<T: ArrayElement> {
    /// 内部のLazyArrayの遅延評価を実行
    fn forward(&self) -> Result<(), ArrayError>;

    /// 内部のLazyArrayをベクタとして取得（遅延評価も実行）
    fn to_vec(&self) -> Result<Vec<T>, ArrayError>;
}

impl<T: ArrayElement, D: Dimension> VariableLazyArrayExt<T> for Variable<LazyArray<T, D>> {
    fn forward(&self) -> Result<(), ArrayError> {
        self.with_value(|arr| arr.forward())
    }

    fn to_vec(&self) -> Result<Vec<T>, ArrayError> {
        self.with_value(|arr| arr.to_vec())
    }
}

// ============================================================================
// テスト
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dim::Dim1;

    #[test]
    fn test_grad_root() {
        let unit = <LazyArray<f32, Dim1> as GradRoot>::unit_grad();
        assert_eq!(unit.shape(), &[1]);
    }

    #[test]
    fn test_zeros_ones_traits() {
        let zeros = <LazyArray<f32, Dim1> as Zeros>::zeros(&[4]);
        assert_eq!(zeros.shape(), &[4]);

        let ones = <LazyArray<f32, Dim1> as Ones>::ones(&[4]);
        assert_eq!(ones.shape(), &[4]);
    }

    #[test]
    fn test_shape_trait() {
        let arr: LazyArray<f32, Dim1> = <LazyArray<f32, Dim1>>::ones([4]);
        assert_eq!(Shape::shape(&arr), &[4]);
    }
}

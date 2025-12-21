//! ndarray に対する Reduce/Expand/Permute/Reshape/Squeeze/Unsqueeze/Linalg/Shape トレイトの実装

use ndarray::{Array0, Array2, Axis, Dimension, Ix0, IxDyn, concatenate};
use num_traits::One;

use crate::primops::{Expand, Matmul, Max, Permute, Prod, Reshape, Shape, Squeeze, Sum, Unsqueeze};

// ============================================================================
// GradRoot の実装 (Array0)
// ============================================================================

use crate::traits::GradRoot;

/// Array0<A> に GradRoot を実装することで
/// Variable<Array0<A>>::backward() が使用可能になる
impl<A> GradRoot for Array0<A>
where
    A: Clone + One,
{
    fn unit_grad() -> Self {
        Array0::from_elem(Ix0(), A::one())
    }
}

// ============================================================================
// Shape の実装
// ============================================================================

impl<A, D> Shape for ndarray::Array<A, D>
where
    D: Dimension,
{
    fn shape(&self) -> &[usize] {
        self.shape()
    }
}

// ============================================================================
// Sum の実装
// ============================================================================

impl<A, D> Sum for ndarray::Array<A, D>
where
    A: Clone + Default + std::ops::Add<Output = A>,
    D: Dimension + ndarray::RemoveAxis,
    D::Smaller: Dimension,
    <D::Smaller as Dimension>::Larger: Dimension,
{
    type Output = Self;

    fn sum(&self, axis: usize) -> Self::Output {
        // fold_axis で軸方向に合計 (次元が1つ減る)
        let summed = self.fold_axis(Axis(axis), A::default(), |acc, x| acc.clone() + x.clone());
        // insert_axis で次元を復元 (その軸のサイズは1になる)
        let result = summed.insert_axis(Axis(axis));
        // 型変換が必要な場合があるため、into_dimensionality で変換
        result
            .into_dimensionality()
            .expect("Dimension mismatch after sum")
    }
}

// ============================================================================
// Prod の実装
// ============================================================================

impl<A, D> Prod for ndarray::Array<A, D>
where
    A: Clone + num_traits::One + std::ops::Mul<Output = A>,
    D: Dimension + ndarray::RemoveAxis,
    D::Smaller: Dimension,
    <D::Smaller as Dimension>::Larger: Dimension,
{
    type Output = Self;

    fn prod(&self, axis: usize) -> Self::Output {
        let product = self.fold_axis(Axis(axis), A::one(), |acc, x| acc.clone() * x.clone());
        let result = product.insert_axis(Axis(axis));
        result
            .into_dimensionality()
            .expect("Dimension mismatch after prod")
    }
}

// ============================================================================
// Max の実装
// ============================================================================

impl<A, D> Max for ndarray::Array<A, D>
where
    A: Clone + PartialOrd + num_traits::Bounded + Default + std::ops::Sub<Output = A>,
    D: Dimension + ndarray::RemoveAxis,
    D::Smaller: Dimension,
    <D::Smaller as Dimension>::Larger: Dimension,
{
    type Output = Self;

    fn max(&self, axis: usize) -> Self::Output {
        let max_val = self.fold_axis(Axis(axis), A::min_value(), |acc, x| {
            if x > acc { x.clone() } else { acc.clone() }
        });
        let result = max_val.insert_axis(Axis(axis));
        result
            .into_dimensionality()
            .expect("Dimension mismatch after max")
    }

    fn max_grad(grad_output: &Self, input: &Self, output: &Self, axis: usize) -> Self {
        // 最大値の位置にのみ勾配を伝播するマスクを作成
        // output を expand して input と比較
        let size = input.shape()[axis];
        let expanded_output = output.expand(axis, size);
        let expanded_grad = grad_output.expand(axis, size);

        // input == max の位置のみ勾配を伝播
        let mut result = ndarray::Array::default(input.raw_dim());
        ndarray::Zip::from(&mut result)
            .and(input)
            .and(&expanded_output)
            .and(&expanded_grad)
            .for_each(|r, inp, out, g| {
                // 浮動小数点の比較: 近似的に等しいか
                let diff = inp.clone() - out.clone();
                // 差が十分小さければ最大値の位置とみなす
                *r = if diff == A::default() {
                    g.clone()
                } else {
                    A::default()
                };
            });
        result
    }
}

// ============================================================================
// Expand の実装
// ============================================================================

impl<A, D> Expand for ndarray::Array<A, D>
where
    A: Clone,
    D: Dimension + ndarray::RemoveAxis,
{
    type Output = Self;

    fn expand(&self, axis: usize, size: usize) -> Self::Output {
        // 指定した軸がサイズ1であることを前提とし、size回繰り返す
        assert_eq!(
            self.shape()[axis],
            1,
            "expand expects axis {} to have size 1, got {}",
            axis,
            self.shape()[axis]
        );

        if size == 1 {
            return self.clone();
        }

        // 自身を size 回連結
        let views: Vec<_> = (0..size).map(|_| self.view()).collect();
        concatenate(Axis(axis), &views).expect("Failed to concatenate for expand")
    }
}

// ============================================================================
// Permute の実装 (Array2 専用)
// ============================================================================

impl<A> Permute for Array2<A>
where
    A: Clone,
{
    fn permute(&self, axes: &[usize]) -> Self {
        assert_eq!(axes.len(), 2, "permute for Array2 requires exactly 2 axes");
        // 2次元の場合、axes = [0, 1] なら恒等、axes = [1, 0] なら転置
        if axes == [0, 1] {
            self.clone()
        } else if axes == [1, 0] {
            self.t().to_owned()
        } else {
            panic!("Invalid axes for 2D permute: {:?}", axes);
        }
    }
}

// ============================================================================
// Reshape の実装
// ============================================================================

impl<A, D> Reshape for ndarray::Array<A, D>
where
    A: Clone,
    D: Dimension,
{
    fn reshape(&self, new_shape: &[usize]) -> Self {
        // 動的次元に変換してから reshape し、元の次元型に戻す
        let dyn_array = self.clone().into_dyn();
        let new_dim = IxDyn(new_shape);
        let reshaped = dyn_array
            .into_shape_with_order(new_dim)
            .expect("reshape: incompatible shapes");
        reshaped
            .into_dimensionality()
            .expect("reshape: dimension mismatch")
    }
}

// ============================================================================
// Squeeze の実装
// ============================================================================

impl<A, D> Squeeze for ndarray::Array<A, D>
where
    A: Clone,
    D: Dimension + ndarray::RemoveAxis,
{
    type Output = ndarray::Array<A, D::Smaller>;

    fn squeeze(&self, axis: usize) -> Self::Output {
        assert_eq!(
            self.shape()[axis],
            1,
            "squeeze requires axis {} to have size 1, got {}",
            axis,
            self.shape()[axis]
        );
        // index_axis で軸を削除（次元が1つ減る）
        self.index_axis(Axis(axis), 0).to_owned()
    }
}

// ============================================================================
// Unsqueeze の実装
// ============================================================================

impl<A, D> Unsqueeze for ndarray::Array<A, D>
where
    A: Clone,
    D: Dimension,
    D::Larger: ndarray::RemoveAxis,
{
    type Output = ndarray::Array<A, D::Larger>;

    fn unsqueeze(&self, axis: usize) -> Self::Output {
        // insert_axis で軸を追加（次元が1つ増える）
        self.clone().insert_axis(Axis(axis))
    }
}

// ============================================================================
// Matmul の実装 (Array2 専用)
// ============================================================================

impl<A> Matmul for Array2<A>
where
    A: Clone + Default + std::ops::Add<Output = A> + std::ops::Mul<Output = A>,
{
    type Output = Array2<A>;

    fn matmul(&self, rhs: &Self) -> Self::Output {
        // (m, k) @ (k, n) -> (m, n)
        let m = self.nrows();
        let k = self.ncols();
        let n = rhs.ncols();

        assert_eq!(
            k,
            rhs.nrows(),
            "matmul dimension mismatch: ({}, {}) @ ({}, {})",
            m,
            k,
            rhs.nrows(),
            n
        );

        let mut result = Array2::default((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut sum = A::default();
                for l in 0..k {
                    sum = sum + self[[i, l]].clone() * rhs[[l, j]].clone();
                }
                result[[i, j]] = sum;
            }
        }
        result
    }
}

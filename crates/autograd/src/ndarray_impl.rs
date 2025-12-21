//! ndarray に対する Reduce/Expand/Linalg/Shape トレイトの実装

use ndarray::{Array2, Axis, Dimension, concatenate};

use crate::primops::{Expand, Matmul, Max, Prod, Shape, Sum, Transpose};

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
    fn sum(&self, axis: usize) -> Self {
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
    fn prod(&self, axis: usize) -> Self {
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
    fn max(&self, axis: usize) -> Self {
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
    fn expand(&self, axis: usize, size: usize) -> Self {
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
// Transpose の実装 (Array2 専用)
// ============================================================================

impl<A> Transpose for Array2<A>
where
    A: Clone,
{
    fn transpose(&self) -> Self {
        self.t().to_owned()
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

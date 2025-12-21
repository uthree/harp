//! 畳み込み操作の共通パラメータ
//!
//! conv/unfold/fold操作で使用されるパラメータを統一的に扱う構造体を提供します。

use crate::graph::shape::Expr;

/// スカラーやタプルをVec<usize>に変換するトレイト
///
/// 畳み込みパラメータ（stride, dilation, padding）を統一的に扱うために使用します。
pub trait IntoSpatialParams {
    fn into_vec(self) -> Vec<usize>;
}

impl IntoSpatialParams for usize {
    fn into_vec(self) -> Vec<usize> {
        vec![self]
    }
}

impl IntoSpatialParams for (usize,) {
    fn into_vec(self) -> Vec<usize> {
        vec![self.0]
    }
}

impl IntoSpatialParams for (usize, usize) {
    fn into_vec(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl IntoSpatialParams for (usize, usize, usize) {
    fn into_vec(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl IntoSpatialParams for Vec<usize> {
    fn into_vec(self) -> Vec<usize> {
        self
    }
}

impl IntoSpatialParams for &[usize] {
    fn into_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

/// N次元畳み込みパラメータ
///
/// kernel_size, stride, dilation, groupsを統一的に扱います。
/// 1D/2D/3Dの畳み込み操作で共通して使用できます。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvParams {
    /// カーネルサイズ (各空間次元)
    pub kernel_size: Vec<usize>,
    /// ストライド (各空間次元)
    pub stride: Vec<usize>,
    /// ディレーション (各空間次元)
    pub dilation: Vec<usize>,
    /// グループ数
    pub groups: usize,
}

impl ConvParams {
    /// 新しいConvParamsを作成
    ///
    /// # Panics
    /// kernel_size, stride, dilationの長さが一致しない場合
    pub fn new(
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    ) -> Self {
        assert_eq!(
            kernel_size.len(),
            stride.len(),
            "kernel_size and stride must have the same length"
        );
        assert_eq!(
            kernel_size.len(),
            dilation.len(),
            "kernel_size and dilation must have the same length"
        );
        assert!(groups > 0, "groups must be positive");
        Self {
            kernel_size,
            stride,
            dilation,
            groups,
        }
    }

    /// 空間次元数を取得
    pub fn ndim(&self) -> usize {
        self.kernel_size.len()
    }

    /// 実効カーネルサイズを計算
    ///
    /// dilationを考慮した実際のカーネルサイズを返します。
    /// effective_kernel_size = (kernel_size - 1) * dilation + 1
    pub fn effective_kernel_size(&self) -> Vec<usize> {
        self.kernel_size
            .iter()
            .zip(self.dilation.iter())
            .map(|(&k, &d)| (k - 1) * d + 1)
            .collect()
    }

    /// 出力サイズを計算
    ///
    /// 入力サイズから出力サイズを計算します（padding=0の場合）。
    /// output_size = (input_size - effective_kernel_size) / stride + 1
    pub fn output_size(&self, input_size: &[Expr]) -> Vec<Expr> {
        let eff_kernel = self.effective_kernel_size();
        input_size
            .iter()
            .zip(eff_kernel.iter())
            .zip(self.stride.iter())
            .map(|((input, &eff_k), &s)| {
                (input.clone() - Expr::from(eff_k as i64)) / Expr::from(s as i64) + Expr::from(1)
            })
            .collect()
    }

    /// 出力サイズを計算（usize版）
    pub fn output_size_usize(&self, input_size: &[usize]) -> Vec<usize> {
        let eff_kernel = self.effective_kernel_size();
        input_size
            .iter()
            .zip(eff_kernel.iter())
            .zip(self.stride.iter())
            .map(|((&input, &eff_k), &s)| (input - eff_k) / s + 1)
            .collect()
    }

    /// 1Dパラメータから作成
    pub fn from_1d(kernel_size: usize, stride: usize, dilation: usize, groups: usize) -> Self {
        Self::new(vec![kernel_size], vec![stride], vec![dilation], groups)
    }

    /// 2Dパラメータから作成
    pub fn from_2d(
        kernel_size: (usize, usize),
        stride: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        Self::new(
            vec![kernel_size.0, kernel_size.1],
            vec![stride.0, stride.1],
            vec![dilation.0, dilation.1],
            groups,
        )
    }

    /// 3Dパラメータから作成
    pub fn from_3d(
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        dilation: (usize, usize, usize),
        groups: usize,
    ) -> Self {
        Self::new(
            vec![kernel_size.0, kernel_size.1, kernel_size.2],
            vec![stride.0, stride.1, stride.2],
            vec![dilation.0, dilation.1, dilation.2],
            groups,
        )
    }

    /// groupsを変更した新しいConvParamsを作成
    pub fn with_groups(&self, groups: usize) -> Self {
        Self {
            kernel_size: self.kernel_size.clone(),
            stride: self.stride.clone(),
            dilation: self.dilation.clone(),
            groups,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_1d() {
        let params = ConvParams::from_1d(3, 1, 1, 1);
        assert_eq!(params.ndim(), 1);
        assert_eq!(params.kernel_size, vec![3]);
        assert_eq!(params.stride, vec![1]);
        assert_eq!(params.dilation, vec![1]);
        assert_eq!(params.groups, 1);
    }

    #[test]
    fn test_from_2d() {
        let params = ConvParams::from_2d((3, 5), (1, 2), (1, 1), 4);
        assert_eq!(params.ndim(), 2);
        assert_eq!(params.kernel_size, vec![3, 5]);
        assert_eq!(params.stride, vec![1, 2]);
        assert_eq!(params.groups, 4);
    }

    #[test]
    fn test_from_3d() {
        let params = ConvParams::from_3d((3, 3, 3), (2, 2, 2), (1, 1, 1), 1);
        assert_eq!(params.ndim(), 3);
        assert_eq!(params.kernel_size, vec![3, 3, 3]);
    }

    #[test]
    fn test_effective_kernel_size() {
        // dilation=1の場合、effective = kernel_size
        let params = ConvParams::from_2d((3, 3), (1, 1), (1, 1), 1);
        assert_eq!(params.effective_kernel_size(), vec![3, 3]);

        // dilation=2の場合、effective = (k-1)*d + 1 = 2*2 + 1 = 5
        let params = ConvParams::from_2d((3, 3), (1, 1), (2, 2), 1);
        assert_eq!(params.effective_kernel_size(), vec![5, 5]);
    }

    #[test]
    fn test_output_size_usize() {
        // input=10, kernel=3, stride=1, dilation=1 -> output = (10-3)/1 + 1 = 8
        let params = ConvParams::from_1d(3, 1, 1, 1);
        assert_eq!(params.output_size_usize(&[10]), vec![8]);

        // input=10, kernel=3, stride=2, dilation=1 -> output = (10-3)/2 + 1 = 4
        let params = ConvParams::from_1d(3, 2, 1, 1);
        assert_eq!(params.output_size_usize(&[10]), vec![4]);

        // input=10, kernel=3, stride=1, dilation=2 -> effective=5, output = (10-5)/1 + 1 = 6
        let params = ConvParams::from_1d(3, 1, 2, 1);
        assert_eq!(params.output_size_usize(&[10]), vec![6]);
    }

    #[test]
    #[should_panic(expected = "kernel_size and stride must have the same length")]
    fn test_mismatched_lengths() {
        ConvParams::new(vec![3, 3], vec![1], vec![1, 1], 1);
    }

    #[test]
    #[should_panic(expected = "groups must be positive")]
    fn test_zero_groups() {
        ConvParams::from_1d(3, 1, 1, 0);
    }
}

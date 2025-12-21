//! 次元トレイトと実装
//!
//! ndarrayクレートを参考に、静的次元と動的次元の両方をサポートします。
//!
//! # 使い分け
//! - `Dim0`, `Dim1`, `Dim2`, ... - コンパイル時に次元数が確定する場合
//! - `DimDyn` - 実行時に次元数が決まる場合（ファイル読み込みなど）

use std::fmt;

/// 次元数を表すトレイト
pub trait Dimension: Clone + fmt::Debug + 'static {
    /// 次元数。動的次元の場合はNone
    const NDIM: Option<usize>;

    /// 実行時の次元数を取得
    fn ndim(&self) -> usize;

    /// デフォルトインスタンスを作成（静的次元用）
    fn default_instance() -> Self
    where
        Self: Sized;
}

// ============================================================================
// 静的次元（コンパイル時に次元数が確定）
// ============================================================================

/// 0次元（スカラー）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim0;

impl Dimension for Dim0 {
    const NDIM: Option<usize> = Some(0);

    fn ndim(&self) -> usize {
        0
    }

    fn default_instance() -> Self {
        Self
    }
}

/// 1次元（ベクトル）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim1;

impl Dimension for Dim1 {
    const NDIM: Option<usize> = Some(1);

    fn ndim(&self) -> usize {
        1
    }

    fn default_instance() -> Self {
        Self
    }
}

/// 2次元（行列）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim2;

impl Dimension for Dim2 {
    const NDIM: Option<usize> = Some(2);

    fn ndim(&self) -> usize {
        2
    }

    fn default_instance() -> Self {
        Self
    }
}

/// 3次元テンソル
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim3;

impl Dimension for Dim3 {
    const NDIM: Option<usize> = Some(3);

    fn ndim(&self) -> usize {
        3
    }

    fn default_instance() -> Self {
        Self
    }
}

/// 4次元テンソル
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim4;

impl Dimension for Dim4 {
    const NDIM: Option<usize> = Some(4);

    fn ndim(&self) -> usize {
        4
    }

    fn default_instance() -> Self {
        Self
    }
}

/// 5次元テンソル
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim5;

impl Dimension for Dim5 {
    const NDIM: Option<usize> = Some(5);

    fn ndim(&self) -> usize {
        5
    }

    fn default_instance() -> Self {
        Self
    }
}

/// 6次元テンソル
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim6;

impl Dimension for Dim6 {
    const NDIM: Option<usize> = Some(6);

    fn ndim(&self) -> usize {
        6
    }

    fn default_instance() -> Self {
        Self
    }
}

// ============================================================================
// 動的次元（実行時に次元数が決まる）
// ============================================================================

/// 動的次元（ndarrayのIxDyn相当）
///
/// 実行時に次元数が決まる場合に使用します。
///
/// # Example
/// ```ignore
/// use array::{ArrayD, DimDyn};
///
/// // 次元数が実行時に決まる
/// let arr: ArrayD<f32> = load_from_file("data.npy")?;
/// println!("Loaded array with {} dimensions", arr.ndim());
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DimDyn(usize);

impl DimDyn {
    /// 指定した次元数で動的次元を作成
    pub fn new(ndim: usize) -> Self {
        Self(ndim)
    }

    /// 次元数を取得
    pub fn ndim_value(&self) -> usize {
        self.0
    }
}

impl Dimension for DimDyn {
    const NDIM: Option<usize> = None;

    fn ndim(&self) -> usize {
        self.0
    }

    fn default_instance() -> Self {
        Self(0)
    }
}

// ============================================================================
// 次元変換トレイト
// ============================================================================

/// 静的次元から動的次元への変換
pub trait IntoDyn {
    /// 動的次元に変換
    fn into_dyn(self) -> DimDyn;
}

impl<D: Dimension> IntoDyn for D {
    fn into_dyn(self) -> DimDyn {
        DimDyn::new(self.ndim())
    }
}

/// 動的次元から静的次元への変換（失敗可能）
pub trait IntoDimensionality<D: Dimension> {
    /// 指定した静的次元に変換
    ///
    /// 次元数が一致しない場合はエラーを返します。
    fn into_dimensionality(self) -> Result<D, DimensionMismatch>;
}

impl<D: Dimension> IntoDimensionality<D> for DimDyn {
    fn into_dimensionality(self) -> Result<D, DimensionMismatch> {
        if let Some(expected) = D::NDIM {
            if self.0 == expected {
                Ok(D::default_instance())
            } else {
                Err(DimensionMismatch {
                    expected,
                    actual: self.0,
                })
            }
        } else {
            // DimDyn -> DimDyn の場合は常に成功
            // 注: これは型システム上は発生しないが、安全のため
            Ok(D::default_instance())
        }
    }
}

/// 次元数の不一致エラー
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionMismatch {
    pub expected: usize,
    pub actual: usize,
}

impl fmt::Display for DimensionMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "dimension mismatch: expected {}, got {}",
            self.expected, self.actual
        )
    }
}

impl std::error::Error for DimensionMismatch {}

// ============================================================================
// 形状からの次元推論
// ============================================================================

/// 形状から次元を作成
pub trait ShapeIntoDim {
    type Dim: Dimension;

    fn shape_into_dim(&self) -> Self::Dim;
}

impl ShapeIntoDim for [usize; 0] {
    type Dim = Dim0;
    fn shape_into_dim(&self) -> Self::Dim {
        Dim0
    }
}

impl ShapeIntoDim for [usize; 1] {
    type Dim = Dim1;
    fn shape_into_dim(&self) -> Self::Dim {
        Dim1
    }
}

impl ShapeIntoDim for [usize; 2] {
    type Dim = Dim2;
    fn shape_into_dim(&self) -> Self::Dim {
        Dim2
    }
}

impl ShapeIntoDim for [usize; 3] {
    type Dim = Dim3;
    fn shape_into_dim(&self) -> Self::Dim {
        Dim3
    }
}

impl ShapeIntoDim for [usize; 4] {
    type Dim = Dim4;
    fn shape_into_dim(&self) -> Self::Dim {
        Dim4
    }
}

impl ShapeIntoDim for [usize; 5] {
    type Dim = Dim5;
    fn shape_into_dim(&self) -> Self::Dim {
        Dim5
    }
}

impl ShapeIntoDim for [usize; 6] {
    type Dim = Dim6;
    fn shape_into_dim(&self) -> Self::Dim {
        Dim6
    }
}

impl ShapeIntoDim for Vec<usize> {
    type Dim = DimDyn;
    fn shape_into_dim(&self) -> Self::Dim {
        DimDyn::new(self.len())
    }
}

impl ShapeIntoDim for &[usize] {
    type Dim = DimDyn;
    fn shape_into_dim(&self) -> Self::Dim {
        DimDyn::new(self.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_dimensions() {
        assert_eq!(Dim0::NDIM, Some(0));
        assert_eq!(Dim1::NDIM, Some(1));
        assert_eq!(Dim2::NDIM, Some(2));
        assert_eq!(Dim3::NDIM, Some(3));
        assert_eq!(Dim4::NDIM, Some(4));

        assert_eq!(Dim2.ndim(), 2);
    }

    #[test]
    fn test_dynamic_dimension() {
        let dyn_dim = DimDyn::new(3);
        assert_eq!(DimDyn::NDIM, None);
        assert_eq!(dyn_dim.ndim(), 3);
    }

    #[test]
    fn test_into_dyn() {
        let dim2 = Dim2;
        let dyn_dim = dim2.into_dyn();
        assert_eq!(dyn_dim.ndim(), 2);
    }

    #[test]
    fn test_into_dimensionality() {
        let dyn_dim = DimDyn::new(2);
        let result: Result<Dim2, _> = dyn_dim.into_dimensionality();
        assert!(result.is_ok());

        let dyn_dim = DimDyn::new(3);
        let result: Result<Dim2, _> = dyn_dim.into_dimensionality();
        assert!(result.is_err());
    }

    #[test]
    fn test_shape_into_dim() {
        let shape = [10, 20];
        let dim = shape.shape_into_dim();
        assert_eq!(dim.ndim(), 2);

        let shape = vec![10, 20, 30];
        let dim = shape.shape_into_dim();
        assert_eq!(dim.ndim(), 3);
    }
}

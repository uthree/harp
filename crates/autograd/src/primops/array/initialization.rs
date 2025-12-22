//! テンソル初期化トレイト
//!
//! 指定した形状のテンソルをゼロまたは1で初期化するためのトレイトを提供する。
//! torch.zeros() / torch.ones() に相当する機能。

// ============================================================================
// Zeros トレイト
// ============================================================================

/// ゼロで満たされた配列を作成するトレイト
///
/// # 使用例
///
/// ```
/// use autograd::primops::Zeros;
///
/// let zeros: f64 = Zeros::zeros(&[]);  // スカラー
/// assert_eq!(zeros, 0.0);
/// ```
pub trait Zeros: Sized {
    /// 指定した形状のゼロ埋め配列を作成
    ///
    /// # 引数
    ///
    /// * `shape` - 作成する配列の形状。スカラーの場合は空スライス `&[]` を指定。
    fn zeros(shape: &[usize]) -> Self;
}

// ============================================================================
// Ones トレイト
// ============================================================================

/// 1で満たされた配列を作成するトレイト
///
/// # 使用例
///
/// ```
/// use autograd::primops::Ones;
///
/// let ones: f64 = Ones::ones(&[]);  // スカラー
/// assert_eq!(ones, 1.0);
/// ```
pub trait Ones: Sized {
    /// 指定した形状の1埋め配列を作成
    ///
    /// # 引数
    ///
    /// * `shape` - 作成する配列の形状。スカラーの場合は空スライス `&[]` を指定。
    fn ones(shape: &[usize]) -> Self;
}

// ============================================================================
// スカラー型への実装
// ============================================================================

impl Zeros for f32 {
    fn zeros(_shape: &[usize]) -> Self {
        0.0
    }
}

impl Zeros for f64 {
    fn zeros(_shape: &[usize]) -> Self {
        0.0
    }
}

impl Ones for f32 {
    fn ones(_shape: &[usize]) -> Self {
        1.0
    }
}

impl Ones for f64 {
    fn ones(_shape: &[usize]) -> Self {
        1.0
    }
}

// ============================================================================
// Variable<T> への実装
// ============================================================================

use crate::Differentiable;
use crate::primops::Shape;
use crate::shape::IntoShape;

impl<T: Zeros + 'static> Differentiable<T> {
    /// 指定した形状のゼロ埋め変数を作成
    ///
    /// 勾配追跡なしで作成される（ReLU のゼロ境界などに使用）。
    ///
    /// # 使用例
    ///
    /// ```
    /// use autograd::Variable;
    /// use autograd::primops::Zeros;
    ///
    /// let zeros: Variable<f64> = Variable::zeros(&[]);
    /// assert_eq!(zeros.value(), 0.0);
    /// ```
    pub fn zeros(shape: &[usize]) -> Differentiable<T> {
        Differentiable::new_no_grad(T::zeros(shape))
    }

    /// 指定した形状のゼロ埋め変数を作成（IntoShape版）
    ///
    /// 配列、タプル、単一値など様々な形式で形状を指定できます。
    ///
    /// # 使用例
    ///
    /// ```
    /// use autograd::Variable;
    ///
    /// // スカラーの場合（空配列で形状を指定）
    /// let zeros: Variable<f64> = Variable::zeros_shape([]);
    /// assert_eq!(zeros.value(), 0.0);
    /// ```
    ///
    /// ndarray featureを有効にした場合は多次元配列も使用可能：
    /// ```ignore
    /// use autograd::Variable;
    /// use ndarray::Array2;
    /// let zeros: Variable<Array2<f64>> = Variable::zeros_shape([3, 4]);
    /// ```
    pub fn zeros_shape<S: IntoShape>(shape: S) -> Differentiable<T> {
        Differentiable::new_no_grad(T::zeros(&shape.into_shape()))
    }
}

impl<T: Ones + 'static> Differentiable<T> {
    /// 指定した形状の1埋め変数を作成
    ///
    /// 勾配追跡なしで作成される。
    ///
    /// # 使用例
    ///
    /// ```
    /// use autograd::Variable;
    /// use autograd::primops::Ones;
    ///
    /// let ones: Variable<f64> = Variable::ones(&[]);
    /// assert_eq!(ones.value(), 1.0);
    /// ```
    pub fn ones(shape: &[usize]) -> Differentiable<T> {
        Differentiable::new_no_grad(T::ones(shape))
    }

    /// 指定した形状の1埋め変数を作成（IntoShape版）
    ///
    /// 配列、タプル、単一値など様々な形式で形状を指定できます。
    ///
    /// # 使用例
    ///
    /// ```
    /// use autograd::Variable;
    ///
    /// // スカラーの場合（空配列で形状を指定）
    /// let ones: Variable<f64> = Variable::ones_shape([]);
    /// assert_eq!(ones.value(), 1.0);
    /// ```
    ///
    /// ndarray featureを有効にした場合は多次元配列も使用可能：
    /// ```ignore
    /// use autograd::Variable;
    /// use ndarray::Array2;
    /// let ones: Variable<Array2<f64>> = Variable::ones_shape([3, 4]);
    /// ```
    pub fn ones_shape<S: IntoShape>(shape: S) -> Differentiable<T> {
        Differentiable::new_no_grad(T::ones(&shape.into_shape()))
    }
}

// ============================================================================
// zeros_like / ones_like
// ============================================================================

impl<T: Clone + Zeros + Shape + 'static> Differentiable<T> {
    /// 自身と同じ形状のゼロ埋め変数を作成
    ///
    /// `torch.zeros_like()` に相当する機能。
    /// 勾配追跡なしで作成される。
    ///
    /// # 使用例
    ///
    /// ```
    /// use autograd::Variable;
    /// use ndarray::array;
    ///
    /// let x = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    /// let zeros = x.zeros_like();
    /// assert_eq!(zeros.value(), array![[0.0, 0.0], [0.0, 0.0]]);
    /// ```
    #[cfg(feature = "ndarray")]
    pub fn zeros_like(&self) -> Differentiable<T> {
        Differentiable::new_no_grad(T::zeros(self.value().shape()))
    }

    /// 自身と同じ形状のゼロ埋め変数を作成（スカラー用）
    #[cfg(not(feature = "ndarray"))]
    pub fn zeros_like(&self) -> Differentiable<T> {
        Differentiable::new_no_grad(T::zeros(&[]))
    }
}

impl<T: Clone + Ones + Shape + 'static> Differentiable<T> {
    /// 自身と同じ形状の1埋め変数を作成
    ///
    /// `torch.ones_like()` に相当する機能。
    /// 勾配追跡なしで作成される。
    ///
    /// # 使用例
    ///
    /// ```
    /// use autograd::Variable;
    /// use ndarray::array;
    ///
    /// let x = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
    /// let ones = x.ones_like();
    /// assert_eq!(ones.value(), array![[1.0, 1.0], [1.0, 1.0]]);
    /// ```
    #[cfg(feature = "ndarray")]
    pub fn ones_like(&self) -> Differentiable<T> {
        Differentiable::new_no_grad(T::ones(self.value().shape()))
    }

    /// 自身と同じ形状の1埋め変数を作成（スカラー用）
    #[cfg(not(feature = "ndarray"))]
    pub fn ones_like(&self) -> Differentiable<T> {
        Differentiable::new_no_grad(T::ones(&[]))
    }
}

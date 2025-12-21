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

use crate::Variable;

impl<T: Zeros + 'static> Variable<T> {
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
    pub fn zeros(shape: &[usize]) -> Variable<T> {
        Variable::new_no_grad(T::zeros(shape))
    }
}

impl<T: Ones + 'static> Variable<T> {
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
    pub fn ones(shape: &[usize]) -> Variable<T> {
        Variable::new_no_grad(T::ones(shape))
    }
}

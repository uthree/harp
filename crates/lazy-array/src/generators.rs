//! 形状指定ユーティリティ
//!
//! 配列の形状を柔軟に指定できるようにするトレイトを提供します。
//! 生成関数は `Array::<f32>::zeros(shape)` のようにArrayの関連関数として使用します。

use harp_core::graph::shape::Expr;

// ============================================================================
// IntoShape トレイト - 形状指定を柔軟に
// ============================================================================

/// 形状に変換可能な型
///
/// 配列やスライスから形状を生成できるようにします。
///
/// # 使用例
///
/// ```
/// use harp_lazy_array::generators::IntoShape;
///
/// // 配列から
/// let shape: Vec<usize> = [3, 4].into_shape();
/// assert_eq!(shape, vec![3, 4]);
///
/// // タプルから
/// let shape: Vec<usize> = (3, 4).into_shape();
/// assert_eq!(shape, vec![3, 4]);
///
/// // 単一値から（1次元配列用）
/// let shape: Vec<usize> = 10usize.into_shape();
/// assert_eq!(shape, vec![10]);
/// ```
pub trait IntoShape {
    /// 形状のベクタに変換
    fn into_shape(self) -> Vec<usize>;

    /// 形状のExprベクタに変換
    fn into_shape_exprs(self) -> Vec<Expr>
    where
        Self: Sized,
    {
        self.into_shape()
            .into_iter()
            .map(|s| Expr::from(s as isize))
            .collect()
    }
}

impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

// 単一次元用
impl IntoShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

// スカラー用（空の形状）
impl IntoShape for () {
    fn into_shape(self) -> Vec<usize> {
        vec![]
    }
}

// タプル実装
impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

impl IntoShape for (usize, usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

impl IntoShape for (usize, usize, usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4, self.5]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_shape_array() {
        let shape: Vec<usize> = [3, 4].into_shape();
        assert_eq!(shape, vec![3, 4]);
    }

    #[test]
    fn test_into_shape_vec() {
        let shape: Vec<usize> = vec![2, 3, 4].into_shape();
        assert_eq!(shape, vec![2, 3, 4]);
    }

    #[test]
    fn test_into_shape_tuple() {
        let shape: Vec<usize> = (5, 6).into_shape();
        assert_eq!(shape, vec![5, 6]);

        let shape3: Vec<usize> = (2, 3, 4).into_shape();
        assert_eq!(shape3, vec![2, 3, 4]);
    }

    #[test]
    fn test_into_shape_single() {
        let shape: Vec<usize> = 10usize.into_shape();
        assert_eq!(shape, vec![10]);
    }

    #[test]
    fn test_into_shape_scalar() {
        let shape: Vec<usize> = ().into_shape();
        assert!(shape.is_empty());
    }

    #[test]
    fn test_into_shape_exprs() {
        let exprs = [3usize, 4].into_shape_exprs();
        assert_eq!(exprs.len(), 2);
        assert_eq!(exprs[0].as_const(), Some(3));
        assert_eq!(exprs[1].as_const(), Some(4));
    }
}

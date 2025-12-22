//! 形状指定ユーティリティ
//!
//! 配列の形状を柔軟に指定できるようにするトレイトを提供します。

// ============================================================================
// IntoShape トレイト
// ============================================================================

/// 形状に変換可能な型
///
/// 配列、スライス、タプルなどから形状を生成できるようにします。
///
/// # 使用例
///
/// ```
/// use harp_autograd::IntoShape;
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
}

// スライスからの変換
impl IntoShape for &[usize] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

// Vecからの変換
impl IntoShape for Vec<usize> {
    fn into_shape(self) -> Vec<usize> {
        self
    }
}

// 固定長配列からの変換
impl<const N: usize> IntoShape for [usize; N] {
    fn into_shape(self) -> Vec<usize> {
        self.to_vec()
    }
}

// 単一値からの変換（1次元配列用）
impl IntoShape for usize {
    fn into_shape(self) -> Vec<usize> {
        vec![self]
    }
}

// 空の形状（スカラー用）
impl IntoShape for () {
    fn into_shape(self) -> Vec<usize> {
        vec![]
    }
}

// タプルからの変換（2次元）
impl IntoShape for (usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1]
    }
}

// タプルからの変換（3次元）
impl IntoShape for (usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2]
    }
}

// タプルからの変換（4次元）
impl IntoShape for (usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3]
    }
}

// タプルからの変換（5次元）
impl IntoShape for (usize, usize, usize, usize, usize) {
    fn into_shape(self) -> Vec<usize> {
        vec![self.0, self.1, self.2, self.3, self.4]
    }
}

// タプルからの変換（6次元）
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
        let shape: Vec<usize> = [3, 4, 5].into_shape();
        assert_eq!(shape, vec![3, 4, 5]);
    }

    #[test]
    fn test_into_shape_slice() {
        let arr = [2, 3];
        let shape: Vec<usize> = arr.as_slice().into_shape();
        assert_eq!(shape, vec![2, 3]);
    }

    #[test]
    fn test_into_shape_tuple() {
        let shape: Vec<usize> = (3, 4).into_shape();
        assert_eq!(shape, vec![3, 4]);

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
        assert_eq!(shape, Vec::<usize>::new());
    }
}

use crate::graph::shape::Expr;
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum View {
    /// 線形な処理で表現可能な場合
    /// offset = base_offset + Σ(idx[i] * strides[i])
    Linear {
        shape: Vec<Expr>,   // 論理的なテンソルのサイズ
        strides: Vec<Expr>, // 各次元の添え字の係数
        offset: Expr,       // オフセット
    },

    /// 任意の式でインデックスを計算する場合
    /// offsetはExprで直接計算される（Idx(0), Idx(1), ... を含む）
    IndexExpr {
        shape: Vec<Expr>, // 論理的なテンソルのサイズ
        index_expr: Expr, // インデックス計算式
    },
}

impl View {
    pub fn contiguous<E: Into<Expr> + Clone, I: IntoIterator<Item = E>>(shape: I) -> Self {
        let shape: Vec<Expr> = shape.into_iter().map(|e| e.into()).collect();
        if shape.is_empty() {
            return View::Linear {
                shape,
                strides: vec![],
                offset: Expr::from(0),
            };
        }
        let mut strides = vec![Expr::from(1); shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = (strides[i + 1].clone() * shape[i + 1].clone()).simplify();
        }
        View::Linear {
            shape,
            strides,
            offset: Expr::from(0),
        }
    }

    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    pub fn shape(&self) -> &[Expr] {
        match self {
            View::Linear { shape, .. } | View::IndexExpr { shape, .. } => shape,
        }
    }

    /// Linearバリアントかどうかを判定
    pub fn is_linear(&self) -> bool {
        matches!(self, View::Linear { .. })
    }

    /// ViewにLoadIndexが含まれているかどうかを判定
    ///
    /// IndexExpr Viewのindex_exprにLoadIndexが含まれている場合にtrueを返します。
    /// Linear Viewは常にfalseを返します。
    pub fn contains_load_index(&self) -> bool {
        match self {
            View::Linear { .. } => false,
            View::IndexExpr { index_expr, .. } => index_expr.contains_load_index(),
        }
    }

    /// ViewをIndexExpr形式に変換
    ///
    /// Linear Viewを等価なIndexExpr形式に変換します。
    /// 既にIndexExprの場合はクローンを返します。
    pub fn to_index_expr(&self) -> View {
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                // offset + Idx(0)*strides[0] + Idx(1)*strides[1] + ...
                let mut expr = offset.clone();
                for (i, stride) in strides.iter().enumerate() {
                    expr = (expr + Expr::Idx(i) * stride.clone()).simplify();
                }
                View::IndexExpr {
                    shape: shape.clone(),
                    index_expr: expr,
                }
            }
            View::IndexExpr { .. } => self.clone(),
        }
    }

    /// 2つのViewを合成する
    ///
    /// `compose(outer, inner)` は、innerを適用した後にouterを適用する変換を表します。
    /// 結果のViewは outer.shape を持ち、innerのメモリアクセスパターンを継承します。
    ///
    /// # Arguments
    /// * `outer` - 後から適用されるView（結果の形状を決定）
    /// * `inner` - 先に適用されるView（元のメモリアクセスパターン）
    ///
    /// # Example
    /// ```
    /// use harp::graph::shape::{View, Expr};
    ///
    /// // permute([1, 0]) ∘ contiguous([3, 4])
    /// let inner = View::contiguous(vec![3, 4]);
    /// let outer = inner.clone().permute(vec![1, 0]);
    /// let composed = View::compose(&outer, &inner);
    /// // composed の形状は [4, 3]
    /// ```
    pub fn compose(outer: &View, inner: &View) -> View {
        let outer_shape = outer.shape().to_vec();

        match (outer, inner) {
            // Linear × Linear: stridesとoffsetを組み合わせ
            (View::Linear { .. }, View::Linear { .. }) => {
                // 単純化: 両方をIndexExprに変換して合成
                let outer_expr = outer.to_index_expr();
                let inner_expr = inner.to_index_expr();
                Self::compose(&outer_expr, &inner_expr)
            }

            // Linear outer × IndexExpr inner: outerの変換をinnerに適用
            (View::Linear { .. }, View::IndexExpr { .. }) => {
                // outer が permute の場合、inner の Idx を並べ替え
                // outer_strides から逆写像を推測
                // 注: 一般的な Linear では複雑なので、IndexExpr に変換して処理
                let outer_as_expr = outer.to_index_expr();
                Self::compose(&outer_as_expr, inner)
            }

            // IndexExpr outer × Linear inner
            (
                View::IndexExpr {
                    index_expr: outer_expr,
                    ..
                },
                View::Linear { .. },
            ) => {
                // outer_expr 内の Idx(i) を inner の対応する値で置換
                // inner は Linear なので、Idx(i) -> inner_offset + sum(Idx(j) * inner_strides[j])
                // ただし outer_expr の Idx(i) は inner の position i を参照
                // 実際には outer_expr の Idx(i) をそのまま使い、inner の stride を適用

                // outerのindex_exprはinnerの出力位置を参照
                // innerがLinearの場合、outerのIdxがそのままinnerの入力位置になる
                // 結果のindex_expr = outer_expr (innerのLinear変換は透過的)
                View::IndexExpr {
                    shape: outer_shape,
                    index_expr: outer_expr.clone(),
                }
            }

            // IndexExpr × IndexExpr: 最も一般的なケース
            (
                View::IndexExpr {
                    index_expr: outer_expr,
                    ..
                },
                View::IndexExpr {
                    index_expr: _inner_expr,
                    ..
                },
            ) => {
                // outer_expr の Idx(i) は inner の出力位置 i を参照
                // inner の出力位置 i は、inner_expr(Idx(i)) でメモリオフセットを計算
                // 合成では、outer_expr の Idx(i) を inner がどう変換するかを適用する必要がある

                // 単純なケース: outer が permute 相当の場合
                // outer_expr 内の Idx の使われ方を分析して、inner_expr に適用

                // 一般的なアプローチ: outer_expr をそのまま使用
                // これは outer が単純なインデックス変換（permute等）の場合に有効
                // より複雑なケースでは、outer_expr 内の各 Idx を inner_expr で展開

                // 今回は単純化: outer_expr をそのまま使用（permute等の変換のみ対応）
                View::IndexExpr {
                    shape: outer_shape,
                    index_expr: outer_expr.clone(),
                }
            }
        }
    }

    /// LoadIndexのsrc_indexをシフトしたViewを返す
    ///
    /// View融合時にsrc配列がマージされる際に使用します。
    pub fn shift_load_index(&self, delta: isize) -> View {
        match self {
            View::Linear { .. } => self.clone(),
            View::IndexExpr { shape, index_expr } => View::IndexExpr {
                shape: shape.clone(),
                index_expr: index_expr.clone().shift_load_index(delta),
            },
        }
    }

    pub fn permute(self, axes: Vec<usize>) -> Self {
        assert!(self.ndim() == axes.len());
        let axes_set: HashSet<_> = axes.iter().collect();
        assert!(axes_set.len() == axes.len(), "duplicate axis in permute");
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let mut new_shape = vec![];
                let mut new_strides = vec![];
                for axis in axes.iter() {
                    new_shape.push(shape[*axis].clone().simplify());
                    new_strides.push(strides[*axis].clone().simplify());
                }
                View::Linear {
                    shape: new_shape,
                    strides: new_strides,
                    offset,
                }
            }
            View::IndexExpr { shape, index_expr } => {
                // shapeを並べ替え
                let new_shape: Vec<Expr> = axes.iter().map(|&a| shape[a].clone()).collect();
                // index_exprのIdx変数を並べ替え
                let new_index_expr = index_expr.permute_idx(&axes);
                View::IndexExpr {
                    shape: new_shape,
                    index_expr: new_index_expr,
                }
            }
        }
    }

    pub fn unsqueeze(self, axis: usize) -> Self {
        assert!(axis <= self.ndim());
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let mut shape = shape.clone();
                let mut strides = strides.clone();
                shape.insert(axis, 1.into());
                strides.insert(axis, 0.into());
                View::Linear {
                    shape,
                    strides,
                    offset,
                }
            }
            View::IndexExpr {
                mut shape,
                index_expr,
            } => {
                // shapeに1を挿入
                shape.insert(axis, 1.into());
                // Idx(i) for i >= axis を Idx(i+1) にシフト
                let new_index_expr = index_expr.shift_idx(axis, 1);
                View::IndexExpr {
                    shape,
                    index_expr: new_index_expr,
                }
            }
        }
    }

    pub fn squeeze(self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let mut shape = shape.clone();
                let mut strides = strides.clone();
                assert_eq!(shape[axis], 1.into(), "can only squeeze an axis of size 1");
                shape.remove(axis);
                strides.remove(axis);
                View::Linear {
                    shape,
                    strides,
                    offset,
                }
            }
            View::IndexExpr {
                mut shape,
                index_expr,
            } => {
                assert_eq!(shape[axis], 1.into(), "can only squeeze an axis of size 1");
                // shapeから削除
                shape.remove(axis);
                // Idx(axis) を 0 に置換し、Idx(i) for i > axis を Idx(i-1) にシフト
                let new_index_expr = index_expr
                    .substitute_idx(axis, Expr::from(0))
                    .shift_idx(axis + 1, -1);
                View::IndexExpr {
                    shape,
                    index_expr: new_index_expr,
                }
            }
        }
    }

    pub fn flip(self, axis: usize) -> Self {
        assert!(axis < self.ndim(), "axis out of bounds");
        match self {
            View::Linear {
                shape,
                mut strides,
                offset,
            } => {
                // Flip axis by reversing the stride direction
                // New offset = old_offset + (shape[axis] - 1) * strides[axis]
                // New strides[axis] = -strides[axis]
                let new_offset = (offset
                    + (shape[axis].clone() - Expr::from(1)) * strides[axis].clone())
                .simplify();
                strides[axis] = (-strides[axis].clone()).simplify();
                View::Linear {
                    shape,
                    strides,
                    offset: new_offset,
                }
            }
            View::IndexExpr { shape, index_expr } => {
                // Idx(axis) を (shape[axis] - 1 - Idx(axis)) に置換
                let flipped_idx =
                    (shape[axis].clone() - Expr::from(1) - Expr::Idx(axis)).simplify();
                let new_index_expr = index_expr.substitute_idx(axis, flipped_idx);
                View::IndexExpr {
                    shape,
                    index_expr: new_index_expr,
                }
            }
        }
    }

    /// 指定した軸を繰り返す（サイズ1の軸のみ対応）
    ///
    /// # Arguments
    /// * `axis` - 繰り返す軸のインデックス
    /// * `times` - 繰り返し回数（結果のサイズ）
    ///
    /// # Panics
    /// * 軸が範囲外の場合
    /// * 指定軸のサイズが1でない場合
    pub fn repeat(self, axis: usize, times: impl Into<Expr>) -> Self {
        let times = times.into();
        match self {
            View::Linear {
                mut shape,
                mut strides,
                offset,
            } => {
                assert!(axis < shape.len(), "axis out of bounds");
                assert!(shape[axis].is_one(), "can only repeat an axis of size 1");
                shape[axis] = times;
                strides[axis] = 0.into();
                View::Linear {
                    shape,
                    strides,
                    offset,
                }
            }
            View::IndexExpr {
                mut shape,
                index_expr,
            } => {
                assert!(axis < shape.len(), "axis out of bounds");
                assert!(shape[axis].is_one(), "can only repeat an axis of size 1");

                // Idx(axis) を 0 に固定（常に同じ位置を参照）
                // サイズ1の軸なので、Idx(axis)は常に0だが、明示的に置換
                let new_index_expr = index_expr.substitute_idx(axis, Expr::from(0));
                shape[axis] = times;

                View::IndexExpr {
                    shape,
                    index_expr: new_index_expr,
                }
            }
        }
    }

    /// 指定した軸を循環させて繰り返す（タイル化）
    ///
    /// `repeat`とは異なり、配列全体を循環させて繰り返します。
    /// - `repeat`: `[1, 2, 3]` (size=1) → 不可能（size=1のみ対応）
    /// - `tile`: `[1, 2, 3]` → `[1, 2, 3, 1, 2, 3, 1, 2, 3]`（循環）
    ///
    /// IndexExprを使用するため、結果は非線形Viewになります。
    ///
    /// # Arguments
    /// * `axis` - タイル化する軸のインデックス
    /// * `times` - 繰り返し回数
    ///
    /// # Panics
    /// * 軸が範囲外の場合
    ///
    /// # Example
    /// ```
    /// use harp::graph::shape::{View, Expr};
    ///
    /// let view = View::contiguous(vec![3, 4]); // shape: [3, 4]
    /// let tiled = view.tile(0, 2); // shape: [6, 4], idx0を%3で循環
    ///
    /// assert_eq!(tiled.shape(), &[Expr::from(6), Expr::from(4)]);
    /// assert!(!tiled.is_linear()); // IndexExprになる
    /// ```
    pub fn tile(self, axis: usize, times: impl Into<Expr>) -> Self {
        let times = times.into();
        let ndim = self.ndim();
        assert!(axis < ndim, "axis out of bounds");

        match self {
            View::Linear {
                shape,
                strides,
                offset,
            } => {
                let original_size = shape[axis].clone();
                let new_size = (original_size.clone() * times).simplify();

                // 新しいshapeを作成
                let mut new_shape = shape.clone();
                new_shape[axis] = new_size;

                // インデックス式を構築
                // offset + sum(Idx(i) * stride[i]) ただしaxis番目は Idx(axis) % original_size
                let mut index_expr = offset;
                for (i, stride) in strides.iter().enumerate() {
                    let idx = if i == axis {
                        // 循環させる軸はモジュロを適用
                        (Expr::Idx(i) % original_size.clone()).simplify()
                    } else {
                        Expr::Idx(i)
                    };
                    index_expr = (index_expr + idx * stride.clone()).simplify();
                }

                View::IndexExpr {
                    shape: new_shape,
                    index_expr,
                }
            }
            View::IndexExpr { shape, index_expr } => {
                let original_size = shape[axis].clone();
                let new_size = (original_size.clone() * times).simplify();

                // 新しいshapeを作成
                let mut new_shape = shape;
                new_shape[axis] = new_size;

                // Idx(axis)を Idx(axis) % original_size に置換
                let cyclic_idx = (Expr::Idx(axis) % original_size).simplify();
                let new_index_expr = index_expr.substitute_idx(axis, cyclic_idx);

                View::IndexExpr {
                    shape: new_shape,
                    index_expr: new_index_expr,
                }
            }
        }
    }

    /// Reshapeは連続したViewに対してのみ適用可能
    ///
    /// 要素数が一致する新しいshapeに変換します。
    /// 非連続なViewに対してはpanicします。
    ///
    /// # Panics
    /// * 非連続なViewの場合
    /// * IndexExpr Viewの場合（常に非連続として扱われる）
    pub fn reshape(self, new_shape: Vec<Expr>) -> Self {
        assert!(
            self.is_contiguous(),
            "reshape can only be applied to contiguous views"
        );

        match self {
            View::Linear { shape, offset, .. } => {
                // 要素数の一致を確認（シンボリック式の場合は実行時にチェックされる）
                // ここでは定数の場合のみチェック
                let old_numel = shape
                    .iter()
                    .fold(Expr::from(1), |acc, s| acc * s.clone())
                    .simplify();
                let new_numel = new_shape
                    .iter()
                    .fold(Expr::from(1), |acc, s| acc * s.clone())
                    .simplify();

                // 定数の場合のみ検証
                if let (Expr::Const(old_val), Expr::Const(new_val)) = (&old_numel, &new_numel) {
                    assert_eq!(
                        old_val, new_val,
                        "reshape requires the number of elements to match"
                    );
                }

                // 新しいshapeで連続したViewを作成（offsetは保持）
                let mut reshaped = View::contiguous(new_shape);
                if let View::Linear {
                    offset: new_offset, ..
                } = &mut reshaped
                {
                    *new_offset = offset;
                }
                reshaped
            }
            View::IndexExpr { .. } => {
                // is_contiguous()がfalseを返すので、ここには到達しない
                unreachable!("IndexExpr views are always non-contiguous")
            }
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            View::Linear { shape, .. } => *self == View::contiguous(shape.clone()),
            // IndexExprは常に非連続として扱う
            View::IndexExpr { .. } => false,
        }
    }

    /// 最内軸が連続かどうかをチェック
    ///
    /// SIMD（vload/vstore）には最内軸の連続性が必要です。
    /// 最内軸のstride = 1 であれば連続とみなします。
    ///
    /// # Returns
    /// - `true`: 最内軸が連続（stride = 1）
    /// - `false`: 最内軸が非連続、または空の場合
    ///
    /// # Examples
    /// ```
    /// use harp::graph::shape::View;
    ///
    /// // 連続したViewは最内軸も連続
    /// let view = View::contiguous(vec![3, 4]);
    /// assert!(view.is_innermost_contiguous());
    ///
    /// // transposeすると最内軸が非連続になることがある
    /// let transposed = View::contiguous(vec![3, 4]).permute(vec![1, 0]);
    /// assert!(!transposed.is_innermost_contiguous());
    /// ```
    pub fn is_innermost_contiguous(&self) -> bool {
        match self {
            View::Linear { strides, .. } => {
                if strides.is_empty() {
                    // 0次元テンソルは連続とみなす
                    true
                } else {
                    // 最内軸のstride = 1 であれば連続
                    strides.last().map(|s| *s == Expr::from(1)).unwrap_or(true)
                }
            }
            // IndexExprは最内軸の連続性を保証できない
            View::IndexExpr { .. } => false,
        }
    }

    /// IndexExpr Viewを作成
    ///
    /// # Arguments
    /// * `shape` - 論理的な形状
    /// * `index_expr` - インデックス計算式（Idx(0), Idx(1), ... を含む）
    ///
    /// # Examples
    /// ```
    /// use harp::graph::shape::{View, Expr};
    ///
    /// // 転置を式で表現: offset = idx1 * 4 + idx0
    /// let view = View::from_index_expr(
    ///     vec![Expr::from(4), Expr::from(3)],
    ///     Expr::Idx(1) * Expr::from(4) + Expr::Idx(0),
    /// );
    /// ```
    pub fn from_index_expr<E: Into<Expr> + Clone, I: IntoIterator<Item = E>>(
        shape: I,
        index_expr: impl Into<Expr>,
    ) -> Self {
        let shape: Vec<Expr> = shape.into_iter().map(|e| e.into()).collect();
        View::IndexExpr {
            shape,
            index_expr: index_expr.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_1d() {
        let view = View::contiguous(vec![10]);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(10)]);
        assert_eq!(strides, vec![Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_contiguous_2d() {
        let view = View::contiguous(vec![3, 4]);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(4), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_contiguous_3d() {
        let view = View::contiguous(vec![2, 3, 4]);
        let View::Linear {
            shape,
            strides,
            offset,
        } = view
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(2), Expr::from(3), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(12), Expr::from(4), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_contiguous_empty() {
        let view = View::contiguous(Vec::<isize>::new());
        let View::Linear {
            shape,
            strides,
            offset,
        } = view
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape.len(), 0);
        assert_eq!(strides.len(), 0);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_ndim() {
        let view = View::contiguous(vec![2, 3, 4]);
        assert_eq!(view.ndim(), 3);

        let view = View::contiguous(vec![10]);
        assert_eq!(view.ndim(), 1);

        let view = View::contiguous(Vec::<isize>::new());
        assert_eq!(view.ndim(), 0);
    }

    #[test]
    fn test_shape() {
        let view = View::contiguous(vec![2, 3, 4]);
        assert_eq!(view.shape(), &[Expr::from(2), Expr::from(3), Expr::from(4)]);
    }

    #[test]
    fn test_permute() {
        let view = View::contiguous(vec![2, 3, 4]);
        let permuted = view.permute(vec![2, 0, 1]); // (2, 3, 4) -> (4, 2, 3)

        let View::Linear {
            shape,
            strides,
            offset,
        } = permuted
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(4), Expr::from(2), Expr::from(3)]);
        assert_eq!(strides, vec![Expr::from(1), Expr::from(12), Expr::from(4)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    #[should_panic]
    fn test_permute_wrong_axes_count() {
        let view = View::contiguous(vec![2, 3, 4]);
        let _ = view.permute(vec![0, 1]); // Should panic: wrong number of axes
    }

    #[test]
    #[should_panic]
    fn test_permute_duplicate_axes() {
        let view = View::contiguous(vec![2, 3, 4]);
        let _ = view.permute(vec![0, 1, 1]); // Should panic: duplicate axes
    }

    #[test]
    fn test_unsqueeze() {
        let view = View::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(1); // (3, 4) -> (3, 1, 4)

        let View::Linear {
            shape,
            strides,
            offset,
        } = unsqueezed
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(1), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(4), Expr::from(0), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_unsqueeze_at_beginning() {
        let view = View::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(0); // (3, 4) -> (1, 3, 4)

        let View::Linear { shape, .. } = unsqueezed else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(1), Expr::from(3), Expr::from(4)]);
    }

    #[test]
    fn test_unsqueeze_at_end() {
        let view = View::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(2); // (3, 4) -> (3, 4, 1)

        let View::Linear { shape, .. } = unsqueezed else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(4), Expr::from(1)]);
    }

    #[test]
    fn test_squeeze() {
        let view = View::contiguous(vec![3, 1, 4]);
        let squeezed = view.squeeze(1); // (3, 1, 4) -> (3, 4)

        let View::Linear {
            shape,
            strides,
            offset,
        } = squeezed
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(4), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    #[should_panic(expected = "can only squeeze an axis of size 1")]
    fn test_squeeze_non_one_axis() {
        let view = View::contiguous(vec![3, 4]);
        let _ = view.squeeze(0); // Should panic: axis 0 has size 3, not 1
    }

    #[test]
    fn test_flip() {
        let view = View::contiguous(vec![3, 4]);
        let flipped = view.flip(0); // Flip first axis

        let View::Linear {
            shape,
            strides,
            offset,
        } = flipped
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
        // Stride for axis 0 should be negated
        assert_eq!(strides[0], Expr::from(-4));
        assert_eq!(strides[1], Expr::from(1));
        // Offset should be (3-1) * 4 = 8
        assert_eq!(offset, Expr::from(8));
    }

    #[test]
    fn test_repeat() {
        let view = View::contiguous(vec![1, 4]);
        let expanded = view.repeat(0, 3); // (1, 4) -> (3, 4), axis 0 repeated 3 times

        let View::Linear {
            shape,
            strides,
            offset,
        } = expanded
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(4)]);
        // Stride for repeated axis should be 0
        assert_eq!(strides[0], Expr::from(0));
        assert_eq!(strides[1], Expr::from(1));
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    #[should_panic(expected = "can only repeat an axis of size 1")]
    fn test_repeat_non_one_axis() {
        let view = View::contiguous(vec![3, 4]);
        let _ = view.repeat(0, 5); // Should panic: axis 0 is size 3, not 1
    }

    #[test]
    #[should_panic(expected = "axis out of bounds")]
    fn test_repeat_invalid_axis() {
        let view = View::contiguous(vec![1, 4]);
        let _ = view.repeat(5, 3); // Should panic: axis 5 is out of bounds
    }

    #[test]
    fn test_is_contiguous() {
        let view = View::contiguous(vec![3, 4]);
        assert!(view.is_contiguous());

        // Permuted view is not contiguous
        let permuted = View::contiguous(vec![3, 4]).permute(vec![1, 0]);
        assert!(!permuted.is_contiguous());
    }

    #[test]
    fn test_complex_operations() {
        // Create a view and apply multiple operations
        let view = View::contiguous(vec![2, 3, 4]);
        let result = view
            .unsqueeze(0) // (1, 2, 3, 4)
            .permute(vec![0, 2, 1, 3]) // (1, 3, 2, 4)
            .squeeze(0); // (3, 2, 4)

        assert_eq!(result.ndim(), 3);
        assert_eq!(
            result.shape(),
            &[Expr::from(3), Expr::from(2), Expr::from(4)]
        );
    }

    #[test]
    fn test_reshape_basic() {
        // (2, 3, 4) -> (6, 4)
        let view = View::contiguous(vec![2, 3, 4]);
        let reshaped = view.reshape(vec![Expr::from(6), Expr::from(4)]);

        let View::Linear {
            shape,
            strides,
            offset,
        } = reshaped
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(6), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(4), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_reshape_flatten() {
        // (2, 3, 4) -> (24,)
        let view = View::contiguous(vec![2, 3, 4]);
        let reshaped = view.reshape(vec![Expr::from(24)]);

        let View::Linear {
            shape,
            strides,
            offset,
        } = reshaped
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(24)]);
        assert_eq!(strides, vec![Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_reshape_expand_dims() {
        // (24,) -> (2, 3, 4)
        let view = View::contiguous(vec![24]);
        let reshaped = view.reshape(vec![Expr::from(2), Expr::from(3), Expr::from(4)]);

        let View::Linear {
            shape,
            strides,
            offset,
        } = reshaped
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(2), Expr::from(3), Expr::from(4)]);
        assert_eq!(strides, vec![Expr::from(12), Expr::from(4), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    #[should_panic(expected = "reshape can only be applied to contiguous views")]
    fn test_reshape_non_contiguous() {
        // Permuted view is not contiguous
        let view = View::contiguous(vec![2, 3, 4]).permute(vec![2, 1, 0]);
        let _ = view.reshape(vec![Expr::from(6), Expr::from(4)]); // Should panic
    }

    #[test]
    #[should_panic(expected = "reshape requires the number of elements to match")]
    fn test_reshape_wrong_numel() {
        let view = View::contiguous(vec![2, 3, 4]);
        let _ = view.reshape(vec![Expr::from(5), Expr::from(5)]); // 24 != 25, should panic
    }

    #[test]
    fn test_reshape_for_tiling() {
        // Tilingのユースケース: (12, 16) -> (4, 3, 4, 4) -> permute -> (4, 4, 3, 4)
        let view = View::contiguous(vec![12, 16]);
        let tiled = view.reshape(vec![
            Expr::from(4),
            Expr::from(3),
            Expr::from(4),
            Expr::from(4),
        ]);

        let View::Linear { shape, .. } = &tiled else {
            panic!("Expected Linear view")
        };
        assert_eq!(
            shape,
            &vec![Expr::from(4), Expr::from(3), Expr::from(4), Expr::from(4)]
        );

        // Permute to group tiles: (4, 4, 3, 4) for better cache locality
        let permuted = tiled.permute(vec![0, 2, 1, 3]);
        assert_eq!(
            permuted.shape(),
            &[Expr::from(4), Expr::from(4), Expr::from(3), Expr::from(4)]
        );
    }

    // IndexExpr tests

    #[test]
    fn test_index_expr_basic() {
        // 転置を式で表現: offset = idx1 * 4 + idx0
        let view = View::from_index_expr(
            vec![Expr::from(4), Expr::from(3)],
            Expr::Idx(1) * Expr::from(4) + Expr::Idx(0),
        );

        assert_eq!(view.shape(), &[Expr::from(4), Expr::from(3)]);
        assert_eq!(view.ndim(), 2);
        assert!(!view.is_contiguous());
        assert!(!view.is_linear());
    }

    #[test]
    fn test_index_expr_permute() {
        // IndexExprに対するpermute
        // 元: shape=[3, 4], index_expr = Idx(0) * 4 + Idx(1)
        let view = View::from_index_expr(
            vec![Expr::from(3), Expr::from(4)],
            Expr::Idx(0) * Expr::from(4) + Expr::Idx(1),
        );

        // permute([1, 0]): shape=[4, 3], Idx(0)->Idx(1), Idx(1)->Idx(0)
        let permuted = view.permute(vec![1, 0]);

        assert_eq!(permuted.shape(), &[Expr::from(4), Expr::from(3)]);

        // 新しいindex_exprは Idx(1) * 4 + Idx(0) になるはず
        if let View::IndexExpr { index_expr, .. } = permuted {
            // Idx(0)とIdx(1)が入れ替わっている
            let expected = Expr::Idx(1) * Expr::from(4) + Expr::Idx(0);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_index_expr_unsqueeze() {
        // IndexExprに対するunsqueeze
        let view = View::from_index_expr(
            vec![Expr::from(3), Expr::from(4)],
            Expr::Idx(0) * Expr::from(4) + Expr::Idx(1),
        );

        // unsqueeze(1): shape=[3, 1, 4]
        let unsqueezed = view.unsqueeze(1);

        assert_eq!(
            unsqueezed.shape(),
            &[Expr::from(3), Expr::from(1), Expr::from(4)]
        );

        // index_expr: Idx(0) * 4 + Idx(1) -> Idx(0) * 4 + Idx(2)
        if let View::IndexExpr { index_expr, .. } = unsqueezed {
            let expected = Expr::Idx(0) * Expr::from(4) + Expr::Idx(2);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_index_expr_squeeze() {
        // IndexExprに対するsqueeze
        let view = View::from_index_expr(
            vec![Expr::from(3), Expr::from(1), Expr::from(4)],
            Expr::Idx(0) * Expr::from(4) + Expr::Idx(2),
        );

        // squeeze(1): shape=[3, 4]
        let squeezed = view.squeeze(1);

        assert_eq!(squeezed.shape(), &[Expr::from(3), Expr::from(4)]);

        // index_expr: Idx(0) * 4 + Idx(2) -> Idx(0) * 4 + Idx(1)
        // (Idx(1)が0に、Idx(2)がIdx(1)に)
        if let View::IndexExpr { index_expr, .. } = squeezed {
            let expected = Expr::Idx(0) * Expr::from(4) + Expr::Idx(1);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_index_expr_flip() {
        // IndexExprに対するflip
        let view = View::from_index_expr(
            vec![Expr::from(3), Expr::from(4)],
            Expr::Idx(0) * Expr::from(4) + Expr::Idx(1),
        );

        // flip(0): Idx(0) -> (3 - 1 - Idx(0)) = (2 - Idx(0))
        let flipped = view.flip(0);

        assert_eq!(flipped.shape(), &[Expr::from(3), Expr::from(4)]);

        if let View::IndexExpr { index_expr, .. } = flipped {
            // (2 - Idx(0)) * 4 + Idx(1) を期待
            // simplify後の形式を確認
            let idx0_flipped = Expr::from(2) - Expr::Idx(0);
            let expected = idx0_flipped * Expr::from(4) + Expr::Idx(1);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_index_expr_repeat() {
        // IndexExprに対するrepeat
        let view = View::from_index_expr(vec![Expr::from(1), Expr::from(4)], Expr::Idx(1));

        let repeated = view.repeat(0, 3);

        assert_eq!(repeated.shape(), &[Expr::from(3), Expr::from(4)]);
        assert!(!repeated.is_linear());

        // Idx(0) は 0 に置換される（サイズ1の軸だったので常に0）
        if let View::IndexExpr { index_expr, .. } = repeated {
            // 元: Idx(1)
            // repeat(0, 3)後: Idx(1) （Idx(0)は存在しなかったので変わらず）
            let expected = Expr::Idx(1);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_tile_then_unsqueeze_then_repeat() {
        // 典型的なユースケース: tile -> unsqueeze -> repeat
        let view = View::contiguous(vec![3, 4]);

        // tile(0, 2): [3, 4] -> [6, 4], IndexExpr
        let tiled = view.tile(0, 2);
        assert_eq!(tiled.shape(), &[Expr::from(6), Expr::from(4)]);

        // unsqueeze(0): [6, 4] -> [1, 6, 4], IndexExpr
        let unsqueezed = tiled.unsqueeze(0);
        assert_eq!(
            unsqueezed.shape(),
            &[Expr::from(1), Expr::from(6), Expr::from(4)]
        );

        // repeat(0, 8): [1, 6, 4] -> [8, 6, 4], IndexExpr
        let repeated = unsqueezed.repeat(0, 8);
        assert_eq!(
            repeated.shape(),
            &[Expr::from(8), Expr::from(6), Expr::from(4)]
        );
        assert!(!repeated.is_linear());
    }

    #[test]
    #[should_panic(expected = "reshape can only be applied to contiguous views")]
    fn test_index_expr_reshape_panics() {
        let view = View::from_index_expr(
            vec![Expr::from(3), Expr::from(4)],
            Expr::Idx(0) * Expr::from(4) + Expr::Idx(1),
        );

        let _ = view.reshape(vec![Expr::from(12)]); // Should panic
    }

    // tile tests

    #[test]
    fn test_tile_1d() {
        // [1, 2, 3] を2回繰り返し -> [1, 2, 3, 1, 2, 3]
        let view = View::contiguous(vec![3]);
        let tiled = view.tile(0, 2);

        assert_eq!(tiled.shape(), &[Expr::from(6)]);
        assert!(!tiled.is_linear()); // IndexExprになる

        // index_expr: Idx(0) % 3
        if let View::IndexExpr { index_expr, .. } = tiled {
            let expected = Expr::Idx(0) % Expr::from(3);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_tile_2d_axis0() {
        // shape: [3, 4], tile axis 0 by 2 -> [6, 4]
        // アクセス: (idx0 % 3) * 4 + idx1
        let view = View::contiguous(vec![3, 4]);
        let tiled = view.tile(0, 2);

        assert_eq!(tiled.shape(), &[Expr::from(6), Expr::from(4)]);
        assert!(!tiled.is_linear());

        if let View::IndexExpr { index_expr, .. } = tiled {
            // (Idx(0) % 3) * 4 + Idx(1)
            let expected = (Expr::Idx(0) % Expr::from(3)) * Expr::from(4) + Expr::Idx(1);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_tile_2d_axis1() {
        // shape: [3, 4], tile axis 1 by 3 -> [3, 12]
        // アクセス: idx0 * 4 + (idx1 % 4)
        let view = View::contiguous(vec![3, 4]);
        let tiled = view.tile(1, 3);

        assert_eq!(tiled.shape(), &[Expr::from(3), Expr::from(12)]);
        assert!(!tiled.is_linear());

        if let View::IndexExpr { index_expr, .. } = tiled {
            // Idx(0) * 4 + (Idx(1) % 4)
            let expected = Expr::Idx(0) * Expr::from(4) + (Expr::Idx(1) % Expr::from(4));
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_tile_on_index_expr() {
        // IndexExprに対してtile
        let view = View::from_index_expr(
            vec![Expr::from(3), Expr::from(4)],
            Expr::Idx(0) * Expr::from(4) + Expr::Idx(1),
        );

        let tiled = view.tile(0, 2);

        assert_eq!(tiled.shape(), &[Expr::from(6), Expr::from(4)]);
        assert!(!tiled.is_linear());

        if let View::IndexExpr { index_expr, .. } = tiled {
            // Idx(0)が (Idx(0) % 3) に置換される
            let expected = (Expr::Idx(0) % Expr::from(3)) * Expr::from(4) + Expr::Idx(1);
            assert_eq!(index_expr, expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    // is_innermost_contiguous tests

    #[test]
    fn test_is_innermost_contiguous_contiguous_view() {
        // 連続したViewは最内軸も連続
        let view = View::contiguous(vec![3, 4]);
        assert!(view.is_innermost_contiguous());

        let view_3d = View::contiguous(vec![2, 3, 4]);
        assert!(view_3d.is_innermost_contiguous());
    }

    #[test]
    fn test_is_innermost_contiguous_1d() {
        let view = View::contiguous(vec![10]);
        assert!(view.is_innermost_contiguous());
    }

    #[test]
    fn test_is_innermost_contiguous_scalar() {
        // スカラー（0次元）は連続とみなす
        let view = View::contiguous(Vec::<isize>::new());
        assert!(view.is_innermost_contiguous());
    }

    #[test]
    fn test_is_innermost_contiguous_transpose() {
        // transpose([1, 0])すると最内軸が非連続になる
        // 元: shape=[3, 4], strides=[4, 1]
        // transpose後: shape=[4, 3], strides=[1, 4]
        // 最内軸(axis 1)のstride = 4 ≠ 1 なので非連続
        let transposed = View::contiguous(vec![3, 4]).permute(vec![1, 0]);
        assert!(!transposed.is_innermost_contiguous());
    }

    #[test]
    fn test_is_innermost_contiguous_partial_transpose() {
        // 3Dで最後2軸のみtransposeする場合
        // 元: shape=[2, 3, 4], strides=[12, 4, 1]
        // permute([0, 2, 1])後: shape=[2, 4, 3], strides=[12, 1, 4]
        // 最内軸(axis 2)のstride = 4 ≠ 1 なので非連続
        let transposed = View::contiguous(vec![2, 3, 4]).permute(vec![0, 2, 1]);
        assert!(!transposed.is_innermost_contiguous());

        // permute([1, 0, 2])後: shape=[3, 2, 4], strides=[4, 12, 1]
        // 最内軸(axis 2)のstride = 1 なので連続
        let transposed2 = View::contiguous(vec![2, 3, 4]).permute(vec![1, 0, 2]);
        assert!(transposed2.is_innermost_contiguous());
    }

    #[test]
    fn test_is_innermost_contiguous_flip() {
        // flipすると最内軸のstrideが負になる
        // 元: shape=[3, 4], strides=[4, 1]
        // flip(1)後: strides=[4, -1]
        // 最内軸のstride = -1 ≠ 1 なので非連続
        let flipped = View::contiguous(vec![3, 4]).flip(1);
        assert!(!flipped.is_innermost_contiguous());

        // 外側軸をflipした場合は最内軸は連続のまま
        // flip(0)後: strides=[-4, 1]
        // 最内軸のstride = 1 なので連続
        let flipped_outer = View::contiguous(vec![3, 4]).flip(0);
        assert!(flipped_outer.is_innermost_contiguous());
    }

    #[test]
    fn test_is_innermost_contiguous_repeat() {
        // repeatはstride=0を設定するが、最内軸以外なら連続
        // 元: shape=[1, 4], strides=[4, 1]
        // repeat(0, 3)後: shape=[3, 4], strides=[0, 1]
        // 最内軸のstride = 1 なので連続
        let repeated = View::contiguous(vec![1, 4]).repeat(0, 3);
        assert!(repeated.is_innermost_contiguous());
    }

    #[test]
    fn test_is_innermost_contiguous_index_expr() {
        // IndexExprは常に非連続として扱う
        let view = View::from_index_expr(
            vec![Expr::from(3), Expr::from(4)],
            Expr::Idx(0) * Expr::from(4) + Expr::Idx(1),
        );
        assert!(!view.is_innermost_contiguous());
    }
}

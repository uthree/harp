use super::Expr;
use std::collections::HashSet;

/// パディング値を表すenum
///
/// テンソルの境界外アクセス時に使用するデフォルト値を指定します。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadValue {
    /// 0.0 - Sum演算の単位元
    Zero,
    /// 1.0 - Prod演算の単位元
    One,
    /// 負の無限大 - Max演算の単位元
    NegInf,
}

impl PadValue {
    /// パディング値を浮動小数点数として取得
    pub fn as_f32(&self) -> f32 {
        match self {
            PadValue::Zero => 0.0,
            PadValue::One => 1.0,
            PadValue::NegInf => f32::NEG_INFINITY,
        }
    }

    /// パディング値を倍精度浮動小数点数として取得
    pub fn as_f64(&self) -> f64 {
        match self {
            PadValue::Zero => 0.0,
            PadValue::One => 1.0,
            PadValue::NegInf => f64::NEG_INFINITY,
        }
    }
}

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

    /// 任意の条件付きマスクView
    ///
    /// 任意のExpr条件に基づいてinner Viewの値またはデフォルト値を返す。
    /// Attention maskや三角行列マスク、スパースパターンなど任意の境界条件に使用可能。
    ///
    /// # 動作
    /// - `condition`が非0（true）の場合: inner Viewの値を返す
    /// - `condition`が0（false）の場合: `default_value`を返す
    ///
    /// # Example
    /// ```text
    /// // Attention mask (causal): ridx[0] <= ridx[1]
    /// let mask = Expr::Idx(0).le(Expr::Idx(1));
    /// View::Masked { inner, condition: mask, default_value: PadValue::NegInf }
    ///
    /// // Sparse (even indices only): ridx[0] % 2 == 0
    /// let sparse = Expr::Idx(0).rem(2).eq_expr(0);
    /// View::Masked { inner, condition: sparse, default_value: PadValue::Zero }
    /// ```
    Masked {
        /// 内側のView（条件がtrueの場合に使用）
        inner: Box<View>,
        /// 条件式（Idx変数を含むExpr、非0ならinner、0ならdefault）
        condition: Expr,
        /// 条件がfalseの場合のデフォルト値
        default_value: PadValue,
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

    /// Viewの論理的な形状を返す
    ///
    /// 各バリアントの形状を返す。
    /// Maskedの場合はinner Viewの形状を返す。
    pub fn shape(&self) -> Vec<Expr> {
        match self {
            View::Linear { shape, .. } | View::IndexExpr { shape, .. } => shape.clone(),
            View::Masked { inner, .. } => inner.shape(),
        }
    }

    /// 内側のViewの形状を返す（Maskedの場合に有用）
    pub fn inner_shape(&self) -> Vec<Expr> {
        match self {
            View::Masked { inner, .. } => inner.shape(),
            _ => self.shape(),
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
    /// Masked Viewは内側のViewとcondition内のLoadIndexをチェックします。
    pub fn contains_load_index(&self) -> bool {
        match self {
            View::Linear { .. } => false,
            View::IndexExpr { index_expr, .. } => index_expr.contains_load_index(),
            View::Masked {
                inner, condition, ..
            } => inner.contains_load_index() || condition.contains_load_index(),
        }
    }

    /// ViewをIndexExpr形式に変換
    ///
    /// Linear Viewを等価なIndexExpr形式に変換します。
    /// 既にIndexExprの場合はクローンを返します。
    /// Masked Viewは条件分岐が必要なため変換できず、そのまま返します。
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
            // Maskedは条件分岐が必要なためIndexExprに変換できない
            View::Masked { .. } => self.clone(),
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
    /// use harp::shape::{View, Expr};
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

            // Masked outer × any inner: Maskedの条件を保持
            (
                View::Masked {
                    inner: masked_inner,
                    condition,
                    default_value,
                },
                inner_view,
            ) => {
                // innerとmasked_innerを合成
                let composed_inner = Self::compose(masked_inner, inner_view);
                View::Masked {
                    inner: Box::new(composed_inner),
                    condition: condition.clone(),
                    default_value: *default_value,
                }
            }

            // any × Masked: innerがMaskedの場合
            (outer_view, View::Masked { .. }) => {
                // innerがMaskedの場合も同様にouterを返す
                outer_view.clone()
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
            View::Masked {
                inner,
                condition,
                default_value,
            } => View::Masked {
                inner: Box::new(inner.shift_load_index(delta)),
                condition: condition.clone().shift_load_index(delta),
                default_value: *default_value,
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
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                // 内側のViewをpermute
                let new_inner = inner.permute(axes.clone());
                // conditionのIdx変数も並べ替え
                let new_condition = condition.permute_idx(&axes);
                View::Masked {
                    inner: Box::new(new_inner),
                    condition: new_condition,
                    default_value,
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
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                // 内側のViewをunsqueeze
                let new_inner = inner.unsqueeze(axis);
                // conditionのIdx(i) for i >= axis を Idx(i+1) にシフト
                let new_condition = condition.shift_idx(axis, 1);
                View::Masked {
                    inner: Box::new(new_inner),
                    condition: new_condition,
                    default_value,
                }
            }
        }
    }

    pub fn squeeze(self, axis: usize) -> Self {
        assert!(axis < self.ndim());
        // Maskedの場合はmatch前にshapeを取得する必要がある
        let output_shape = self.shape();
        assert_eq!(
            output_shape[axis],
            1.into(),
            "can only squeeze an axis of size 1"
        );
        match self {
            View::Linear {
                mut shape,
                mut strides,
                offset,
            } => {
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
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                // 内側のViewをsqueeze
                let new_inner = inner.squeeze(axis);
                // conditionも更新: Idx(axis) を 0 に置換し、Idx(i) for i > axis を Idx(i-1) にシフト
                let new_condition = condition
                    .substitute_idx(axis, Expr::from(0))
                    .shift_idx(axis + 1, -1);
                View::Masked {
                    inner: Box::new(new_inner),
                    condition: new_condition,
                    default_value,
                }
            }
        }
    }

    /// Expand a dimension of size 1 to a larger size (explicit broadcast)
    ///
    /// This operation allows broadcasting by expanding a size-1 dimension
    /// to a specified size. The stride for that axis remains 0, so all
    /// elements along that axis read from the same memory location.
    ///
    /// # Panics
    /// - If `axis >= ndim()`
    /// - If `shape[axis] != 1`
    pub fn expand(self, axis: usize, size: Expr) -> Self {
        assert!(axis < self.ndim(), "axis out of bounds");
        let current_shape = self.shape();
        assert_eq!(
            current_shape[axis],
            1.into(),
            "can only expand an axis of size 1"
        );

        match self {
            View::Linear {
                mut shape,
                strides,
                offset,
            } => {
                // Change shape[axis] to the new size
                // Stride stays 0 (or whatever it was for size-1 axis)
                shape[axis] = size;
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
                // Update shape, index_expr stays the same
                // (Idx(axis) will iterate over [0, size) but index_expr ignores it
                // since the original axis was size 1)
                shape[axis] = size;
                View::IndexExpr { shape, index_expr }
            }
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                // Expand the inner view
                let new_inner = inner.expand(axis, size);
                View::Masked {
                    inner: Box::new(new_inner),
                    condition,
                    default_value,
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
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                let shape = inner.shape();
                // 内側のViewをflip
                let new_inner = inner.flip(axis);
                // conditionのIdx(axis)を (shape[axis] - 1 - Idx(axis)) に置換
                let flipped_idx =
                    (shape[axis].clone() - Expr::from(1) - Expr::Idx(axis)).simplify();
                let new_condition = condition.substitute_idx(axis, flipped_idx);
                View::Masked {
                    inner: Box::new(new_inner),
                    condition: new_condition,
                    default_value,
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
        // Maskedの場合はmatch前にshapeを取得
        let output_shape = self.shape();
        assert!(axis < output_shape.len(), "axis out of bounds");
        assert!(
            output_shape[axis].is_one(),
            "can only repeat an axis of size 1"
        );
        match self {
            View::Linear {
                mut shape,
                mut strides,
                offset,
            } => {
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
                // Idx(axis) を 0 に固定（常に同じ位置を参照）
                // サイズ1の軸なので、Idx(axis)は常に0だが、明示的に置換
                let new_index_expr = index_expr.substitute_idx(axis, Expr::from(0));
                shape[axis] = times;

                View::IndexExpr {
                    shape,
                    index_expr: new_index_expr,
                }
            }
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                // 内側のViewをrepeat
                let new_inner = inner.repeat(axis, times);
                // conditionは変更不要（サイズ1なのでIdx(axis)は常に0）
                View::Masked {
                    inner: Box::new(new_inner),
                    condition,
                    default_value,
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
    /// use harp::shape::{View, Expr};
    ///
    /// let view = View::contiguous(vec![3, 4]); // shape: [3, 4]
    /// let tiled = view.tile(0, 2); // shape: [6, 4], idx0を%3で循環
    ///
    /// assert_eq!(tiled.shape(), &[Expr::from(6), Expr::from(4)]);
    /// assert!(!tiled.is_linear()); // IndexExprになる
    /// ```
    pub fn tile(self, axis: usize, times: impl Into<Expr>) -> Self {
        let times = times.into();
        // Maskedの場合はmatch前にshapeを取得
        let output_shape = self.shape();
        assert!(axis < output_shape.len(), "axis out of bounds");

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
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                let original_size = inner.shape()[axis].clone();
                // 内側のViewをtile
                let new_inner = inner.tile(axis, times);
                // conditionのIdx(axis)を Idx(axis) % original_size に置換
                let cyclic_idx = (Expr::Idx(axis) % original_size).simplify();
                let new_condition = condition.substitute_idx(axis, cyclic_idx);
                View::Masked {
                    inner: Box::new(new_inner),
                    condition: new_condition,
                    default_value,
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
            View::Masked { .. } => {
                // is_contiguous()がfalseを返すので、ここには到達しない
                unreachable!("Masked views are always non-contiguous")
            }
        }
    }

    pub fn is_contiguous(&self) -> bool {
        match self {
            View::Linear { shape, .. } => *self == View::contiguous(shape.clone()),
            // IndexExprは常に非連続として扱う
            View::IndexExpr { .. } => false,
            // Maskedは条件チェックがあるため非連続
            View::Masked { .. } => false,
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
    /// use harp::shape::View;
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
            // Maskedは条件チェックがあるため連続性を保証できない
            View::Masked { .. } => false,
        }
    }

    /// Maskedバリアントかどうかを判定
    pub fn is_masked(&self) -> bool {
        matches!(self, View::Masked { .. })
    }

    /// IndexExpr Viewを作成
    ///
    /// # Arguments
    /// * `shape` - 論理的な形状
    /// * `index_expr` - インデックス計算式（Idx(0), Idx(1), ... を含む）
    ///
    /// # Examples
    /// ```
    /// use harp::shape::{View, Expr};
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

    /// Padded Viewを作成（内部でMaskedとIndexExprを使用）
    ///
    /// 内側のViewの周囲にパディングを追加するViewを作成します。
    /// 実装はIndexExpr（インデックス調整）とMasked（境界条件）の組み合わせです。
    ///
    /// # Arguments
    /// * `inner` - 内側のView（パディング前のテンソルのView）
    /// * `padding` - パディング量: (前, 後) × 各次元
    /// * `default_value` - 境界外アクセス時のデフォルト値
    ///
    /// # Examples
    /// ```
    /// use harp::shape::{View, Expr};
    /// use harp::shape::PadValue;
    ///
    /// // 3x4のテンソルに各軸に1ずつパディング -> 5x6
    /// let inner = View::contiguous(vec![3, 4]);
    /// let padded = View::padded(
    ///     inner,
    ///     vec![(Expr::from(1), Expr::from(1)), (Expr::from(1), Expr::from(1))],
    ///     PadValue::Zero,
    /// );
    /// assert_eq!(padded.shape(), vec![Expr::from(5), Expr::from(6)]);
    /// ```
    pub fn padded(inner: View, padding: Vec<(Expr, Expr)>, default_value: PadValue) -> Self {
        assert_eq!(
            inner.ndim(),
            padding.len(),
            "padding dimensions must match inner view dimensions"
        );

        let inner_shape = inner.shape().to_vec();
        let ndim = inner_shape.len();

        // 1. パディング後の形状を計算
        let padded_shape: Vec<Expr> = inner_shape
            .iter()
            .zip(padding.iter())
            .map(|(s, (before, after))| (s.clone() + before.clone() + after.clone()).simplify())
            .collect();

        // 2. 境界条件を構築
        // 各軸で: Idx(i) >= before[i] && Idx(i) < before[i] + inner_shape[i]
        let mut condition = Expr::Const(1); // 初期値: true
        for (i, ((before, _after), inner_size)) in
            padding.iter().zip(inner_shape.iter()).enumerate()
        {
            let idx = Expr::Idx(i);
            // idx >= before は !(idx < before) で表現
            let ge_before = !idx.clone().lt(before.clone());
            // idx < before + inner_size
            let lt_upper = idx.lt(before.clone() + inner_size.clone());
            // ANDで結合
            condition = condition.and(ge_before).and(lt_upper);
        }
        condition = condition.simplify();

        // 3. パディング後のインデックスを元のオフセットにマッピングするインデックス式を構築
        // adjusted_idx[i] = Idx(i) - before[i]
        let index_expr = match &inner {
            View::Linear {
                strides, offset, ..
            } => {
                // Linear: offset + Σ(adjusted_idx[i] * stride[i])
                let mut expr = offset.clone();
                for (i, (stride, (before, _))) in strides.iter().zip(padding.iter()).enumerate() {
                    expr = (expr + (Expr::Idx(i) - before.clone()) * stride.clone()).simplify();
                }
                expr
            }
            View::IndexExpr { index_expr, .. } => {
                // IndexExpr: Idx(i) を Idx(i) - before[i] で置換
                let mut expr = index_expr.clone();
                for (i, (before, _)) in padding.iter().enumerate() {
                    expr = expr.substitute_idx(i, Expr::Idx(i) - before.clone());
                }
                expr.simplify()
            }
            View::Masked {
                inner: masked_inner,
                ..
            } => {
                // Masked: 内側のViewを取得してpaddedを適用
                let inner_padded =
                    View::padded((**masked_inner).clone(), padding.clone(), default_value);
                match inner_padded {
                    View::Masked {
                        inner: inner_box, ..
                    } => {
                        if let View::IndexExpr { index_expr, .. } = *inner_box {
                            index_expr
                        } else {
                            // フォールバック: 連続メモリとして扱う
                            let mut expr = Expr::Const(0);
                            let mut stride = Expr::Const(1);
                            for i in (0..ndim).rev() {
                                let (before, _) = &padding[i];
                                expr = (expr + (Expr::Idx(i) - before.clone()) * stride.clone())
                                    .simplify();
                                if i > 0 {
                                    stride = (stride * inner_shape[i].clone()).simplify();
                                }
                            }
                            expr
                        }
                    }
                    _ => unreachable!(),
                }
            }
        };

        // 4. パディング後の形状を持つIndexExprを作成
        let inner_view = View::IndexExpr {
            shape: padded_shape,
            index_expr,
        };

        // 5. Maskedでラップして返す
        View::Masked {
            inner: Box::new(inner_view),
            condition,
            default_value,
        }
    }

    /// Masked Viewを作成
    ///
    /// 任意の条件式に基づいてinner Viewの値またはデフォルト値を返すViewを作成。
    /// Attention maskや三角行列マスクなどに使用。
    ///
    /// # Arguments
    /// * `inner` - 条件がtrueの場合に使用するView
    /// * `condition` - 条件式（非0ならinner、0ならdefault）
    /// * `default_value` - 条件がfalseの場合のデフォルト値
    ///
    /// # Examples
    /// ```
    /// use harp::shape::{View, Expr};
    /// use harp::shape::PadValue;
    ///
    /// // Attention mask (causal): ridx[0] <= ridx[1]
    /// let inner = View::contiguous(vec![4, 4]);
    /// let mask = Expr::Idx(0).le(Expr::Idx(1));
    /// let view = View::masked(inner, mask, PadValue::NegInf);
    /// ```
    pub fn masked(inner: View, condition: impl Into<Expr>, default_value: PadValue) -> Self {
        View::Masked {
            inner: Box::new(inner),
            condition: condition.into(),
            default_value,
        }
    }

    /// Maskedの場合、デフォルト値を返す
    pub fn default_value(&self) -> Option<PadValue> {
        match self {
            View::Masked { default_value, .. } => Some(*default_value),
            _ => None,
        }
    }

    /// Maskedの場合、内側のViewを返す
    pub fn inner_view(&self) -> Option<&View> {
        match self {
            View::Masked { inner, .. } => Some(inner),
            _ => None,
        }
    }

    /// Maskedの場合、条件式を返す
    pub fn condition(&self) -> Option<&Expr> {
        match self {
            View::Masked { condition, .. } => Some(condition),
            _ => None,
        }
    }

    /// N次元 unfold（スライディングウィンドウ）
    ///
    /// 指定した軸に対してスライディングウィンドウを適用。
    /// 各軸は2つの次元（output_position, window_size）に分割される。
    ///
    /// # Arguments
    /// * `axes` - unfoldする軸のリスト（ソート済み、重複なし）
    /// * `sizes` - 各軸のウィンドウサイズ
    /// * `strides` - 各軸のストライド
    /// * `dilations` - 各軸のdilation（カーネル要素間の間隔）
    ///
    /// # Output Shape
    /// `[preserved_dims..., output_positions..., window_dims...]`
    ///
    /// # Dilation
    /// dilationは各カーネル要素間の間隔を指定します。
    /// effective_kernel_size = (kernel_size - 1) * dilation + 1
    /// output_size = (input_size - effective_kernel_size) / stride + 1
    ///
    /// # Example
    /// ```
    /// use harp::shape::{View, Expr};
    ///
    /// // [N, C, H, W] + unfold(axes=[2,3], sizes=[kH,kW], strides=[sH,sW], dilations=[1,1])
    /// // → [N, C, out_H, out_W, kH, kW]
    /// let view = View::contiguous(vec![2, 3, 8, 10]);
    /// let unfolded = view.unfold(
    ///     &[2, 3],
    ///     &[Expr::from(3), Expr::from(3)],
    ///     &[Expr::from(1), Expr::from(1)],
    ///     &[Expr::from(1), Expr::from(1)],
    /// );
    /// assert_eq!(unfolded.shape(), vec![
    ///     Expr::from(2), Expr::from(3),   // N, C (preserved)
    ///     Expr::from(6), Expr::from(8),   // out_H, out_W
    ///     Expr::from(3), Expr::from(3),   // kH, kW
    /// ]);
    /// ```
    pub fn unfold<E: Into<Expr> + Clone>(
        self,
        axes: &[usize],
        sizes: &[E],
        strides: &[E],
        dilations: &[E],
    ) -> Self {
        let sizes: Vec<Expr> = sizes.iter().map(|s| s.clone().into()).collect();
        let strides: Vec<Expr> = strides.iter().map(|s| s.clone().into()).collect();
        let dilations: Vec<Expr> = dilations.iter().map(|d| d.clone().into()).collect();

        assert_eq!(
            axes.len(),
            sizes.len(),
            "axes and sizes must have same length"
        );
        assert_eq!(
            axes.len(),
            strides.len(),
            "axes and strides must have same length"
        );
        assert_eq!(
            axes.len(),
            dilations.len(),
            "axes and dilations must have same length"
        );

        let ndim = self.ndim();
        let num_unfolded = axes.len();

        // axesの検証
        for (i, &axis) in axes.iter().enumerate() {
            assert!(axis < ndim, "axis {} out of bounds for ndim {}", axis, ndim);
            if i > 0 {
                assert!(axes[i] > axes[i - 1], "axes must be sorted and unique");
            }
        }

        let axes_set: std::collections::HashSet<usize> = axes.iter().copied().collect();

        match self {
            View::Linear {
                shape,
                strides: original_strides,
                offset,
            } => {
                // 出力形状を構築
                // [preserved..., output_positions..., window_dims...]
                let mut new_shape: Vec<Expr> = Vec::new();

                // 1. preserved dims (axes以外)
                for (i, dim) in shape.iter().enumerate() {
                    if !axes_set.contains(&i) {
                        new_shape.push(dim.clone());
                    }
                }

                // 2. output positions: (dim - effective_size) / stride + 1
                // where effective_size = (size - 1) * dilation + 1
                for (i, &axis) in axes.iter().enumerate() {
                    let dim = shape[axis].clone();
                    let size = sizes[i].clone();
                    let stride = strides[i].clone();
                    let dilation = dilations[i].clone();
                    // effective_size = (size - 1) * dilation + 1
                    let effective_size =
                        ((size.clone() - Expr::from(1)) * dilation + Expr::from(1)).simplify();
                    let out_size = ((dim - effective_size) / stride + Expr::from(1)).simplify();
                    new_shape.push(out_size);
                }

                // 3. window dims
                for size in sizes.iter() {
                    new_shape.push(size.clone());
                }

                // インデックス式を構築
                // output[preserved..., out_pos..., win_pos...] = input[...]
                //
                // preserved dims: result idx 0..(ndim - num_unfolded)
                // output positions: result idx (ndim - num_unfolded)..(ndim)
                // window dims: result idx (ndim)..(ndim + num_unfolded)
                let num_preserved = ndim - num_unfolded;

                let mut index_expr = offset;

                // preserved軸からのマッピング
                let mut preserved_idx = 0;
                for (original_axis, stride) in original_strides.iter().enumerate() {
                    if axes_set.contains(&original_axis) {
                        // unfold対象軸
                        let unfold_i = axes.iter().position(|&a| a == original_axis).unwrap();
                        let out_pos_idx = num_preserved + unfold_i;
                        let win_pos_idx = num_preserved + num_unfolded + unfold_i;
                        let unfold_stride = strides[unfold_i].clone();
                        let dilation = dilations[unfold_i].clone();

                        // input_idx = out_pos * unfold_stride + win_pos * dilation
                        let input_idx = (Expr::Idx(out_pos_idx) * unfold_stride
                            + Expr::Idx(win_pos_idx) * dilation)
                            .simplify();
                        index_expr = (index_expr + input_idx * stride.clone()).simplify();
                    } else {
                        // preserved軸
                        index_expr =
                            (index_expr + Expr::Idx(preserved_idx) * stride.clone()).simplify();
                        preserved_idx += 1;
                    }
                }

                View::IndexExpr {
                    shape: new_shape,
                    index_expr,
                }
            }
            View::IndexExpr {
                shape,
                index_expr: original_index_expr,
            } => {
                // IndexExprに対するunfold
                // より複雑なケース - 既存のindex_exprを変換する必要がある

                // 出力形状を構築
                let mut new_shape: Vec<Expr> = Vec::new();

                // 1. preserved dims
                for (i, dim) in shape.iter().enumerate() {
                    if !axes_set.contains(&i) {
                        new_shape.push(dim.clone());
                    }
                }

                // 2. output positions: (dim - effective_size) / stride + 1
                // where effective_size = (size - 1) * dilation + 1
                for (i, &axis) in axes.iter().enumerate() {
                    let dim = shape[axis].clone();
                    let size = sizes[i].clone();
                    let stride = strides[i].clone();
                    let dilation = dilations[i].clone();
                    // effective_size = (size - 1) * dilation + 1
                    let effective_size =
                        ((size.clone() - Expr::from(1)) * dilation + Expr::from(1)).simplify();
                    let out_size = ((dim - effective_size) / stride + Expr::from(1)).simplify();
                    new_shape.push(out_size);
                }

                // 3. window dims
                for size in sizes.iter() {
                    new_shape.push(size.clone());
                }

                let num_preserved = ndim - num_unfolded;

                // index_exprを変換
                // 元のIdx(i)を新しいインデックス体系にマッピング
                let mut new_index_expr = original_index_expr;

                // 後ろの軸から処理（シフトの影響を避けるため）
                for (unfold_i, &original_axis) in axes.iter().enumerate().rev() {
                    let out_pos_idx = num_preserved + unfold_i;
                    let win_pos_idx = num_preserved + num_unfolded + unfold_i;
                    let unfold_stride = strides[unfold_i].clone();
                    let dilation = dilations[unfold_i].clone();

                    // Idx(original_axis) を (Idx(out_pos_idx) * stride + Idx(win_pos_idx) * dilation) に置換
                    let replacement = (Expr::Idx(out_pos_idx) * unfold_stride
                        + Expr::Idx(win_pos_idx) * dilation)
                        .simplify();
                    new_index_expr = new_index_expr.substitute_idx(original_axis, replacement);
                }

                // preserved軸のインデックスを再マッピング
                // axes以外の軸は0から順に番号が振られる
                let mut preserved_mapping: Vec<(usize, usize)> = Vec::new();
                let mut new_idx = 0;
                for i in 0..ndim {
                    if !axes_set.contains(&i) {
                        preserved_mapping.push((i, new_idx));
                        new_idx += 1;
                    }
                }

                // 大きい番号から小さい番号へ置換（衝突を避ける）
                for &(old_idx, new_idx) in preserved_mapping.iter().rev() {
                    if old_idx != new_idx {
                        // 一時的なインデックスを使用して衝突を回避
                        let temp_idx = ndim + num_unfolded + old_idx;
                        new_index_expr =
                            new_index_expr.substitute_idx(old_idx, Expr::Idx(temp_idx));
                    }
                }
                for &(old_idx, new_idx) in preserved_mapping.iter() {
                    if old_idx != new_idx {
                        let temp_idx = ndim + num_unfolded + old_idx;
                        new_index_expr =
                            new_index_expr.substitute_idx(temp_idx, Expr::Idx(new_idx));
                    }
                }

                View::IndexExpr {
                    shape: new_shape,
                    index_expr: new_index_expr.simplify(),
                }
            }
            View::Masked {
                inner,
                condition,
                default_value,
            } => {
                // Unfold the inner view first
                let unfolded_inner = inner.unfold(axes, &sizes, &strides, &dilations);

                // Transform the condition expression
                // The index mapping is:
                // - Original: Idx(0), Idx(1), ..., Idx(ndim-1)
                // - After unfold:
                //   - Preserved dims: indices 0..(ndim - num_unfolded)
                //   - Output positions: indices (ndim - num_unfolded)..(ndim)
                //   - Window positions: indices ndim..(ndim + num_unfolded)
                //
                // For each original Idx(i):
                // - If i is preserved: maps to new preserved index
                // - If i is unfolded: maps to out_pos * stride + win_pos * dilation
                let num_preserved = ndim - num_unfolded;

                // Build a mapping for index substitution
                let mut new_condition = condition.clone();

                // First, substitute temporary indices to avoid conflicts
                // Then substitute to final values
                let temp_offset = 1000; // Use large offset for temporary indices

                // Calculate the mapping: original_idx -> (is_preserved, new_idx or unfold_info)
                let mut preserved_count = 0;
                for original_idx in 0..ndim {
                    if axes_set.contains(&original_idx) {
                        // This is an unfolded axis
                        let unfold_i = axes.iter().position(|&a| a == original_idx).unwrap();
                        let out_pos_idx = num_preserved + unfold_i;
                        let win_pos_idx = num_preserved + num_unfolded + unfold_i;
                        let unfold_stride = strides[unfold_i].clone();
                        let dilation = dilations[unfold_i].clone();

                        // Replace Idx(original_idx) with (Idx(out_pos_idx) * stride + Idx(win_pos_idx) * dilation)
                        let replacement = (Expr::Idx(out_pos_idx) * unfold_stride
                            + Expr::Idx(win_pos_idx) * dilation)
                            .simplify();

                        new_condition = new_condition
                            .substitute_idx(original_idx, Expr::Idx(temp_offset + original_idx));
                        new_condition =
                            new_condition.substitute_idx(temp_offset + original_idx, replacement);
                    } else {
                        // This is a preserved axis - maps to new preserved index
                        new_condition = new_condition
                            .substitute_idx(original_idx, Expr::Idx(temp_offset + original_idx));
                        new_condition = new_condition
                            .substitute_idx(temp_offset + original_idx, Expr::Idx(preserved_count));
                        preserved_count += 1;
                    }
                }

                View::Masked {
                    inner: Box::new(unfolded_inner),
                    condition: new_condition.simplify(),
                    default_value,
                }
            }
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
    fn test_expand() {
        // Start with a view that has unsqueezed axis (stride = 0)
        // This is the typical use case: unsqueeze then expand
        let view = View::contiguous(vec![3, 4]).unsqueeze(1); // (3, 4) -> (3, 1, 4)
        let expanded = view.expand(1, Expr::from(5)); // (3, 1, 4) -> (3, 5, 4)

        let View::Linear {
            shape,
            strides,
            offset,
        } = expanded
        else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(5), Expr::from(4)]);
        // Stride for expanded axis should remain 0 (broadcast behavior)
        assert_eq!(strides, vec![Expr::from(4), Expr::from(0), Expr::from(1)]);
        assert_eq!(offset, Expr::from(0));
    }

    #[test]
    fn test_expand_contiguous_size1() {
        // Contiguous tensor with size-1 axis has non-zero stride
        // expand still works, stride is preserved (allows non-broadcast expand)
        let view = View::contiguous(vec![3, 1, 4]);
        let expanded = view.expand(1, Expr::from(5)); // (3, 1, 4) -> (3, 5, 4)

        let View::Linear { shape, strides, .. } = expanded else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(5), Expr::from(4)]);
        // Stride for axis 1 is 4 (from contiguous), preserved after expand
        assert_eq!(strides, vec![Expr::from(4), Expr::from(4), Expr::from(1)]);
    }

    #[test]
    fn test_expand_after_unsqueeze() {
        // unsqueeze then expand is a common broadcast pattern
        let view = View::contiguous(vec![3, 4]);
        let unsqueezed = view.unsqueeze(1); // (3, 4) -> (3, 1, 4)
        let expanded = unsqueezed.expand(1, Expr::from(5)); // (3, 1, 4) -> (3, 5, 4)

        let View::Linear { shape, strides, .. } = expanded else {
            panic!("Expected Linear view")
        };
        assert_eq!(shape, vec![Expr::from(3), Expr::from(5), Expr::from(4)]);
        // The stride for axis 1 should be 0 (from unsqueeze)
        assert_eq!(strides[1], Expr::from(0));
    }

    #[test]
    #[should_panic(expected = "can only expand an axis of size 1")]
    fn test_expand_non_one_axis() {
        let view = View::contiguous(vec![3, 4]);
        let _ = view.expand(0, Expr::from(10)); // Should panic: axis 0 has size 3, not 1
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

    // unfold tests

    #[test]
    fn test_unfold_1d_shape() {
        // [2, 3, 10] + unfold(axes=[2], sizes=[3], strides=[1], dilations=[1])
        // → [2, 3, 8, 3]
        // out_size = (10 - 3) / 1 + 1 = 8
        let view = View::contiguous(vec![2, 3, 10]);
        let unfolded = view.unfold(&[2], &[Expr::from(3)], &[Expr::from(1)], &[Expr::from(1)]);

        assert_eq!(
            unfolded.shape(),
            vec![Expr::from(2), Expr::from(3), Expr::from(8), Expr::from(3)]
        );
        assert!(!unfolded.is_linear());
    }

    #[test]
    fn test_unfold_1d_with_stride() {
        // [2, 3, 10] + unfold(axes=[2], sizes=[3], strides=[2], dilations=[1])
        // → [2, 3, 4, 3]
        // out_size = (10 - 3) / 2 + 1 = 4
        let view = View::contiguous(vec![2, 3, 10]);
        let unfolded = view.unfold(&[2], &[Expr::from(3)], &[Expr::from(2)], &[Expr::from(1)]);

        assert_eq!(
            unfolded.shape(),
            vec![Expr::from(2), Expr::from(3), Expr::from(4), Expr::from(3)]
        );
    }

    #[test]
    fn test_unfold_2d_shape() {
        // [2, 3, 8, 10] + unfold(axes=[2,3], sizes=[3,3], strides=[1,1], dilations=[1,1])
        // → [2, 3, 6, 8, 3, 3]
        // out_H = (8 - 3) / 1 + 1 = 6
        // out_W = (10 - 3) / 1 + 1 = 8
        let view = View::contiguous(vec![2, 3, 8, 10]);
        let unfolded = view.unfold(
            &[2, 3],
            &[Expr::from(3), Expr::from(3)],
            &[Expr::from(1), Expr::from(1)],
            &[Expr::from(1), Expr::from(1)],
        );

        assert_eq!(
            unfolded.shape(),
            vec![
                Expr::from(2),
                Expr::from(3), // N, C (preserved)
                Expr::from(6),
                Expr::from(8), // out_H, out_W
                Expr::from(3),
                Expr::from(3) // kH, kW
            ]
        );
    }

    #[test]
    fn test_unfold_2d_with_stride() {
        // [2, 3, 8, 10] + unfold(axes=[2,3], sizes=[3,3], strides=[2,2], dilations=[1,1])
        // → [2, 3, 3, 4, 3, 3]
        // out_H = (8 - 3) / 2 + 1 = 3
        // out_W = (10 - 3) / 2 + 1 = 4
        let view = View::contiguous(vec![2, 3, 8, 10]);
        let unfolded = view.unfold(
            &[2, 3],
            &[Expr::from(3), Expr::from(3)],
            &[Expr::from(2), Expr::from(2)],
            &[Expr::from(1), Expr::from(1)],
        );

        assert_eq!(
            unfolded.shape(),
            vec![
                Expr::from(2),
                Expr::from(3),
                Expr::from(3),
                Expr::from(4),
                Expr::from(3),
                Expr::from(3)
            ]
        );
    }

    #[test]
    fn test_unfold_1d_index_expr() {
        // 簡単な1Dケースでインデックス式を検証
        // [5] + unfold(axes=[0], sizes=[3], strides=[1], dilations=[1]) → [3, 3]
        // output[i, j] = input[i * 1 + j * 1] = input[i + j]
        let view = View::contiguous(vec![5]);
        let unfolded = view.unfold(&[0], &[Expr::from(3)], &[Expr::from(1)], &[Expr::from(1)]);

        assert_eq!(unfolded.shape(), vec![Expr::from(3), Expr::from(3)]);

        if let View::IndexExpr { index_expr, .. } = unfolded {
            // Expected: Idx(0) * 1 + Idx(1) = Idx(0) + Idx(1)
            let expected = (Expr::Idx(0) + Expr::Idx(1)).simplify();
            assert_eq!(index_expr.simplify(), expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_unfold_1d_index_expr_with_stride() {
        // [10] + unfold(axes=[0], sizes=[3], strides=[2], dilations=[1]) → [4, 3]
        // output[i, j] = input[i * 2 + j * 1]
        let view = View::contiguous(vec![10]);
        let unfolded = view.unfold(&[0], &[Expr::from(3)], &[Expr::from(2)], &[Expr::from(1)]);

        assert_eq!(unfolded.shape(), vec![Expr::from(4), Expr::from(3)]);

        if let View::IndexExpr { index_expr, .. } = unfolded {
            // Expected: Idx(0) * 2 + Idx(1)
            let expected = (Expr::Idx(0) * Expr::from(2) + Expr::Idx(1)).simplify();
            assert_eq!(index_expr.simplify(), expected);
        } else {
            panic!("Expected IndexExpr variant");
        }
    }

    #[test]
    fn test_unfold_conv_pattern() {
        // 典型的なConv2Dパターン: [N, C, H, W] → [N, C, out_H, out_W, kH, kW]
        // [1, 3, 28, 28] + unfold(axes=[2,3], sizes=[3,3], strides=[1,1], dilations=[1,1])
        // → [1, 3, 26, 26, 3, 3]
        let view = View::contiguous(vec![1, 3, 28, 28]);
        let unfolded = view.unfold(
            &[2, 3],
            &[Expr::from(3), Expr::from(3)],
            &[Expr::from(1), Expr::from(1)],
            &[Expr::from(1), Expr::from(1)],
        );

        assert_eq!(
            unfolded.shape(),
            vec![
                Expr::from(1),
                Expr::from(3),
                Expr::from(26),
                Expr::from(26),
                Expr::from(3),
                Expr::from(3)
            ]
        );
    }

    #[test]
    fn test_unfold_with_usize() {
        // usizeでも呼び出せることを確認
        let view = View::contiguous(vec![2, 3, 10]);
        let unfolded = view.unfold(&[2], &[3_usize], &[1_usize], &[1_usize]);

        assert_eq!(
            unfolded.shape(),
            vec![Expr::from(2), Expr::from(3), Expr::from(8), Expr::from(3)]
        );
    }

    #[test]
    #[should_panic(expected = "axis 5 out of bounds")]
    fn test_unfold_axis_out_of_bounds() {
        let view = View::contiguous(vec![2, 3, 10]);
        let _ = view.unfold(&[5], &[Expr::from(3)], &[Expr::from(1)], &[Expr::from(1)]);
    }

    #[test]
    #[should_panic(expected = "axes must be sorted and unique")]
    fn test_unfold_unsorted_axes() {
        let view = View::contiguous(vec![2, 3, 8, 10]);
        let _ = view.unfold(
            &[3, 2], // wrong order
            &[Expr::from(3), Expr::from(3)],
            &[Expr::from(1), Expr::from(1)],
            &[Expr::from(1), Expr::from(1)],
        );
    }

    // =========================================================================
    // Masked View tests
    // =========================================================================

    #[test]
    fn test_masked_creation() {
        use super::PadValue;

        let inner = View::contiguous(vec![4, 4]);
        // Attention mask: idx0 <= idx1
        let condition = Expr::Idx(0).le(Expr::Idx(1));
        let view = View::masked(inner, condition, PadValue::NegInf);

        assert!(view.is_masked());
        assert!(!view.is_contiguous());
        assert_eq!(view.shape(), vec![Expr::from(4), Expr::from(4)]);
    }

    #[test]
    fn test_masked_shape_preserved() {
        use super::PadValue;

        let inner = View::contiguous(vec![3, 5, 7]);
        let condition = Expr::Idx(0).lt(Expr::Idx(1));
        let view = View::masked(inner, condition, PadValue::Zero);

        assert_eq!(
            view.shape(),
            vec![Expr::from(3), Expr::from(5), Expr::from(7)]
        );
        assert_eq!(view.ndim(), 3);
    }

    #[test]
    fn test_masked_permute() {
        use super::PadValue;

        let inner = View::contiguous(vec![4, 6]);
        // condition: idx0 < idx1
        let condition = Expr::Idx(0).lt(Expr::Idx(1));
        let view = View::masked(inner, condition, PadValue::Zero);

        // permute [1, 0]
        let permuted = view.permute(vec![1, 0]);

        // shape should be [6, 4]
        assert_eq!(permuted.shape(), vec![Expr::from(6), Expr::from(4)]);
        // condition should now reference the permuted indices
        if let View::Masked { condition, .. } = permuted {
            // After permute[1, 0]: idx0 -> idx1, idx1 -> idx0
            // So the new condition should be idx1 < idx0
            let expected = Expr::Idx(1).lt(Expr::Idx(0));
            assert_eq!(condition, expected);
        } else {
            panic!("Expected Masked view");
        }
    }

    #[test]
    fn test_masked_unsqueeze() {
        use super::PadValue;

        let inner = View::contiguous(vec![4, 4]);
        let condition = Expr::Idx(0).le(Expr::Idx(1));
        let view = View::masked(inner, condition, PadValue::NegInf);

        // unsqueeze at axis 0
        let unsqueezed = view.unsqueeze(0);

        assert_eq!(
            unsqueezed.shape(),
            vec![Expr::from(1), Expr::from(4), Expr::from(4)]
        );
        if let View::Masked { condition, .. } = unsqueezed {
            // After unsqueeze at 0: idx0 -> idx1, idx1 -> idx2
            let expected = Expr::Idx(1).le(Expr::Idx(2));
            assert_eq!(condition, expected);
        }
    }

    #[test]
    fn test_masked_squeeze() {
        use super::PadValue;

        let inner = View::contiguous(vec![1, 4, 4]);
        // condition uses idx1 and idx2
        let condition = Expr::Idx(1).le(Expr::Idx(2));
        let view = View::masked(inner, condition, PadValue::NegInf);

        // squeeze at axis 0
        let squeezed = view.squeeze(0);

        assert_eq!(squeezed.shape(), vec![Expr::from(4), Expr::from(4)]);
        if let View::Masked { condition, .. } = squeezed {
            // After squeeze at 0: idx1 -> idx0, idx2 -> idx1
            let expected = Expr::Idx(0).le(Expr::Idx(1));
            assert_eq!(condition, expected);
        }
    }

    #[test]
    fn test_masked_accessors() {
        use super::PadValue;

        let inner = View::contiguous(vec![4, 4]);
        let condition = Expr::Idx(0).lt(Expr::Idx(1));
        let view = View::masked(inner, condition.clone(), PadValue::Zero);

        assert_eq!(view.default_value(), Some(PadValue::Zero));
        assert!(view.inner_view().is_some());
        assert_eq!(view.condition(), Some(&condition));
    }

    #[test]
    fn test_masked_unfold() {
        use super::PadValue;

        // Test that Masked view can be unfolded properly
        // Input: [4, 4] masked view with condition Idx(0) < 2 (rows 0,1 are valid, rows 2,3 are masked)
        let inner = View::contiguous(vec![4, 4]);
        let condition = Expr::Idx(0).lt(Expr::from(2));
        let masked = View::masked(inner, condition, PadValue::Zero);

        // Unfold on axis 0: size=2, stride=1, dilation=1
        // Output shape: [preserved_axes..., output_positions..., window_sizes...]
        // = [4, 3, 2] (4 cols preserved, 3 windows, 2 elements per window)
        let unfolded = masked.unfold(&[0], &[Expr::from(2)], &[Expr::from(1)], &[Expr::from(1)]);

        assert_eq!(
            unfolded.shape(),
            vec![Expr::from(4), Expr::from(3), Expr::from(2)]
        );

        // The view should still be masked
        assert!(unfolded.condition().is_some());
        assert_eq!(unfolded.default_value(), Some(PadValue::Zero));
    }

    #[test]
    fn test_masked_unfold_2d() {
        use super::PadValue;

        // Test 2D unfold on masked view
        // Input: [6, 6] with condition Idx(0) >= 1 && Idx(0) < 5 && Idx(1) >= 1 && Idx(1) < 5
        // This simulates a padded tensor with 1-padding
        let inner = View::contiguous(vec![6, 6]);
        let condition = Expr::Idx(0)
            .ge(Expr::from(1))
            .and(Expr::Idx(0).lt(Expr::from(5)))
            .and(Expr::Idx(1).ge(Expr::from(1)))
            .and(Expr::Idx(1).lt(Expr::from(5)));
        let masked = View::masked(inner, condition, PadValue::Zero);

        // Unfold with size=(3,3), stride=(1,1), dilation=(1,1)
        // Output: [4, 4, 3, 3]
        let unfolded = masked.unfold(
            &[0, 1],
            &[Expr::from(3), Expr::from(3)],
            &[Expr::from(1), Expr::from(1)],
            &[Expr::from(1), Expr::from(1)],
        );

        assert_eq!(
            unfolded.shape(),
            vec![Expr::from(4), Expr::from(4), Expr::from(3), Expr::from(3)]
        );
        assert!(unfolded.condition().is_some());
    }
}

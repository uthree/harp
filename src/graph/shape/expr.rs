use std::fmt;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    // 定数
    Const(i64),

    // ループインデックス変数 (IndexExpr View用)
    // Idx(0) = ridx0, Idx(1) = ridx1, ...
    Idx(usize),

    // 算術演算
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),

    // 比較・論理演算（プリミティブのみ）
    // 結果は 0 (false) または 1 (true) の整数値
    /// 小なり比較: a < b → 1 if true, 0 if false
    Lt(Box<Self>, Box<Self>),
    /// 論理積: a && b → 1 if both non-zero, 0 otherwise
    And(Box<Self>, Box<Self>),
    /// 論理否定: !a → 1 if a == 0, 0 otherwise
    Not(Box<Self>),

    /// 別ソースバッファからインデックス値を読み込む（Gather操作用）
    ///
    /// GraphNode.srcの指定インデックスのバッファから値を読み込む。
    /// - `src_index`: GraphNode.srcのインデックス（通常1以上、0はデータソース）
    /// - `offset_expr`: 読み込み位置を計算する式（Idx変数を含む）
    ///
    /// # Example
    /// ```text
    /// // gather(input, dim=1, index) の表現:
    /// // output[i][j][k] = input[i][index[i][j][k]][k]
    /// LoadIndex {
    ///     src_index: 1,  // index バッファ
    ///     offset_expr: Idx(0) * J * K + Idx(1) * K + Idx(2),
    /// }
    /// ```
    LoadIndex {
        src_index: usize,
        offset_expr: Box<Self>,
    },
}

impl From<Expr> for crate::ast::AstNode {
    fn from(expr: Expr) -> Self {
        use crate::ast::{AstNode, Literal};

        // 変換前にsimplifyして可読性を向上
        let expr = expr.simplify();
        match expr {
            Expr::Const(c) => AstNode::Const(Literal::I64(c)),
            Expr::Idx(i) => {
                // ループインデックス変数をridx変数に変換
                AstNode::Var(format!("ridx{}", i))
            }
            Expr::Add(l, r) => AstNode::Add(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Sub(l, r) => {
                // a - b = a + (-b)
                let left: AstNode = (*l).into();
                let right: AstNode = (*r).into();
                left + (-right)
            }
            Expr::Mul(l, r) => AstNode::Mul(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Div(l, r) => {
                // a / b = a * recip(b)
                let left: AstNode = (*l).into();
                let right: AstNode = (*r).into();
                left * crate::ast::helper::recip(right)
            }
            Expr::Rem(l, r) => AstNode::Rem(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Lt(l, r) => AstNode::Lt(Box::new((*l).into()), Box::new((*r).into())),
            Expr::And(l, r) => AstNode::And(Box::new((*l).into()), Box::new((*r).into())),
            Expr::Not(a) => AstNode::Not(Box::new((*a).into())),
            Expr::LoadIndex { .. } => {
                // LoadIndexはLowering時に特別な処理が必要
                // ここでは直接変換できないのでpanic
                panic!("LoadIndex cannot be directly converted to AstNode; use Lowering instead")
            }
        }
    }
}

impl TryFrom<&crate::ast::AstNode> for Expr {
    type Error = &'static str;

    /// AstNodeからExprへの変換（シンプルな算術式のみ対応）
    fn try_from(node: &crate::ast::AstNode) -> Result<Self, Self::Error> {
        use crate::ast::{AstNode, Literal};

        match node {
            AstNode::Const(Literal::I64(v)) => Ok(Expr::Const(*v)),
            AstNode::Const(Literal::I32(v)) => Ok(Expr::Const(*v as i64)),
            AstNode::Const(Literal::F32(v)) => Ok(Expr::Const(*v as i64)),
            AstNode::Var(name) => {
                // ridx形式の変数名をIdxに変換
                if let Some(stripped) = name.strip_prefix("ridx")
                    && let Ok(idx) = stripped.parse::<usize>()
                {
                    return Ok(Expr::Idx(idx));
                }
                Err("Variable expressions are not supported; only loop indices (ridxN) are allowed")
            }
            AstNode::Add(l, r) => {
                let left = Expr::try_from(l.as_ref())?;
                let right = Expr::try_from(r.as_ref())?;
                Ok(left + right)
            }
            AstNode::Mul(l, r) => {
                let left = Expr::try_from(l.as_ref())?;
                let right = Expr::try_from(r.as_ref())?;
                Ok(left * right)
            }
            AstNode::Rem(l, r) => {
                let left = Expr::try_from(l.as_ref())?;
                let right = Expr::try_from(r.as_ref())?;
                Ok(left % right)
            }
            AstNode::Idiv(l, r) => {
                let left = Expr::try_from(l.as_ref())?;
                let right = Expr::try_from(r.as_ref())?;
                Ok(left / right)
            }
            _ => Err("Unsupported AstNode for Expr conversion"),
        }
    }
}

impl Expr {
    pub fn is_zero(&self) -> bool {
        matches!(self, Expr::Const(0))
    }

    pub fn is_one(&self) -> bool {
        matches!(self, Expr::Const(1))
    }

    /// 定数値を取得（定数の場合のみ）
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::shape::Expr;
    ///
    /// let expr = Expr::Const(42);
    /// assert_eq!(expr.as_const(), Some(42));
    ///
    /// let idx = Expr::Idx(0);
    /// assert_eq!(idx.as_const(), None);
    /// ```
    pub fn as_const(&self) -> Option<i64> {
        match self {
            Expr::Const(v) => Some(*v),
            _ => None,
        }
    }

    /// 定数値をusizeとして取得（定数の場合のみ）
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::shape::Expr;
    ///
    /// let expr = Expr::Const(42);
    /// assert_eq!(expr.as_usize(), Some(42));
    ///
    /// let negative = Expr::Const(-1);
    /// assert_eq!(negative.as_usize(), None);
    /// ```
    pub fn as_usize(&self) -> Option<usize> {
        match self {
            Expr::Const(v) if *v >= 0 => Some(*v as usize),
            _ => None,
        }
    }

    /// 定数値を強制的に取得（定数でない場合はパニック）
    ///
    /// # Panics
    ///
    /// 式が定数でない場合にパニックします。
    ///
    /// # Examples
    ///
    /// ```should_panic
    /// use harp::shape::Expr;
    ///
    /// let idx = Expr::Idx(0);
    /// idx.expect_const("loop index not allowed"); // パニック
    /// ```
    pub fn expect_const(&self, msg: &str) -> i64 {
        self.as_const()
            .unwrap_or_else(|| panic!("Expected constant expression: {}", msg))
    }

    /// 定数値をusizeとして強制的に取得（定数でない場合はパニック）
    ///
    /// # Panics
    ///
    /// 式が定数でない場合、または負の値の場合にパニックします。
    pub fn expect_usize(&self, msg: &str) -> usize {
        self.as_usize()
            .unwrap_or_else(|| panic!("Expected non-negative constant: {}", msg))
    }

    /// 定数式を評価する
    ///
    /// # Returns
    /// * `Ok(i64)` - 評価結果
    /// * `Err(String)` - 評価できない式（ループインデックスなど）の場合
    ///
    /// # Examples
    ///
    /// ```
    /// use harp::shape::Expr;
    ///
    /// let expr = Expr::Const(8) * Expr::Const(4);
    /// assert_eq!(expr.evaluate(), Ok(32));
    ///
    /// let idx_expr = Expr::Idx(0) + Expr::Const(1);
    /// assert!(idx_expr.evaluate().is_err());
    /// ```
    pub fn evaluate(&self) -> Result<i64, String> {
        match self {
            Expr::Const(v) => Ok(*v),
            Expr::Idx(i) => Err(format!("Cannot evaluate loop index Idx({})", i)),
            Expr::Add(l, r) => Ok(l.evaluate()? + r.evaluate()?),
            Expr::Sub(l, r) => Ok(l.evaluate()? - r.evaluate()?),
            Expr::Mul(l, r) => Ok(l.evaluate()? * r.evaluate()?),
            Expr::Div(l, r) => {
                let rv = r.evaluate()?;
                if rv == 0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(l.evaluate()? / rv)
                }
            }
            Expr::Rem(l, r) => {
                let rv = r.evaluate()?;
                if rv == 0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(l.evaluate()? % rv)
                }
            }
            Expr::Lt(l, r) => Ok(if l.evaluate()? < r.evaluate()? { 1 } else { 0 }),
            Expr::And(l, r) => {
                let lv = l.evaluate()?;
                let rv = r.evaluate()?;
                Ok(if lv != 0 && rv != 0 { 1 } else { 0 })
            }
            Expr::Not(a) => Ok(if a.evaluate()? == 0 { 1 } else { 0 }),
            Expr::LoadIndex { src_index, .. } => Err(format!(
                "Cannot evaluate LoadIndex(src_index={}): requires runtime buffer access",
                src_index
            )),
        }
    }

    // =====================================================================
    // 比較・論理演算ヘルパーメソッド
    // =====================================================================

    /// 小なり比較: self < other → 1 if true, 0 if false
    pub fn lt(self, other: impl Into<Expr>) -> Expr {
        Expr::Lt(Box::new(self), Box::new(other.into()))
    }

    /// 大なり比較: self > other → 導出: other < self
    pub fn gt(self, other: impl Into<Expr>) -> Expr {
        other.into().lt(self)
    }

    /// 以下比較: self <= other → 導出: !(other < self)
    pub fn le(self, other: impl Into<Expr>) -> Expr {
        let other = other.into();
        Expr::Not(Box::new(other.lt(self)))
    }

    /// 以上比較: self >= other → 導出: !(self < other)
    pub fn ge(self, other: impl Into<Expr>) -> Expr {
        Expr::Not(Box::new(self.lt(other)))
    }

    /// 等価比較: self == other → 導出: (self <= other) && (other <= self)
    pub fn eq_expr(self, other: impl Into<Expr>) -> Expr {
        let other = other.into();
        let le_ab = self.clone().le(other.clone());
        let le_ba = other.le(self);
        Expr::And(Box::new(le_ab), Box::new(le_ba))
    }

    /// 非等価比較: self != other → 導出: !(self == other)
    pub fn ne_expr(self, other: impl Into<Expr>) -> Expr {
        Expr::Not(Box::new(self.eq_expr(other)))
    }

    /// 論理積: self && other → 1 if both non-zero, 0 otherwise
    pub fn and(self, other: impl Into<Expr>) -> Expr {
        Expr::And(Box::new(self), Box::new(other.into()))
    }

    /// 論理和: self || other → 導出: !(!self && !other)
    pub fn or(self, other: impl Into<Expr>) -> Expr {
        let not_self = Expr::Not(Box::new(self));
        let not_other = Expr::Not(Box::new(other.into()));
        Expr::Not(Box::new(Expr::And(Box::new(not_self), Box::new(not_other))))
    }

    /// 定数式をusizeとして評価する
    ///
    /// # Returns
    /// * `Ok(usize)` - 評価結果（非負の場合）
    /// * `Err(String)` - 評価できない式、または結果が負の場合
    pub fn evaluate_usize(&self) -> Result<usize, String> {
        let result = self.evaluate()?;
        if result < 0 {
            Err(format!("Expected non-negative value, got {}", result))
        } else {
            Ok(result as usize)
        }
    }

    /// Idx変数を置換する
    ///
    /// # Arguments
    /// * `idx` - 置換するインデックス
    /// * `replacement` - 置換後の式
    ///
    /// # Examples
    /// ```
    /// use harp::shape::Expr;
    ///
    /// let expr = Expr::Idx(0) * Expr::from(4) + Expr::Idx(1);
    /// let substituted = expr.substitute_idx(0, Expr::from(5));
    /// // => 5 * 4 + Idx(1) = 20 + Idx(1)
    /// ```
    pub fn substitute_idx(self, idx: usize, replacement: Expr) -> Self {
        match self {
            Expr::Idx(i) if i == idx => replacement,
            Expr::Add(l, r) => Expr::Add(
                Box::new(l.substitute_idx(idx, replacement.clone())),
                Box::new(r.substitute_idx(idx, replacement)),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(l.substitute_idx(idx, replacement.clone())),
                Box::new(r.substitute_idx(idx, replacement)),
            ),
            Expr::Mul(l, r) => Expr::Mul(
                Box::new(l.substitute_idx(idx, replacement.clone())),
                Box::new(r.substitute_idx(idx, replacement)),
            ),
            Expr::Div(l, r) => Expr::Div(
                Box::new(l.substitute_idx(idx, replacement.clone())),
                Box::new(r.substitute_idx(idx, replacement)),
            ),
            Expr::Rem(l, r) => Expr::Rem(
                Box::new(l.substitute_idx(idx, replacement.clone())),
                Box::new(r.substitute_idx(idx, replacement)),
            ),
            Expr::Lt(l, r) => Expr::Lt(
                Box::new(l.substitute_idx(idx, replacement.clone())),
                Box::new(r.substitute_idx(idx, replacement)),
            ),
            Expr::And(l, r) => Expr::And(
                Box::new(l.substitute_idx(idx, replacement.clone())),
                Box::new(r.substitute_idx(idx, replacement)),
            ),
            Expr::Not(a) => Expr::Not(Box::new(a.substitute_idx(idx, replacement))),
            Expr::LoadIndex {
                src_index,
                offset_expr,
            } => Expr::LoadIndex {
                src_index,
                offset_expr: Box::new(offset_expr.substitute_idx(idx, replacement)),
            },
            other => other,
        }
    }

    /// Idx変数を順列に従って並べ替える
    ///
    /// axes[i] = j は「旧軸jが新軸iになる」を意味する。
    /// したがって Idx(j) -> Idx(inverse_axes[j]) に変換する。
    ///
    /// # Arguments
    /// * `axes` - 順列（axes[new_axis] = old_axis）
    ///
    /// # Examples
    /// ```
    /// use harp::shape::Expr;
    ///
    /// // permute([2, 0, 1]): 旧軸2->新軸0, 旧軸0->新軸1, 旧軸1->新軸2
    /// let expr = Expr::Idx(0) * Expr::from(12) + Expr::Idx(1) * Expr::from(4) + Expr::Idx(2);
    /// let permuted = expr.permute_idx(&[2, 0, 1]);
    /// // Idx(0)->Idx(1), Idx(1)->Idx(2), Idx(2)->Idx(0)
    /// // => Idx(1) * 12 + Idx(2) * 4 + Idx(0)
    /// ```
    pub fn permute_idx(self, axes: &[usize]) -> Self {
        // inverse_axes[old_axis] = new_axis
        let mut inverse_axes = vec![0; axes.len()];
        for (new_axis, &old_axis) in axes.iter().enumerate() {
            inverse_axes[old_axis] = new_axis;
        }
        self.permute_idx_with_inverse(&inverse_axes)
    }

    fn permute_idx_with_inverse(self, inverse_axes: &[usize]) -> Self {
        match self {
            Expr::Idx(i) if i < inverse_axes.len() => Expr::Idx(inverse_axes[i]),
            Expr::Add(l, r) => Expr::Add(
                Box::new(l.permute_idx_with_inverse(inverse_axes)),
                Box::new(r.permute_idx_with_inverse(inverse_axes)),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(l.permute_idx_with_inverse(inverse_axes)),
                Box::new(r.permute_idx_with_inverse(inverse_axes)),
            ),
            Expr::Mul(l, r) => Expr::Mul(
                Box::new(l.permute_idx_with_inverse(inverse_axes)),
                Box::new(r.permute_idx_with_inverse(inverse_axes)),
            ),
            Expr::Div(l, r) => Expr::Div(
                Box::new(l.permute_idx_with_inverse(inverse_axes)),
                Box::new(r.permute_idx_with_inverse(inverse_axes)),
            ),
            Expr::Rem(l, r) => Expr::Rem(
                Box::new(l.permute_idx_with_inverse(inverse_axes)),
                Box::new(r.permute_idx_with_inverse(inverse_axes)),
            ),
            Expr::Lt(l, r) => Expr::Lt(
                Box::new(l.permute_idx_with_inverse(inverse_axes)),
                Box::new(r.permute_idx_with_inverse(inverse_axes)),
            ),
            Expr::And(l, r) => Expr::And(
                Box::new(l.permute_idx_with_inverse(inverse_axes)),
                Box::new(r.permute_idx_with_inverse(inverse_axes)),
            ),
            Expr::Not(a) => Expr::Not(Box::new(a.permute_idx_with_inverse(inverse_axes))),
            Expr::LoadIndex {
                src_index,
                offset_expr,
            } => Expr::LoadIndex {
                src_index,
                offset_expr: Box::new(offset_expr.permute_idx_with_inverse(inverse_axes)),
            },
            other => other,
        }
    }

    /// Idx変数をシフトする（unsqueeze/squeeze用）
    ///
    /// threshold以上のIdx(i)をIdx(i + delta)に変換する。
    /// deltaが負の場合はIdx(i - |delta|)になる。
    ///
    /// # Arguments
    /// * `threshold` - この値以上のインデックスをシフト
    /// * `delta` - シフト量（正または負）
    pub fn shift_idx(self, threshold: usize, delta: isize) -> Self {
        match self {
            Expr::Idx(i) if i >= threshold => {
                let new_i = (i as isize + delta) as usize;
                Expr::Idx(new_i)
            }
            Expr::Add(l, r) => Expr::Add(
                Box::new(l.shift_idx(threshold, delta)),
                Box::new(r.shift_idx(threshold, delta)),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(l.shift_idx(threshold, delta)),
                Box::new(r.shift_idx(threshold, delta)),
            ),
            Expr::Mul(l, r) => Expr::Mul(
                Box::new(l.shift_idx(threshold, delta)),
                Box::new(r.shift_idx(threshold, delta)),
            ),
            Expr::Div(l, r) => Expr::Div(
                Box::new(l.shift_idx(threshold, delta)),
                Box::new(r.shift_idx(threshold, delta)),
            ),
            Expr::Rem(l, r) => Expr::Rem(
                Box::new(l.shift_idx(threshold, delta)),
                Box::new(r.shift_idx(threshold, delta)),
            ),
            Expr::Lt(l, r) => Expr::Lt(
                Box::new(l.shift_idx(threshold, delta)),
                Box::new(r.shift_idx(threshold, delta)),
            ),
            Expr::And(l, r) => Expr::And(
                Box::new(l.shift_idx(threshold, delta)),
                Box::new(r.shift_idx(threshold, delta)),
            ),
            Expr::Not(a) => Expr::Not(Box::new(a.shift_idx(threshold, delta))),
            Expr::LoadIndex {
                src_index,
                offset_expr,
            } => Expr::LoadIndex {
                src_index,
                offset_expr: Box::new(offset_expr.shift_idx(threshold, delta)),
            },
            other => other,
        }
    }

    pub fn simplify(self) -> Self {
        match self {
            Expr::Add(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(0), e) | (e, Expr::Const(0)) => e,
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l + r),
                    (l, r) => l + r,
                }
            }
            Expr::Sub(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (e, Expr::Const(0)) => e,
                    (l, r) if l == r => Expr::Const(0),
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l - r),
                    (Expr::Add(a, b), r) if *b == r => *a,
                    (Expr::Add(a, b), r) if *a == r => *b,
                    // Simplify double negation: 0 - (a - b) -> b - a
                    (Expr::Const(0), Expr::Sub(a, b)) => (*b - *a).simplify(),
                    (l, r) => l - r,
                }
            }
            Expr::Mul(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (Expr::Const(0), _) | (_, Expr::Const(0)) => Expr::Const(0),
                    (Expr::Const(1), e) | (e, Expr::Const(1)) => e,
                    (Expr::Const(-1), e) => (-e).simplify(),
                    (e, Expr::Const(-1)) => (-e).simplify(),
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l * r),
                    (l, r) => l * r,
                }
            }
            Expr::Div(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (_, Expr::Const(0)) => panic!("division by zero"),
                    (e, Expr::Const(1)) => e,
                    (l, r) if l == r => Expr::Const(1),
                    (Expr::Const(0), _) => Expr::Const(0),
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l / r),
                    (l, r) => l / r,
                }
            }
            Expr::Rem(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (lhs, rhs) {
                    (_, Expr::Const(0)) => panic!("division by zero"),
                    (_, Expr::Const(1)) => Expr::Const(0),
                    (l, r) if l == r => Expr::Const(0),
                    (Expr::Const(0), _) => Expr::Const(0),
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(l % r),
                    (l, r) => l % r,
                }
            }
            Expr::LoadIndex {
                src_index,
                offset_expr,
            } => Expr::LoadIndex {
                src_index,
                offset_expr: Box::new(offset_expr.simplify()),
            },
            // 比較・論理演算
            Expr::Lt(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (&lhs, &rhs) {
                    (Expr::Const(l), Expr::Const(r)) => Expr::Const(if l < r { 1 } else { 0 }),
                    // x < x は常に false
                    _ if lhs == rhs => Expr::Const(0),
                    (l, r) => Expr::Lt(Box::new(l.clone()), Box::new(r.clone())),
                }
            }
            Expr::And(lhs, rhs) => {
                let lhs = lhs.simplify();
                let rhs = rhs.simplify();
                match (&lhs, &rhs) {
                    // 定数畳み込み
                    (Expr::Const(0), _) | (_, Expr::Const(0)) => Expr::Const(0),
                    (Expr::Const(l), Expr::Const(r)) => {
                        Expr::Const(if *l != 0 && *r != 0 { 1 } else { 0 })
                    }
                    // 0以外の定数は true として扱う
                    (Expr::Const(_), e) => e.clone(),
                    (e, Expr::Const(_)) => e.clone(),
                    // 冪等性: x && x = x
                    _ if lhs == rhs => lhs,
                    // x && !x = 0
                    (x, Expr::Not(y)) | (Expr::Not(y), x) if *x == **y => Expr::Const(0),
                    (l, r) => Expr::And(Box::new(l.clone()), Box::new(r.clone())),
                }
            }
            Expr::Not(a) => {
                let a = a.simplify();
                match a {
                    // 定数畳み込み
                    Expr::Const(0) => Expr::Const(1),
                    Expr::Const(_) => Expr::Const(0),
                    // 二重否定除去: !!x = x
                    Expr::Not(inner) => *inner,
                    other => Expr::Not(Box::new(other)),
                }
            }
            _ => self,
        }
    }

    /// LoadIndexを含むかどうかを判定
    pub fn contains_load_index(&self) -> bool {
        match self {
            Expr::LoadIndex { .. } => true,
            Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::Mul(l, r)
            | Expr::Div(l, r)
            | Expr::Rem(l, r)
            | Expr::Lt(l, r)
            | Expr::And(l, r) => l.contains_load_index() || r.contains_load_index(),
            Expr::Not(a) => a.contains_load_index(),
            _ => false,
        }
    }

    /// LoadIndexのsrc_indexをシフトする
    ///
    /// View融合時にsrc配列がマージされる際、LoadIndexが参照するインデックスを
    /// 調整するために使用します。
    ///
    /// # Arguments
    /// * `delta` - シフト量（正の値で増加、負の値で減少）
    ///
    /// # Example
    /// ```
    /// use harp::shape::Expr;
    ///
    /// // LoadIndex { src_index: 1, ... } -> LoadIndex { src_index: 2, ... }
    /// let expr = Expr::LoadIndex {
    ///     src_index: 1,
    ///     offset_expr: Box::new(Expr::Idx(0)),
    /// };
    /// let shifted = expr.shift_load_index(1);
    /// if let Expr::LoadIndex { src_index, .. } = shifted {
    ///     assert_eq!(src_index, 2);
    /// }
    /// ```
    pub fn shift_load_index(self, delta: isize) -> Self {
        match self {
            Expr::LoadIndex {
                src_index,
                offset_expr,
            } => {
                let new_index = (src_index as isize + delta) as usize;
                Expr::LoadIndex {
                    src_index: new_index,
                    offset_expr: Box::new(offset_expr.shift_load_index(delta)),
                }
            }
            Expr::Add(l, r) => Expr::Add(
                Box::new(l.shift_load_index(delta)),
                Box::new(r.shift_load_index(delta)),
            ),
            Expr::Sub(l, r) => Expr::Sub(
                Box::new(l.shift_load_index(delta)),
                Box::new(r.shift_load_index(delta)),
            ),
            Expr::Mul(l, r) => Expr::Mul(
                Box::new(l.shift_load_index(delta)),
                Box::new(r.shift_load_index(delta)),
            ),
            Expr::Div(l, r) => Expr::Div(
                Box::new(l.shift_load_index(delta)),
                Box::new(r.shift_load_index(delta)),
            ),
            Expr::Rem(l, r) => Expr::Rem(
                Box::new(l.shift_load_index(delta)),
                Box::new(r.shift_load_index(delta)),
            ),
            Expr::Lt(l, r) => Expr::Lt(
                Box::new(l.shift_load_index(delta)),
                Box::new(r.shift_load_index(delta)),
            ),
            Expr::And(l, r) => Expr::And(
                Box::new(l.shift_load_index(delta)),
                Box::new(r.shift_load_index(delta)),
            ),
            Expr::Not(a) => Expr::Not(Box::new(a.shift_load_index(delta))),
            // Const, Idxはそのまま
            _ => self,
        }
    }
}

macro_rules! impl_from_integer_for_expr {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Expr {
                fn from(n: $t) -> Self {
                    Expr::Const(n as i64)
                }
            }
        )*
    };
}

impl_from_integer_for_expr!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// 数値型からVec<Expr>を初期化するマクロ
///
/// - 数値型（i32, usize等）→ `Expr::Const`
///
/// # 使用例
///
/// ```
/// use harp::shape;
/// use harp::shape::Expr;
///
/// // 静的な形状
/// let shape = shape![2, 3, 4];
/// assert_eq!(shape, vec![Expr::Const(2), Expr::Const(3), Expr::Const(4)]);
///
/// // 1次元
/// let shape = shape![32];
/// assert_eq!(shape, vec![Expr::Const(32)]);
/// ```
#[macro_export]
macro_rules! shape {
    () => {
        ::std::vec::Vec::<$crate::shape::Expr>::new()
    };
    ($($elem:expr),+ $(,)?) => {
        vec![$(::std::convert::Into::<$crate::shape::Expr>::into($elem)),+]
    };
}

macro_rules! impl_expr_binary_op {
    ($trait:ident, $fname:ident, $variant:expr) => {
        impl<T: Into<Expr>> $trait<T> for Expr {
            type Output = Expr;
            fn $fname(self, rhs: T) -> Self::Output {
                $variant(Box::new(self), Box::new(rhs.into()))
            }
        }
    };
}

impl_expr_binary_op!(Add, add, Expr::Add);
impl_expr_binary_op!(Sub, sub, Expr::Sub);
impl_expr_binary_op!(Mul, mul, Expr::Mul);
impl_expr_binary_op!(Div, div, Expr::Div);
impl_expr_binary_op!(Rem, rem, Expr::Rem);

macro_rules! impl_expr_assign_op {
    ($trait:ident, $fname:ident, $op:tt) => {
        impl<T: Into<Expr>> $trait<T> for Expr {
            fn $fname(&mut self, rhs: T) {
                *self = self.clone() $op rhs.into();
            }
        }
    };
}

impl_expr_assign_op!(AddAssign, add_assign, +);
impl_expr_assign_op!(SubAssign, sub_assign, -);
impl_expr_assign_op!(MulAssign, mul_assign, *);
impl_expr_assign_op!(DivAssign, div_assign, /);
impl_expr_assign_op!(RemAssign, rem_assign, %);

impl Neg for Expr {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Expr::from(0isize) - self
    }
}

impl std::ops::Not for Expr {
    type Output = Self;

    fn not(self) -> Self::Output {
        Expr::Not(Box::new(self))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(n) => write!(f, "{}", n),
            Expr::Idx(i) => write!(f, "idx{}", i),
            Expr::Add(lhs, rhs) => {
                // Add parentheses only when necessary
                let needs_parens_lhs = matches!(**lhs, Expr::Sub(_, _));
                let needs_parens_rhs = matches!(**rhs, Expr::Sub(_, _));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " + ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Sub(lhs, rhs) => {
                let needs_parens_rhs = !matches!(**rhs, Expr::Const(_) | Expr::Idx(_));

                write!(f, "{}", lhs)?;
                write!(f, " - ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Mul(lhs, rhs) => {
                let needs_parens_lhs = matches!(**lhs, Expr::Add(_, _) | Expr::Sub(_, _));
                let needs_parens_rhs = matches!(**rhs, Expr::Add(_, _) | Expr::Sub(_, _));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " * ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Div(lhs, rhs) => {
                let needs_parens_lhs = matches!(**lhs, Expr::Add(_, _) | Expr::Sub(_, _));
                let needs_parens_rhs = !matches!(**rhs, Expr::Const(_) | Expr::Idx(_));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " / ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::Rem(lhs, rhs) => {
                let needs_parens_lhs = matches!(**lhs, Expr::Add(_, _) | Expr::Sub(_, _));
                let needs_parens_rhs = !matches!(**rhs, Expr::Const(_) | Expr::Idx(_));

                if needs_parens_lhs {
                    write!(f, "({})", lhs)?;
                } else {
                    write!(f, "{}", lhs)?;
                }
                write!(f, " % ")?;
                if needs_parens_rhs {
                    write!(f, "({})", rhs)
                } else {
                    write!(f, "{}", rhs)
                }
            }
            Expr::LoadIndex {
                src_index,
                offset_expr,
            } => {
                write!(f, "load[{}]({})", src_index, offset_expr)
            }
            Expr::Lt(lhs, rhs) => {
                write!(f, "({} < {})", lhs, rhs)
            }
            Expr::And(lhs, rhs) => {
                write!(f, "({} && {})", lhs, rhs)
            }
            Expr::Not(a) => {
                write!(f, "!({})", a)
            }
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

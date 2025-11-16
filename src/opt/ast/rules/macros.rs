//! AST書き換えルール生成のためのマクロ集
//!
//! このモジュールは、一般的な代数的変形パターンを簡潔に記述するためのマクロを提供します。

/// 単位元ルールを生成するマクロ
/// op(a, identity) = a および op(identity, a) = a のパターン
macro_rules! identity_rules {
    ($right_name:ident, $left_name:ident, $op:ident, $identity:expr, $right_doc:expr, $left_doc:expr) => {
        #[doc = $right_doc]
        pub fn $right_name() -> Rc<AstRewriteRule> {
            astpat!(|a| {
                AstNode::$op(Box::new(a), Box::new(AstNode::Const($identity)))
            } => {
                a
            })
        }

        #[doc = $left_doc]
        pub fn $left_name() -> Rc<AstRewriteRule> {
            astpat!(|a| {
                AstNode::$op(Box::new(AstNode::Const($identity)), Box::new(a))
            } => {
                a
            })
        }
    };
}

/// 零元ルールを生成するマクロ
/// op(a, zero) = zero および op(zero, a) = zero のパターン
macro_rules! zero_rules {
    ($right_name:ident, $left_name:ident, $op:ident, $zero:expr, $right_doc:expr, $left_doc:expr) => {
        #[doc = $right_doc]
        pub fn $right_name() -> Rc<AstRewriteRule> {
            astpat!(|_a| {
                AstNode::$op(Box::new(_a), Box::new(AstNode::Const($zero)))
            } => {
                AstNode::Const($zero)
            })
        }

        #[doc = $left_doc]
        pub fn $left_name() -> Rc<AstRewriteRule> {
            astpat!(|_a| {
                AstNode::$op(Box::new(AstNode::Const($zero)), Box::new(_a))
            } => {
                AstNode::Const($zero)
            })
        }
    };
}

/// 交換則ルールを生成するマクロ
/// op(a, b) = op(b, a) のパターン
macro_rules! commutative_rule {
    ($name:ident, $op:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $name() -> Rc<AstRewriteRule> {
            astpat!(|a, b| {
                AstNode::$op(Box::new(a), Box::new(b))
            } => {
                AstNode::$op(Box::new(b), Box::new(a))
            })
        }
    };
}

/// 結合則ルールを生成するマクロ
/// op(op(a, b), c) = op(a, op(b, c)) とその逆のパターン
macro_rules! associative_rules {
    ($left_to_right:ident, $right_to_left:ident, $op:ident, $ltr_doc:expr, $rtl_doc:expr) => {
        #[doc = $ltr_doc]
        pub fn $left_to_right() -> Rc<AstRewriteRule> {
            astpat!(|a, b, c| {
                AstNode::$op(
                    Box::new(AstNode::$op(Box::new(a), Box::new(b))),
                    Box::new(c)
                )
            } => {
                AstNode::$op(
                    Box::new(a),
                    Box::new(AstNode::$op(Box::new(b), Box::new(c)))
                )
            })
        }

        #[doc = $rtl_doc]
        pub fn $right_to_left() -> Rc<AstRewriteRule> {
            astpat!(|a, b, c| {
                AstNode::$op(
                    Box::new(a),
                    Box::new(AstNode::$op(Box::new(b), Box::new(c)))
                )
            } => {
                AstNode::$op(
                    Box::new(AstNode::$op(Box::new(a), Box::new(b))),
                    Box::new(c)
                )
            })
        }
    };
}

// マクロを他のモジュールから使えるようにエクスポート
pub(crate) use associative_rules;
pub(crate) use commutative_rule;
pub(crate) use identity_rules;
pub(crate) use zero_rules;

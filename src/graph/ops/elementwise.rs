use crate::graph::{GraphNode, GraphOp};
use std::fmt;

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum ElementwiseOp {
    Add(GraphNode, GraphNode),
    Mul(GraphNode, GraphNode),
    Max(GraphNode, GraphNode),
    Mod(GraphNode, GraphNode),
    Neg(GraphNode),
    Recip(GraphNode),
    Sin(GraphNode),
    Sqrt(GraphNode),
    Log2(GraphNode),
    Exp2(GraphNode),
    // Comparison operations (return Bool type)
    LessThan(GraphNode, GraphNode),
    Eq(GraphNode, GraphNode),
    // Conditional selection (ternary operator)
    Select(GraphNode, GraphNode, GraphNode), // (cond, true_val, false_val)
}

impl fmt::Display for ElementwiseOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ElementwiseOp::Add(_, _) => write!(f, "Add"),
            ElementwiseOp::Mul(_, _) => write!(f, "Mul"),
            ElementwiseOp::Max(_, _) => write!(f, "Max"),
            ElementwiseOp::Mod(_, _) => write!(f, "Mod"),
            ElementwiseOp::Neg(_) => write!(f, "Neg"),
            ElementwiseOp::Recip(_) => write!(f, "Recip"),
            ElementwiseOp::Sin(_) => write!(f, "Sin"),
            ElementwiseOp::Sqrt(_) => write!(f, "Sqrt"),
            ElementwiseOp::Log2(_) => write!(f, "Log2"),
            ElementwiseOp::Exp2(_) => write!(f, "Exp2"),
            ElementwiseOp::LessThan(_, _) => write!(f, "LessThan"),
            ElementwiseOp::Eq(_, _) => write!(f, "Eq"),
            ElementwiseOp::Select(_, _, _) => write!(f, "Select"),
        }
    }
}

macro_rules! impl_elementwise_binary_op {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl std::ops::$trait for GraphNode {
            type Output = GraphNode;
            fn $method(self, rhs: Self) -> Self::Output {
                // dtypeチェック
                assert_eq!(
                    self.dtype, rhs.dtype,
                    "dtypes must match for element-wise operations"
                );

                // 結果のviewを決定
                let result_view = self.view.elementwise_result_view(&rhs.view);

                // 新しいGraphNodeを作成
                GraphNode::new(
                    GraphOp::Elementwise(ElementwiseOp::$variant(self.clone(), rhs.clone())),
                    self.dtype.clone(),
                    result_view,
                )
            }
        }
    };
}

macro_rules! impl_elementwise_unary_op {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl std::ops::$trait for GraphNode {
            type Output = GraphNode;
            fn $method(self) -> Self::Output {
                // 結果のviewはそのまま継承
                let result_view = self.view.clone();

                // 新しいGraphNodeを作成
                GraphNode::new(
                    GraphOp::Elementwise(ElementwiseOp::$variant(self.clone())),
                    self.dtype.clone(),
                    result_view,
                )
            }
        }
    };
}

impl_elementwise_binary_op!(Add, add, Add);
impl_elementwise_binary_op!(Mul, mul, Mul);
impl_elementwise_binary_op!(Rem, rem, Mod);
impl_elementwise_unary_op!(Neg, neg, Neg);

impl GraphNode {
    pub fn cmp_max(self, rhs: Self) -> Self {
        // dtypeチェック
        assert_eq!(
            self.dtype, rhs.dtype,
            "dtypes must match for element-wise operations"
        );

        // 結果のviewを決定
        let result_view = self.view.elementwise_result_view(&rhs.view);

        // 新しいGraphNodeを作成
        GraphNode::new(
            GraphOp::Elementwise(ElementwiseOp::Max(self.clone(), rhs.clone())),
            self.dtype.clone(),
            result_view,
        )
    }

    pub fn recip(self) -> Self {
        self.apply_unary_op(ElementwiseOp::Recip)
    }

    pub fn sin(self) -> Self {
        self.apply_unary_op(ElementwiseOp::Sin)
    }

    pub fn sqrt(self) -> Self {
        self.apply_unary_op(ElementwiseOp::Sqrt)
    }

    pub fn log2(self) -> Self {
        self.apply_unary_op(ElementwiseOp::Log2)
    }

    pub fn exp2(self) -> Self {
        self.apply_unary_op(ElementwiseOp::Exp2)
    }

    /// Less than comparison: self < rhs
    /// Returns a Bool tensor
    pub fn less_than(self, rhs: Self) -> Self {
        use crate::ast::DType;

        // 結果のviewを決定
        let result_view = self.view.elementwise_result_view(&rhs.view);

        // 新しいGraphNodeを作成（結果はBool型）
        GraphNode::new(
            GraphOp::Elementwise(ElementwiseOp::LessThan(self.clone(), rhs.clone())),
            DType::Bool,
            result_view,
        )
    }

    /// Equality comparison: self == rhs
    /// Returns a Bool tensor
    pub fn equal(self, rhs: Self) -> Self {
        use crate::ast::DType;

        // 結果のviewを決定
        let result_view = self.view.elementwise_result_view(&rhs.view);

        // 新しいGraphNodeを作成（結果はBool型）
        GraphNode::new(
            GraphOp::Elementwise(ElementwiseOp::Eq(self.clone(), rhs.clone())),
            DType::Bool,
            result_view,
        )
    }

    /// Conditional selection: cond ? true_val : false_val
    /// cond must be Bool type
    pub fn select(cond: Self, true_val: Self, false_val: Self) -> Self {
        use crate::ast::DType;

        // condはBool型でなければならない
        assert_eq!(
            cond.dtype,
            DType::Bool,
            "condition must be Bool type for select operation"
        );

        // true_valとfalse_valのdtypeは一致していなければならない
        assert_eq!(
            true_val.dtype, false_val.dtype,
            "true_val and false_val must have the same dtype"
        );

        // 結果のviewを決定（3つのオペランド全てを考慮）
        let result_view = cond.view.elementwise_result_view(&true_val.view);
        let result_view = result_view.elementwise_result_view(&false_val.view);

        // 新しいGraphNodeを作成（結果の型はtrue_val/false_valの型）
        GraphNode::new(
            GraphOp::Elementwise(ElementwiseOp::Select(
                cond.clone(),
                true_val.clone(),
                false_val.clone(),
            )),
            true_val.dtype.clone(),
            result_view,
        )
    }

    fn apply_unary_op(self, op_constructor: fn(GraphNode) -> ElementwiseOp) -> Self {
        // 結果のviewはそのまま継承
        let result_view = self.view.clone();

        // 新しいGraphNodeを作成
        GraphNode::new(
            GraphOp::Elementwise(op_constructor(self.clone())),
            self.dtype.clone(),
            result_view,
        )
    }
}

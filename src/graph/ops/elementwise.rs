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

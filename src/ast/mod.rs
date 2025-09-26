pub mod pattern;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Debug, Clone, PartialEq, Default, Eq, Hash)]
pub enum DType {
    #[default]
    F32, // float
    Usize, // size_t
    Isize, // ssize_t
    Void,

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array (for SIMD vectorization)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstLiteral {
    F32(f32),
    Usize(usize),
    Isize(isize),
}

// f32はEqを実装していないので手動でEqとHashを実装
impl Eq for ConstLiteral {}

impl std::hash::Hash for ConstLiteral {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ConstLiteral::F32(f) => {
                0u8.hash(state);
                f.to_bits().hash(state);
            }
            ConstLiteral::Usize(u) => {
                1u8.hash(state);
                u.hash(state);
            }
            ConstLiteral::Isize(i) => {
                2u8.hash(state);
                i.hash(state);
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub dtype: DType,
    pub constant: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Scope {
    pub declarations: Vec<VariableDecl>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AstNode {
    Const(ConstLiteral), // constant value
    Var(String),         // get value from variable
    Cast {
        dtype: DType,
        expr: Box<Self>,
    }, // convert another type

    // numeric ops
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Div(Box<Self>, Box<Self>),
    Max(Box<Self>, Box<Self>),
    Rem(Box<Self>, Box<Self>),
    Neg(Box<Self>),
    Recip(Box<Self>),
    Sin(Box<Self>),
    Sqrt(Box<Self>),
    Log2(Box<Self>),
    Exp2(Box<Self>),
    Rand, // 一様乱数(0.0~1.0まで)を生成
    CallFunction {
        name: String,
        args: Vec<Self>,
    },

    // statements
    Block {
        scope: Scope,
        statements: Vec<AstNode>,
    },
    Index {
        target: Box<Self>,
        index: Box<Self>,
    },
    Assign(Box<Self>, Box<Self>), // assign value to variable

    Range {
        // Forループ
        counter_name: String, // ループカウンタの変数名
        max: Box<Self>,       // ループ回数
        body: Box<Self>,
    },

    Drop(String), // drop (local) variable explicitly

    Barrier,

    // for pattern matching
    Capture(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub(crate) name: String,
    pub(crate) body: AstNode,
    pub(crate) arguments: Vec<(String, DType)>,
    pub(crate) return_type: DType,
}

impl Function {
    pub fn new(
        name: String,
        arguments: Vec<(String, DType)>,
        return_type: DType,
        body: AstNode,
    ) -> Self {
        assert!(matches!(body, AstNode::Block { .. }));
        Self {
            name,
            arguments,
            return_type,
            body,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn body(&self) -> &AstNode {
        &self.body
    }

    pub fn arguments(&self) -> &[(String, DType)] {
        &self.arguments
    }

    pub fn return_type(&self) -> &DType {
        &self.return_type
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub(crate) functions: Vec<Function>,
    pub(crate) entry_point: String,
}

macro_rules! impl_from_num_for_astnode {
    ($(($t:ty, $v: ident)),*) => {
        $(
            impl From<$t> for AstNode {
                fn from(n: $t) -> Self {
                    AstNode::Const(ConstLiteral::$v(n))
                }
            }
        )*
    };
}
impl_from_num_for_astnode!((usize, Usize), (isize, Isize), (f32, F32));

impl From<ConstLiteral> for AstNode {
    fn from(c: ConstLiteral) -> Self {
        AstNode::Const(c)
    }
}

macro_rules! impl_astnode_binary_op {
    ($trait:ident, $fname:ident, $variant:ident) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: T) -> Self::Output {
                AstNode::$variant(Box::new(self), Box::new(rhs.into()))
            }
        }

        impl $trait<&AstNode> for &AstNode {
            type Output = AstNode;
            fn $fname(self, rhs: &AstNode) -> Self::Output {
                AstNode::$variant(Box::new(self.clone()), Box::new(rhs.clone()))
            }
        }
    };
}

impl_astnode_binary_op!(Add, add, Add);
impl_astnode_binary_op!(Mul, mul, Mul);
impl_astnode_binary_op!(Div, div, Div);
impl_astnode_binary_op!(Rem, rem, Rem);

// Subtraction: a - b = a + (-b)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Into<AstNode>> Sub<T> for AstNode {
    type Output = AstNode;
    fn sub(self, rhs: T) -> Self::Output {
        self + AstNode::Neg(Box::new(rhs.into()))
    }
}

macro_rules! impl_expr_assign_op {
    ($trait:ident, $fname:ident, $op:tt) => {
        impl<T: Into<AstNode>> $trait<T> for AstNode {
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

impl Neg for AstNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        AstNode::Neg(Box::new(self))
    }
}

macro_rules! impl_astnode_unary_op {
    ($fname:ident, $variant:ident) => {
        impl AstNode {
            pub fn $fname(self) -> Self {
                AstNode::$variant(Box::new(self))
            }
        }
    };
}

impl_astnode_unary_op!(recip, Recip);
impl_astnode_unary_op!(sin, Sin);
impl_astnode_unary_op!(sqrt, Sqrt);
impl_astnode_unary_op!(exp2, Exp2);
impl_astnode_unary_op!(log2, Log2);

impl AstNode {
    pub fn capture(n: usize) -> AstNode {
        AstNode::Capture(n)
    }

    pub fn children(&self) -> Vec<&AstNode> {
        match self {
            AstNode::Const(_) => vec![],
            AstNode::Var(_) => vec![],
            AstNode::Cast { expr, .. } => vec![expr.as_ref()],
            AstNode::Add(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Mul(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Div(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Max(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Rem(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Index { target, index } => vec![target.as_ref(), index.as_ref()],
            AstNode::Assign(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Neg(n) => vec![n.as_ref()],
            AstNode::Recip(n) => vec![n.as_ref()],
            AstNode::Sin(n) => vec![n.as_ref()],
            AstNode::Sqrt(n) => vec![n.as_ref()],
            AstNode::Log2(n) => vec![n.as_ref()],
            AstNode::Exp2(n) => vec![n.as_ref()],
            AstNode::CallFunction { args, .. } => args.iter().collect(),
            AstNode::Range { max, body, .. } => {
                vec![max.as_ref(), body.as_ref()]
            }
            AstNode::Block { statements, .. } => statements.iter().collect(),
            AstNode::Drop(_) => vec![],
            AstNode::Barrier => vec![],
            AstNode::Capture(_) => vec![],
            AstNode::Rand => vec![],
        }
    }

    // Replace all occurrences of a specific node with a new node
    pub fn replace_node(self, target: &AstNode, replacement: AstNode) -> AstNode {
        // First, recursively apply to children
        let children: Vec<AstNode> = self
            .children()
            .into_iter()
            .map(|child| child.clone().replace_node(target, replacement.clone()))
            .collect();

        // Rebuild current node with transformed children
        let new_node = self.replace_children(children);

        // Check if current node should be replaced
        if &new_node == target {
            replacement
        } else {
            new_node
        }
    }

    // Replace nodes matching a predicate with the result of a transform function
    pub fn replace_if<F, T>(self, predicate: F, transform: T) -> AstNode
    where
        F: Fn(&AstNode) -> bool + Clone,
        T: Fn(AstNode) -> AstNode + Clone,
    {
        // First, recursively apply to children
        let children: Vec<AstNode> = self
            .children()
            .into_iter()
            .map(|child| {
                child
                    .clone()
                    .replace_if(predicate.clone(), transform.clone())
            })
            .collect();

        // Rebuild current node with transformed children
        let new_node = self.replace_children(children);

        // Check if current node should be transformed
        if predicate(&new_node) {
            transform(new_node)
        } else {
            new_node
        }
    }

    // Helper function to replace children of an AstNode.
    // This is a bit verbose, but it's the only way to do it without macros.
    pub fn replace_children(self, new_children: Vec<AstNode>) -> AstNode {
        let mut children_iter = new_children.into_iter();
        match self {
            AstNode::Add(_, _) => AstNode::Add(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Mul(_, _) => AstNode::Mul(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Div(_, _) => AstNode::Div(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Max(_, _) => AstNode::Max(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Rem(_, _) => AstNode::Rem(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Index { .. } => AstNode::Index {
                target: Box::new(children_iter.next().unwrap()),
                index: Box::new(children_iter.next().unwrap()),
            },
            AstNode::Assign(_, _) => AstNode::Assign(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Cast { dtype, .. } => AstNode::Cast {
                dtype,
                expr: Box::new(children_iter.next().unwrap()),
            },
            AstNode::Neg(_) => AstNode::Neg(Box::new(children_iter.next().unwrap())),
            AstNode::Recip(_) => AstNode::Recip(Box::new(children_iter.next().unwrap())),
            AstNode::Sin(_) => AstNode::Sin(Box::new(children_iter.next().unwrap())),
            AstNode::Sqrt(_) => AstNode::Sqrt(Box::new(children_iter.next().unwrap())),
            AstNode::Log2(_) => AstNode::Log2(Box::new(children_iter.next().unwrap())),
            AstNode::Exp2(_) => AstNode::Exp2(Box::new(children_iter.next().unwrap())),
            AstNode::CallFunction { name, .. } => AstNode::CallFunction {
                name,
                args: children_iter.collect(),
            },
            AstNode::Range { counter_name, .. } => AstNode::Range {
                counter_name, // moved
                max: Box::new(children_iter.next().unwrap()),
                body: Box::new(children_iter.next().unwrap()),
            },
            AstNode::Block { scope, .. } => {
                let statements = children_iter.collect();
                AstNode::Block { scope, statements }
            }
            // Nodes without children are returned as is (moved).
            _ => self,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(1.0f32, AstNode::Const(ConstLiteral::F32(1.0)))]
    #[case(42usize, AstNode::Const(ConstLiteral::Usize(42)))]
    #[case(-10isize, AstNode::Const(ConstLiteral::Isize(-10)))]
    fn test_from_numeric_literals(#[case] input: impl Into<AstNode>, #[case] expected: AstNode) {
        assert_eq!(input.into(), expected);
    }

    #[test]
    fn test_addition() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a + b;
        assert_eq!(
            expr,
            AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_subtraction() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a - b;
        assert_eq!(
            expr,
            AstNode::Add(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Neg(Box::new(AstNode::Const(ConstLiteral::F32(
                    2.0
                )))))
            )
        );
    }

    #[test]
    fn test_multiplication() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a * b;
        assert_eq!(
            expr,
            AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_division() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a / b;
        assert_eq!(
            expr,
            AstNode::Div(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_remainder() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2.0f32.into();
        let expr = a % b;
        assert_eq!(
            expr,
            AstNode::Rem(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );
    }

    #[test]
    fn test_negation() {
        let a = AstNode::Var("a".to_string());
        let expr = -a;
        assert_eq!(expr, AstNode::Neg(Box::new(AstNode::Var("a".to_string()))));
    }

    #[test]
    fn test_unary_ops() {
        let a = AstNode::Var("a".to_string());
        assert_eq!(
            a.clone().recip(),
            AstNode::Recip(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().sin(),
            AstNode::Sin(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().sqrt(),
            AstNode::Sqrt(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.clone().exp2(),
            AstNode::Exp2(Box::new(AstNode::Var("a".to_string())))
        );
        assert_eq!(
            a.log2(),
            AstNode::Log2(Box::new(AstNode::Var("a".to_string())))
        );
    }

    #[test]
    fn test_complex_expression() {
        let a = AstNode::Var("a".to_string());
        let b = AstNode::Var("b".to_string());
        let c = 3.0f32;
        // -(a + b) * c
        let expr = -(a.clone() + b.clone()) * c;
        assert_eq!(
            expr,
            AstNode::Mul(
                Box::new(AstNode::Neg(Box::new(AstNode::Add(
                    Box::new(AstNode::Var("a".to_string())),
                    Box::new(AstNode::Var("b".to_string()))
                )))),
                Box::new(AstNode::Const(ConstLiteral::F32(3.0)))
            )
        );
    }

    #[test]
    fn test_replace_node() {
        let a = AstNode::Var("a".to_string());
        let target = AstNode::Const(ConstLiteral::F32(1.0));
        let replacement = AstNode::Const(ConstLiteral::F32(2.0));

        // a + 1.0 -> a + 2.0
        let expr = a.clone() + target.clone();
        let result = expr.replace_node(&target, replacement.clone());
        assert_eq!(result, a + replacement);
    }

    #[test]
    fn test_replace_if() {
        let a = AstNode::Var("a".to_string());
        let expr = a.clone() + AstNode::Const(ConstLiteral::F32(0.0));

        // Replace any addition with 0 with just the left operand
        let result = expr.replace_if(
            |node| matches!(node, AstNode::Add(_, r) if **r == AstNode::Const(ConstLiteral::F32(0.0))),
            |node| if let AstNode::Add(l, _) = node { *l } else { node }
        );

        assert_eq!(result, a);
    }
}

pub mod pattern;
use std::fmt;
use std::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Rem, RemAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
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

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::Usize => write!(f, "Usize"),
            DType::Isize => write!(f, "Isize"),
            DType::Void => write!(f, "Void"),
            DType::Ptr(inner) => write!(f, "Ptr<{}>", inner),
            DType::Vec(inner, size) => write!(f, "Vec<{}, {}>", inner, size),
        }
    }
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
    pub size_expr: Option<Box<AstNode>>, // For dynamic arrays, the size expression
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

    // bitwise ops
    BitAnd(Box<Self>, Box<Self>), // ビット論理積 (&)
    BitOr(Box<Self>, Box<Self>),  // ビット論理和 (|)
    BitXor(Box<Self>, Box<Self>), // ビット排他的論理和 (^)
    Shl(Box<Self>, Box<Self>),    // 左シフト (<<)
    Shr(Box<Self>, Box<Self>),    // 右シフト (>>)
    BitNot(Box<Self>),            // ビット否定 (~)

    // statements
    Block {
        scope: Scope,
        statements: Vec<AstNode>,
    },
    Assign(String, Box<Self>), // assign value to variable (lhs is variable name)
    Store {
        target: Box<Self>,
        index: Box<Self>,
        value: Box<Self>,
    }, // store value to memory location (target[index] = value)
    Deref(Box<Self>),          // dereference pointer (read value from *expr)

    Range {
        // Forループ (start から max-1 まで、stepずつインクリメント)
        counter_name: String, // ループカウンタの変数名
        start: Box<Self>,     // 開始値（デフォルトは0）
        max: Box<Self>,       // 終了値
        step: Box<Self>,      // インクリメント量（デフォルトは1）
        body: Box<Self>,
        unroll: Option<usize>, // #pragma unroll相当のヒント (None=no unroll, Some(0)=full unroll, Some(n)=unroll n times)
    },

    Drop(String), // drop (local) variable explicitly

    Barrier, // Synchronization barrier for parallel execution (separates computation generations)

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
impl_astnode_binary_op!(Rem, rem, Rem);
impl_astnode_binary_op!(BitAnd, bitand, BitAnd);
impl_astnode_binary_op!(BitOr, bitor, BitOr);
impl_astnode_binary_op!(BitXor, bitxor, BitXor);
impl_astnode_binary_op!(Shl, shl, Shl);
impl_astnode_binary_op!(Shr, shr, Shr);

// Subtraction: a - b = a + (-b)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Into<AstNode>> Sub<T> for AstNode {
    type Output = AstNode;
    fn sub(self, rhs: T) -> Self::Output {
        self + AstNode::Neg(Box::new(rhs.into()))
    }
}

// Division: a / b = a * (1/b)
#[allow(clippy::suspicious_arithmetic_impl)]
impl<T: Into<AstNode>> Div<T> for AstNode {
    type Output = AstNode;
    fn div(self, rhs: T) -> Self::Output {
        self * AstNode::Recip(Box::new(rhs.into()))
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
impl_expr_assign_op!(BitAndAssign, bitand_assign, &);
impl_expr_assign_op!(BitOrAssign, bitor_assign, |);
impl_expr_assign_op!(BitXorAssign, bitxor_assign, ^);
impl_expr_assign_op!(ShlAssign, shl_assign, <<);
impl_expr_assign_op!(ShrAssign, shr_assign, >>);

impl Neg for AstNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        AstNode::Neg(Box::new(self))
    }
}

impl Not for AstNode {
    type Output = Self;
    fn not(self) -> Self::Output {
        AstNode::BitNot(Box::new(self))
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

/// Builder for Range nodes with default values
pub struct RangeBuilder {
    counter_name: String,
    start: Box<AstNode>,
    max: Box<AstNode>,
    step: Box<AstNode>,
    body: Box<AstNode>,
    unroll: Option<usize>,
}

impl RangeBuilder {
    /// Create a new RangeBuilder with required fields and default start=0, step=1, unroll=None
    pub fn new(
        counter_name: impl Into<String>,
        max: impl Into<AstNode>,
        body: impl Into<AstNode>,
    ) -> Self {
        Self {
            counter_name: counter_name.into(),
            start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
            max: Box::new(max.into()),
            step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
            body: Box::new(body.into()),
            unroll: None,
        }
    }

    pub fn start(mut self, start: impl Into<AstNode>) -> Self {
        self.start = Box::new(start.into());
        self
    }

    pub fn step(mut self, step: impl Into<AstNode>) -> Self {
        self.step = Box::new(step.into());
        self
    }

    /// Enable full unrolling (#pragma unroll)
    pub fn unroll(mut self) -> Self {
        self.unroll = Some(0);
        self
    }

    /// Enable unrolling with a specific factor (#pragma unroll N)
    pub fn unroll_by(mut self, factor: usize) -> Self {
        self.unroll = Some(factor);
        self
    }

    pub fn build(self) -> AstNode {
        AstNode::Range {
            counter_name: self.counter_name,
            start: self.start,
            max: self.max,
            step: self.step,
            body: self.body,
            unroll: self.unroll,
        }
    }
}

impl AstNode {
    // Convenience constructors for common variants

    /// Create a variable reference
    pub fn var(name: impl Into<String>) -> Self {
        AstNode::Var(name.into())
    }

    /// Create a constant from a literal
    pub fn const_val(val: impl Into<ConstLiteral>) -> Self {
        AstNode::Const(val.into())
    }

    /// Create a capture node for pattern matching
    pub fn capture(n: usize) -> Self {
        AstNode::Capture(n)
    }

    /// Create a random value node
    pub fn rand() -> Self {
        AstNode::Rand
    }

    /// Create a barrier node
    pub fn barrier() -> Self {
        AstNode::Barrier
    }

    /// Create a cast node
    pub fn cast(dtype: DType, expr: impl Into<AstNode>) -> Self {
        AstNode::Cast {
            dtype,
            expr: Box::new(expr.into()),
        }
    }

    /// Create a max node
    pub fn max(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> Self {
        AstNode::Max(Box::new(lhs.into()), Box::new(rhs.into()))
    }

    /// Create an assign node
    pub fn assign(var_name: impl Into<String>, value: impl Into<AstNode>) -> Self {
        AstNode::Assign(var_name.into(), Box::new(value.into()))
    }

    /// Create a store node
    pub fn store(
        target: impl Into<AstNode>,
        index: impl Into<AstNode>,
        value: impl Into<AstNode>,
    ) -> Self {
        AstNode::Store {
            target: Box::new(target.into()),
            index: Box::new(index.into()),
            value: Box::new(value.into()),
        }
    }

    /// Create a deref node
    pub fn deref(expr: impl Into<AstNode>) -> Self {
        AstNode::Deref(Box::new(expr.into()))
    }

    /// Create a drop node
    pub fn drop(var_name: impl Into<String>) -> Self {
        AstNode::Drop(var_name.into())
    }

    /// Create a function call node
    pub fn call(name: impl Into<String>, args: Vec<AstNode>) -> Self {
        AstNode::CallFunction {
            name: name.into(),
            args,
        }
    }

    /// Create a block node
    pub fn block(scope: Scope, statements: Vec<AstNode>) -> Self {
        AstNode::Block { scope, statements }
    }

    /// Create a block node with empty scope
    pub fn block_with_statements(statements: Vec<AstNode>) -> Self {
        AstNode::Block {
            scope: Scope {
                declarations: vec![],
            },
            statements,
        }
    }

    /// Helper to create a Range with default start=0 and step=1
    pub fn range(
        counter_name: impl Into<String>,
        max: impl Into<AstNode>,
        body: impl Into<AstNode>,
    ) -> Self {
        RangeBuilder::new(counter_name, max, body).build()
    }

    /// Create a RangeBuilder for more control
    pub fn range_builder(
        counter_name: impl Into<String>,
        max: impl Into<AstNode>,
        body: impl Into<AstNode>,
    ) -> RangeBuilder {
        RangeBuilder::new(counter_name, max, body)
    }

    pub fn children(&self) -> Vec<&AstNode> {
        match self {
            AstNode::Const(_) => vec![],
            AstNode::Var(_) => vec![],
            AstNode::Cast { expr, .. } => vec![expr.as_ref()],
            AstNode::Add(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Mul(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Max(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Rem(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::BitAnd(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::BitOr(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::BitXor(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Shl(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Shr(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Assign(_, r) => vec![r.as_ref()],
            AstNode::Store {
                target,
                index,
                value,
            } => vec![target.as_ref(), index.as_ref(), value.as_ref()],
            AstNode::Deref(n) => vec![n.as_ref()],
            AstNode::Neg(n) => vec![n.as_ref()],
            AstNode::Recip(n) => vec![n.as_ref()],
            AstNode::Sin(n) => vec![n.as_ref()],
            AstNode::Sqrt(n) => vec![n.as_ref()],
            AstNode::Log2(n) => vec![n.as_ref()],
            AstNode::Exp2(n) => vec![n.as_ref()],
            AstNode::BitNot(n) => vec![n.as_ref()],
            AstNode::CallFunction { args, .. } => args.iter().collect(),
            AstNode::Range {
                start,
                max,
                step,
                body,
                ..
            } => {
                vec![start.as_ref(), max.as_ref(), step.as_ref(), body.as_ref()]
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
            AstNode::Max(_, _) => AstNode::Max(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Rem(_, _) => AstNode::Rem(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::BitAnd(_, _) => AstNode::BitAnd(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::BitOr(_, _) => AstNode::BitOr(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::BitXor(_, _) => AstNode::BitXor(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Shl(_, _) => AstNode::Shl(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Shr(_, _) => AstNode::Shr(
                Box::new(children_iter.next().unwrap()),
                Box::new(children_iter.next().unwrap()),
            ),
            AstNode::Assign(var_name, _) => {
                AstNode::Assign(var_name, Box::new(children_iter.next().unwrap()))
            }
            AstNode::Store { .. } => AstNode::Store {
                target: Box::new(children_iter.next().unwrap()),
                index: Box::new(children_iter.next().unwrap()),
                value: Box::new(children_iter.next().unwrap()),
            },
            AstNode::Deref(_) => AstNode::Deref(Box::new(children_iter.next().unwrap())),
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
            AstNode::BitNot(_) => AstNode::BitNot(Box::new(children_iter.next().unwrap())),
            AstNode::CallFunction { name, .. } => AstNode::CallFunction {
                name,
                args: children_iter.collect(),
            },
            AstNode::Range {
                counter_name,
                unroll,
                ..
            } => AstNode::Range {
                counter_name, // moved
                start: Box::new(children_iter.next().unwrap()),
                max: Box::new(children_iter.next().unwrap()),
                step: Box::new(children_iter.next().unwrap()),
                body: Box::new(children_iter.next().unwrap()),
                unroll,
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
            AstNode::Mul(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Recip(Box::new(AstNode::Const(ConstLiteral::F32(
                    2.0
                )))))
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
    fn test_bitwise_ops() {
        let a = AstNode::Var("a".to_string());
        let b: AstNode = 2isize.into();

        // BitAnd
        let expr = a.clone() & b.clone();
        assert_eq!(
            expr,
            AstNode::BitAnd(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // BitOr
        let expr = a.clone() | b.clone();
        assert_eq!(
            expr,
            AstNode::BitOr(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // BitXor
        let expr = a.clone() ^ b.clone();
        assert_eq!(
            expr,
            AstNode::BitXor(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // Shl
        let expr = a.clone() << b.clone();
        assert_eq!(
            expr,
            AstNode::Shl(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // Shr
        let expr = a.clone() >> b.clone();
        assert_eq!(
            expr,
            AstNode::Shr(
                Box::new(AstNode::Var("a".to_string())),
                Box::new(AstNode::Const(ConstLiteral::Isize(2)))
            )
        );

        // BitNot
        let expr = !a.clone();
        assert_eq!(
            expr,
            AstNode::BitNot(Box::new(AstNode::Var("a".to_string())))
        );
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

    #[test]
    fn test_helper_functions() {
        // Test var
        let v = AstNode::var("x");
        assert_eq!(v, AstNode::Var("x".to_string()));

        // Test const_val
        let c = AstNode::const_val(ConstLiteral::F32(1.0));
        assert_eq!(c, AstNode::Const(ConstLiteral::F32(1.0)));

        // Test assign
        let assign = AstNode::assign("x", 1.0f32);
        assert_eq!(
            assign,
            AstNode::Assign(
                "x".to_string(),
                Box::new(AstNode::Const(ConstLiteral::F32(1.0)))
            )
        );

        // Test max
        let max = AstNode::max(1.0f32, 2.0f32);
        assert_eq!(
            max,
            AstNode::Max(
                Box::new(AstNode::Const(ConstLiteral::F32(1.0))),
                Box::new(AstNode::Const(ConstLiteral::F32(2.0)))
            )
        );

        // Test cast
        let cast = AstNode::cast(DType::F32, 1isize);
        assert_eq!(
            cast,
            AstNode::Cast {
                dtype: DType::F32,
                expr: Box::new(AstNode::Const(ConstLiteral::Isize(1)))
            }
        );
    }

    #[test]
    fn test_range_builder() {
        // Test simple range with defaults
        let r1 = AstNode::range("i", 10isize, AstNode::var("x"));
        assert_eq!(
            r1,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: None,
            }
        );

        // Test range builder with custom start and step
        let r2 = AstNode::range_builder("i", 100isize, AstNode::var("x"))
            .start(5isize)
            .step(2isize)
            .build();
        assert_eq!(
            r2,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(5))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(100))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(2))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: None,
            }
        );

        // Test range builder with full unroll
        let r3 = AstNode::range_builder("i", 10isize, AstNode::var("x"))
            .unroll()
            .build();
        assert_eq!(
            r3,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(10))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: Some(0),
            }
        );

        // Test range builder with specific unroll factor
        let r4 = AstNode::range_builder("i", 100isize, AstNode::var("x"))
            .unroll_by(4)
            .build();
        assert_eq!(
            r4,
            AstNode::Range {
                counter_name: "i".to_string(),
                start: Box::new(AstNode::Const(ConstLiteral::Isize(0))),
                max: Box::new(AstNode::Const(ConstLiteral::Isize(100))),
                step: Box::new(AstNode::Const(ConstLiteral::Isize(1))),
                body: Box::new(AstNode::Var("x".to_string())),
                unroll: Some(4),
            }
        );
    }

    #[test]
    fn test_block_helpers() {
        // Test block_with_statements
        let block = AstNode::block_with_statements(vec![
            AstNode::assign("x", 1.0f32),
            AstNode::assign("y", 2.0f32),
        ]);

        assert!(matches!(block, AstNode::Block { .. }));
        if let AstNode::Block { scope, statements } = block {
            assert_eq!(scope.declarations.len(), 0);
            assert_eq!(statements.len(), 2);
        }
    }

    #[test]
    fn test_other_helpers() {
        // Test store
        let store = AstNode::store(AstNode::var("arr"), 0isize, 1.0f32);
        assert!(matches!(store, AstNode::Store { .. }));

        // Test deref
        let deref = AstNode::deref(AstNode::var("ptr"));
        assert_eq!(
            deref,
            AstNode::Deref(Box::new(AstNode::Var("ptr".to_string())))
        );

        // Test call
        let call = AstNode::call(
            "foo",
            vec![
                AstNode::var("x"),
                AstNode::const_val(ConstLiteral::F32(1.0)),
            ],
        );
        assert!(matches!(call, AstNode::CallFunction { .. }));

        // Test rand, barrier, drop
        assert!(matches!(AstNode::rand(), AstNode::Rand));
        assert!(matches!(AstNode::barrier(), AstNode::Barrier));
        assert_eq!(AstNode::drop("x"), AstNode::Drop("x".to_string()));
    }
}

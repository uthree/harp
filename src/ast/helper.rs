use super::{AstNode, ConstLiteral, DType, RangeBuilder, Scope};

// Convenience constructors for common variants

/// Create a variable reference
pub fn var(name: impl Into<String>) -> AstNode {
    AstNode::Var(name.into())
}

/// Create a constant from a literal
pub fn const_val(val: impl Into<ConstLiteral>) -> AstNode {
    AstNode::Const(val.into())
}

/// Create a capture node for pattern matching
pub fn capture(n: usize) -> AstNode {
    AstNode::Capture(n)
}

/// Create a random value node
pub fn rand() -> AstNode {
    AstNode::Rand
}

/// Create a barrier node
pub fn barrier() -> AstNode {
    AstNode::Barrier
}

/// Create a cast node
pub fn cast(dtype: DType, expr: impl Into<AstNode>) -> AstNode {
    AstNode::Cast {
        dtype,
        expr: Box::new(expr.into()),
    }
}

/// Create an add node
pub fn add(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::Add(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a mul node
pub fn mul(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::Mul(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a rem node
pub fn rem(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::Rem(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a neg node
pub fn neg(expr: impl Into<AstNode>) -> AstNode {
    AstNode::Neg(Box::new(expr.into()))
}

/// Create a recip node
pub fn recip(expr: impl Into<AstNode>) -> AstNode {
    AstNode::Recip(Box::new(expr.into()))
}

/// Create a sin node
pub fn sin(expr: impl Into<AstNode>) -> AstNode {
    AstNode::Sin(Box::new(expr.into()))
}

/// Create a sqrt node
pub fn sqrt(expr: impl Into<AstNode>) -> AstNode {
    AstNode::Sqrt(Box::new(expr.into()))
}

/// Create a log2 node
pub fn log2(expr: impl Into<AstNode>) -> AstNode {
    AstNode::Log2(Box::new(expr.into()))
}

/// Create an exp2 node
pub fn exp2(expr: impl Into<AstNode>) -> AstNode {
    AstNode::Exp2(Box::new(expr.into()))
}

/// Create a max node
pub fn max(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::Max(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a bit_and node
pub fn bit_and(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::BitAnd(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a bit_or node
pub fn bit_or(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::BitOr(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a bit_xor node
pub fn bit_xor(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::BitXor(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a bit_not node
pub fn bit_not(expr: impl Into<AstNode>) -> AstNode {
    AstNode::BitNot(Box::new(expr.into()))
}

/// Create a shl node
pub fn shl(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::Shl(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a shr node
pub fn shr(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::Shr(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a less-than comparison node (returns Bool)
pub fn less_than(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::LessThan(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create an equality comparison node (returns Bool)
pub fn eq(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> AstNode {
    AstNode::Eq(Box::new(lhs.into()), Box::new(rhs.into()))
}

/// Create a select (conditional) node
pub fn select(
    cond: impl Into<AstNode>,
    true_val: impl Into<AstNode>,
    false_val: impl Into<AstNode>,
) -> AstNode {
    AstNode::Select {
        cond: Box::new(cond.into()),
        true_val: Box::new(true_val.into()),
        false_val: Box::new(false_val.into()),
    }
}

/// Create an assign node
pub fn assign(var_name: impl Into<String>, value: impl Into<AstNode>) -> AstNode {
    AstNode::Assign(var_name.into(), Box::new(value.into()))
}

/// Create a store node
pub fn store(
    target: impl Into<AstNode>,
    index: impl Into<AstNode>,
    value: impl Into<AstNode>,
) -> AstNode {
    AstNode::Store {
        target: Box::new(target.into()),
        index: Box::new(index.into()),
        value: Box::new(value.into()),
    }
}

/// Create a deref node
pub fn deref(expr: impl Into<AstNode>) -> AstNode {
    AstNode::Deref(Box::new(expr.into()))
}

/// Create a drop node
pub fn drop(var_name: impl Into<String>) -> AstNode {
    AstNode::Drop(var_name.into())
}

/// Create a function call node
pub fn call(name: impl Into<String>, args: Vec<AstNode>) -> AstNode {
    AstNode::CallFunction {
        name: name.into(),
        args,
    }
}

/// Create a block node
pub fn block(scope: Scope, statements: Vec<AstNode>) -> AstNode {
    AstNode::Block { scope, statements }
}

/// Create a block node with empty scope
pub fn block_with_statements(statements: Vec<AstNode>) -> AstNode {
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
) -> AstNode {
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

/// Create a Function node
pub fn function(
    name: impl Into<String>,
    arguments: Vec<(String, DType)>,
    return_type: DType,
    scope: Scope,
    statements: Vec<AstNode>,
) -> AstNode {
    AstNode::Function {
        name: name.into(),
        scope,
        statements,
        arguments,
        return_type,
    }
}

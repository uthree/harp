use super::{AstNode, ConstLiteral, DType, RangeBuilder, Scope};

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

    /// Create a less-than comparison node (returns Bool)
    pub fn less_than(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> Self {
        AstNode::LessThan(Box::new(lhs.into()), Box::new(rhs.into()))
    }

    /// Create an equality comparison node (returns Bool)
    pub fn eq(lhs: impl Into<AstNode>, rhs: impl Into<AstNode>) -> Self {
        AstNode::Eq(Box::new(lhs.into()), Box::new(rhs.into()))
    }

    /// Create a select (conditional) node
    pub fn select(
        cond: impl Into<AstNode>,
        true_val: impl Into<AstNode>,
        false_val: impl Into<AstNode>,
    ) -> Self {
        AstNode::Select {
            cond: Box::new(cond.into()),
            true_val: Box::new(true_val.into()),
            false_val: Box::new(false_val.into()),
        }
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

    /// Create a Function node
    pub fn function(
        name: impl Into<String>,
        arguments: Vec<(String, DType)>,
        return_type: DType,
        scope: Scope,
        statements: Vec<AstNode>,
    ) -> Self {
        AstNode::Function {
            name: name.into(),
            scope,
            statements,
            arguments,
            return_type,
        }
    }
}

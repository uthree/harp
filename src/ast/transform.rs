use super::AstNode;

impl AstNode {
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
            AstNode::LessThan(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Eq(l, r) => vec![l.as_ref(), r.as_ref()],
            AstNode::Assign(_, r) => vec![r.as_ref()],
            AstNode::Load { target, index, .. } => vec![target.as_ref(), index.as_ref()],
            AstNode::Store {
                target,
                index,
                value,
                ..
            } => vec![target.as_ref(), index.as_ref(), value.as_ref()],
            AstNode::Neg(n) => vec![n.as_ref()],
            AstNode::Recip(n) => vec![n.as_ref()],
            AstNode::Sin(n) => vec![n.as_ref()],
            AstNode::Sqrt(n) => vec![n.as_ref()],
            AstNode::Log2(n) => vec![n.as_ref()],
            AstNode::Exp2(n) => vec![n.as_ref()],
            AstNode::BitNot(n) => vec![n.as_ref()],
            AstNode::Select {
                cond,
                true_val,
                false_val,
            } => vec![cond.as_ref(), true_val.as_ref(), false_val.as_ref()],
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
            AstNode::Function { statements, .. } => statements.iter().collect(),
            AstNode::Program { functions, .. } => functions.iter().collect(),
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
    // Uses macros to reduce repetitive code.
    pub fn replace_children(self, new_children: Vec<AstNode>) -> AstNode {
        let mut children_iter = new_children.into_iter();

        // Macro to handle binary operations (2 children)
        macro_rules! binary_op {
            ($variant:ident) => {
                AstNode::$variant(
                    Box::new(children_iter.next().unwrap()),
                    Box::new(children_iter.next().unwrap()),
                )
            };
        }

        // Macro to handle unary operations (1 child)
        macro_rules! unary_op {
            ($variant:ident) => {
                AstNode::$variant(Box::new(children_iter.next().unwrap()))
            };
        }

        // Macro to handle ternary operations (3 children) with named fields
        macro_rules! ternary_op_named {
            ($variant:ident { $field1:ident, $field2:ident, $field3:ident }) => {
                AstNode::$variant {
                    $field1: Box::new(children_iter.next().unwrap()),
                    $field2: Box::new(children_iter.next().unwrap()),
                    $field3: Box::new(children_iter.next().unwrap()),
                }
            };
        }

        match self {
            // Binary operations
            AstNode::Add(_, _) => binary_op!(Add),
            AstNode::Mul(_, _) => binary_op!(Mul),
            AstNode::Max(_, _) => binary_op!(Max),
            AstNode::Rem(_, _) => binary_op!(Rem),
            AstNode::BitAnd(_, _) => binary_op!(BitAnd),
            AstNode::BitOr(_, _) => binary_op!(BitOr),
            AstNode::BitXor(_, _) => binary_op!(BitXor),
            AstNode::Shl(_, _) => binary_op!(Shl),
            AstNode::Shr(_, _) => binary_op!(Shr),
            AstNode::LessThan(_, _) => binary_op!(LessThan),
            AstNode::Eq(_, _) => binary_op!(Eq),

            // Unary operations
            AstNode::Neg(_) => unary_op!(Neg),
            AstNode::Recip(_) => unary_op!(Recip),
            AstNode::Sin(_) => unary_op!(Sin),
            AstNode::Sqrt(_) => unary_op!(Sqrt),
            AstNode::Log2(_) => unary_op!(Log2),
            AstNode::Exp2(_) => unary_op!(Exp2),
            AstNode::BitNot(_) => unary_op!(BitNot),

            // Binary/Ternary operations with named fields
            AstNode::Load { vector_width, .. } => AstNode::Load {
                target: Box::new(children_iter.next().unwrap()),
                index: Box::new(children_iter.next().unwrap()),
                vector_width,
            },
            AstNode::Store { vector_width, .. } => AstNode::Store {
                target: Box::new(children_iter.next().unwrap()),
                index: Box::new(children_iter.next().unwrap()),
                value: Box::new(children_iter.next().unwrap()),
                vector_width,
            },
            AstNode::Select { .. } => {
                ternary_op_named!(Select {
                    cond,
                    true_val,
                    false_val
                })
            }

            // Special cases
            AstNode::Assign(var_name, _) => {
                AstNode::Assign(var_name, Box::new(children_iter.next().unwrap()))
            }
            AstNode::Cast { dtype, .. } => AstNode::Cast {
                dtype,
                expr: Box::new(children_iter.next().unwrap()),
            },
            AstNode::CallFunction { name, .. } => AstNode::CallFunction {
                name,
                args: children_iter.collect(),
            },
            AstNode::Range {
                counter_name,
                unroll,
                ..
            } => AstNode::Range {
                counter_name,
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
            AstNode::Function {
                name,
                scope,
                arguments,
                return_type,
                ..
            } => {
                let statements = children_iter.collect();
                AstNode::Function {
                    name,
                    scope,
                    arguments,
                    return_type,
                    statements,
                }
            }
            AstNode::Program { entry_point, .. } => {
                let functions = children_iter.collect();
                AstNode::Program {
                    functions,
                    entry_point,
                }
            }
            // Nodes without children are returned as is (moved).
            _ => self,
        }
    }
}

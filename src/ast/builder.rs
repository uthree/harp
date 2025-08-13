use crate::ast::{dtype, node::AstNode, op::AstOp};

impl AstNode {
    // --- AST Construction Helpers ---

    /// Creates a new `FuncDef` node.
    pub fn func_def(name: &str, args: Vec<(String, dtype::DType)>, body: Vec<AstNode>) -> Self {
        Self::new(
            AstOp::Func {
                name: name.to_string(),
                args,
            },
            body,
            dtype::DType::Void,
        )
    }

    /// Creates a new `Call` node.
    pub fn call(name: &str, args: Vec<AstNode>) -> Self {
        Self::new(
            AstOp::Call(name.to_string()),
            args,
            dtype::DType::Any, // Return type is often context-dependent
        )
    }

    /// Creates a new `Block` node.
    pub fn block(body: Vec<AstNode>) -> Self {
        Self::new(AstOp::Block, body, dtype::DType::Void)
    }

    /// Creates a new `Range` (for-loop) node.
    pub fn range(loop_var: String, max: AstNode, mut block: Vec<AstNode>, parallel: bool) -> Self {
        let mut src = vec![max];
        src.append(&mut block);
        Self::new(
            AstOp::Range {
                loop_var,
                step: 1,
                parallel,
            },
            src,
            dtype::DType::Void,
        )
    }

    /// Creates a new `BufferIndex` node.
    pub fn buffer_index(self, index: AstNode) -> Self {
        let ptr_dtype = if let dtype::DType::Ptr(inner) = self.dtype.clone() {
            *inner
        } else {
            dtype::DType::Any // Or panic, depending on strictness
        };
        Self::new(AstOp::BufferIndex, vec![self, index], ptr_dtype)
    }

    /// Creates a new `Deref` node.
    pub fn deref(addr: AstNode) -> Self {
        let dtype = addr.dtype.clone();
        Self::new(AstOp::Deref, vec![addr], dtype)
    }

    /// Creates a new `Store` node.
    pub fn store(dst: AstNode, src: AstNode) -> Self {
        Self::new(AstOp::Store, vec![dst, src], dtype::DType::Void)
    }

    /// Creates a new `Assign` node.
    pub fn assign(dst: AstNode, src: AstNode) -> Self {
        Self::new(AstOp::Assign, vec![dst, src], dtype::DType::Void)
    }

    /// Creates a new `Declare` node.
    pub fn declare(name: String, dtype: dtype::DType, value: AstNode) -> Self {
        Self::new(
            AstOp::Declare { name, dtype },
            vec![value],
            dtype::DType::Void,
        )
    }

    /// Nests a statement inside a series of loops.
    pub fn build_loops(loops: Vec<AstNode>, mut statements: Vec<AstNode>) -> AstNode {
        let final_node = if statements.len() == 1 {
            statements.remove(0)
        } else {
            AstNode::block(statements)
        };

        let mut final_block = final_node;
        for mut loop_node in loops.into_iter().rev() {
            // The loop body is now part of the `src` field, so we need to update it there.
            if let AstOp::Range { .. } = loop_node.op {
                // The first element of src is the range max, the rest is the body.
                let mut new_src = vec![loop_node.src.remove(0)];
                new_src.push(final_block);
                loop_node.src = new_src;
            }
            final_block = loop_node;
        }
        final_block
    }

    /// Creates a new `Malloc` node.
    pub fn malloc(size: AstNode, dtype: dtype::DType) -> Self {
        let ptr_type = dtype::DType::Ptr(Box::new(dtype.clone()));
        Self::new(AstOp::Malloc(dtype), vec![size], ptr_type)
    }

    /// Creates a new `Free` node.
    pub fn free(ptr: AstNode) -> Self {
        Self::new(AstOp::Free, vec![ptr], dtype::DType::Void)
    }
}

// --- Macro implementations for operators ---

macro_rules! impl_unary_op {
    ($op: ident, $fname: ident) => {
        impl AstNode {
            fn $fname(self: Self) -> Self {
                let dtype = &self.dtype;
                if !(dtype.is_real() || dtype.is_integer() || *dtype == dtype::DType::Any) {
                    panic!("Cannot apply {} to {:?}", stringify!($op), self.dtype)
                }
                AstNode::new(AstOp::$op, vec![self.clone()], self.dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self) -> Self {
                let dtype = &self.dtype;
                if !(dtype.is_real() || *dtype == dtype::DType::Any) {
                    panic!("Cannot apply {} to {:?}", stringify!($op), self.dtype)
                }
                AstNode::new(AstOp::$op, vec![self.clone()], self.dtype)
            }
        }
    };
}

impl_unary_op!(Neg, neg_);
impl_unary_op!(pub, Recip, recip);
impl_unary_op!(pub, Sqrt, sqrt);
impl_unary_op!(pub, Sin, sin);
impl_unary_op!(pub, Log2, log2);
impl_unary_op!(pub, Exp2, exp2);

macro_rules! impl_binary_op {
    ($op: ident, $fname: ident) => {
        impl AstNode {
            fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let mut lhs = self;
                let mut rhs = other.into();

                let is_pattern = matches!(lhs.op, AstOp::Capture(_, _))
                    || matches!(rhs.op, AstOp::Capture(_, _));

                if !is_pattern {
                    if lhs.dtype != rhs.dtype {
                        // Attempt to promote types
                        let (l, r) = (&lhs.dtype, &rhs.dtype);
                        if l == &dtype::DType::Any {
                            lhs = lhs.cast(r.clone());
                        } else if r == &dtype::DType::Any {
                            rhs = rhs.cast(l.clone());
                        } else if l.is_real() && r.is_integer() {
                            rhs = rhs.cast(l.clone());
                        } else if l.is_integer() && r.is_real() {
                            lhs = lhs.cast(r.clone());
                        } else if l == &dtype::DType::F32 && r == &dtype::DType::F64 {
                            lhs = lhs.cast(dtype::DType::F64);
                        } else if l == &dtype::DType::F64 && r == &dtype::DType::F32 {
                            rhs = rhs.cast(dtype::DType::F64);
                        } else if l.is_integer() && r.is_integer() {
                            // Promote integer types to the larger one
                            if l.size_in_bytes() > r.size_in_bytes() {
                                rhs = rhs.cast(l.clone());
                            } else if r.size_in_bytes() > l.size_in_bytes() {
                                lhs = lhs.cast(r.clone());
                            }
                        }
                    }

                    if lhs.dtype != rhs.dtype {
                        panic!(
                            "Cannot apply {} to {:?} and {:?}",
                            stringify!($op),
                            lhs.dtype,
                            rhs.dtype
                        );
                    }
                }

                let result_dtype = if lhs.dtype == rhs.dtype {
                    lhs.dtype.clone()
                } else {
                    // Manual common_type logic for patterns
                    let (l, r) = (&lhs.dtype, &rhs.dtype);
                    if l == &dtype::DType::Any {
                        r.clone()
                    } else if r == &dtype::DType::Any {
                        l.clone()
                    } else if l.is_real() && r.is_integer() {
                        l.clone()
                    } else if l.is_integer() && r.is_real() {
                        r.clone()
                    } else if l == &dtype::DType::F32 && r == &dtype::DType::F64 {
                        dtype::DType::F64
                    } else if l == &dtype::DType::F64 && r == &dtype::DType::F32 {
                        dtype::DType::F64
                    } else {
                        // Fallback or panic
                        panic!(
                            "Cannot find common type for pattern with {:?} and {:?}",
                            l, r
                        );
                    }
                };

                AstNode::new(AstOp::$op, vec![lhs, rhs], result_dtype)
            }
        }
    };

    (pub, $op: ident, $fname: ident) => {
        impl AstNode {
            pub fn $fname(self: Self, other: impl Into<AstNode>) -> Self {
                let mut lhs = self;
                let mut rhs = other.into();

                let is_pattern = matches!(lhs.op, AstOp::Capture(_, _))
                    || matches!(rhs.op, AstOp::Capture(_, _));

                if !is_pattern {
                    if lhs.dtype != rhs.dtype {
                        // Attempt to promote types
                        let (l, r) = (&lhs.dtype, &rhs.dtype);
                        if l == &dtype::DType::Any {
                            lhs = lhs.cast(r.clone());
                        } else if r == &dtype::DType::Any {
                            rhs = rhs.cast(l.clone());
                        } else if l.is_real() && r.is_integer() {
                            rhs = rhs.cast(l.clone());
                        } else if l.is_integer() && r.is_real() {
                            lhs = lhs.cast(r.clone());
                        } else if l == &dtype::DType::F32 && r == &dtype::DType::F64 {
                            lhs = lhs.cast(dtype::DType::F64);
                        } else if l == &dtype::DType::F64 && r == &dtype::DType::F32 {
                            rhs = rhs.cast(dtype::DType::F64);
                        } else if l.is_integer() && r.is_integer() {
                            // Promote integer types to the larger one
                            if l.size_in_bytes() > r.size_in_bytes() {
                                rhs = rhs.cast(l.clone());
                            } else if r.size_in_bytes() > l.size_in_bytes() {
                                lhs = lhs.cast(r.clone());
                            }
                        }
                    }

                    if lhs.dtype != rhs.dtype {
                        panic!(
                            "Cannot apply {} to {:?} and {:?}",
                            stringify!($op),
                            lhs.dtype,
                            rhs.dtype
                        );
                    }
                }

                let result_dtype = if lhs.dtype == rhs.dtype {
                    lhs.dtype.clone()
                } else {
                    // Manual common_type logic for patterns
                    let (l, r) = (&lhs.dtype, &rhs.dtype);
                    if l == &dtype::DType::Any {
                        r.clone()
                    } else if r == &dtype::DType::Any {
                        l.clone()
                    } else if l.is_real() && r.is_integer() {
                        l.clone()
                    } else if l.is_integer() && r.is_real() {
                        r.clone()
                    } else if l == &dtype::DType::F32 && r == &dtype::DType::F64 {
                        dtype::DType::F64
                    } else if l == &dtype::DType::F64 && r == &dtype::DType::F32 {
                        dtype::DType::F64
                    } else {
                        // Fallback or panic
                        panic!(
                            "Cannot find common type for pattern with {:?} and {:?}",
                            l, r
                        );
                    }
                };

                AstNode::new(AstOp::$op, vec![lhs, rhs], result_dtype)
            }
        }
    };
}

impl_binary_op!(Add, add_);
impl_binary_op!(Mul, mul_);
impl_binary_op!(pub, Max, max);
impl_binary_op!(Rem, rem_);

// --- Operator trait implementations for AstNode ---

impl<T> std::ops::Add<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn add(self, rhs: T) -> Self::Output {
        self.add_(rhs.into())
    }
}

impl<T> std::ops::Sub<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self::Output {
        self.add_(rhs.into().neg_())
    }
}

impl<T> std::ops::Mul<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        self.mul_(rhs.into())
    }
}

impl<T> std::ops::Div<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        self.mul_(rhs.into().recip())
    }
}

impl<T> std::ops::Rem<T> for AstNode
where
    T: Into<AstNode>,
{
    type Output = Self;
    fn rem(self, rhs: T) -> Self::Output {
        self.rem_(rhs.into())
    }
}

impl std::ops::Neg for AstNode {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.neg_()
    }
}

macro_rules! impl_ast_assign_op {
    ($trait:ident, $fname:ident, $op_trait:ident, $op_fname:ident) => {
        impl<T> std::ops::$trait<T> for AstNode
        where
            T: Into<AstNode>,
        {
            fn $fname(&mut self, rhs: T) {
                *self = std::ops::$op_trait::$op_fname(self.clone(), rhs.into());
            }
        }
    };
}

impl_ast_assign_op!(AddAssign, add_assign, Add, add);
impl_ast_assign_op!(SubAssign, sub_assign, Sub, sub);
impl_ast_assign_op!(MulAssign, mul_assign, Mul, mul);
impl_ast_assign_op!(DivAssign, div_assign, Div, div);
impl_ast_assign_op!(RemAssign, rem_assign, Rem, rem);

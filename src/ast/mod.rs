#[derive(Clone)]
pub enum AstNode {
    // arithmetics
    Const(Literal),
    Add(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Max(Box<AstNode>, Box<AstNode>),
    Rem(Box<AstNode>, Box<AstNode>),
    Idiv(Box<AstNode>, Box<AstNode>),
    Recip(Box<AstNode>),
    Sqrt(Box<AstNode>),
    Log2(Box<AstNode>),
    Exp2(Box<AstNode>),
    Sin(Box<AstNode>),
    Cast(Box<AstNode>, DType),
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Isize,                  // signed integer
    Usize,                  // unsigned integer (for array indexing)
    F32,                    // float
    Ptr(Box<DType>),        // pointer for memory buffer
    Vec(Box<DType>, usize), // fixed size vector for SIMD
    Tuple(Vec<DType>),
    Unknown,
}

#[derive(Debug, Clone)]
pub enum Literal {
    Isize(isize),
    Usize(usize),
    F32(f32),
}

// Conversion from numeric types to Literal
impl From<f32> for Literal {
    fn from(value: f32) -> Self {
        Literal::F32(value)
    }
}

impl From<isize> for Literal {
    fn from(value: isize) -> Self {
        Literal::Isize(value)
    }
}

impl From<usize> for Literal {
    fn from(value: usize) -> Self {
        Literal::Usize(value)
    }
}

impl Literal {
    /// Get the DType of this literal
    pub fn dtype(&self) -> DType {
        match self {
            Literal::F32(_) => DType::F32,
            Literal::Isize(_) => DType::Isize,
            Literal::Usize(_) => DType::Usize,
        }
    }
}

impl AstNode {
    /// Get child nodes of this AST node
    pub fn children(&self) -> Vec<&AstNode> {
        match self {
            AstNode::Const(_) => vec![],
            AstNode::Add(left, right)
            | AstNode::Mul(left, right)
            | AstNode::Max(left, right)
            | AstNode::Rem(left, right)
            | AstNode::Idiv(left, right) => vec![left.as_ref(), right.as_ref()],
            AstNode::Recip(operand)
            | AstNode::Sqrt(operand)
            | AstNode::Log2(operand)
            | AstNode::Exp2(operand)
            | AstNode::Sin(operand)
            | AstNode::Cast(operand, _) => vec![operand.as_ref()],
        }
    }

    /// Recursively infer the type of this AST node by traversing child nodes
    pub fn infer_type(&self) -> DType {
        match self {
            AstNode::Const(lit) => lit.dtype(),
            AstNode::Cast(_, dtype) => dtype.clone(),

            // Binary operations - infer from operands
            AstNode::Add(left, right)
            | AstNode::Mul(left, right)
            | AstNode::Max(left, right)
            | AstNode::Rem(left, right)
            | AstNode::Idiv(left, right) => {
                let left_type = left.infer_type();
                let right_type = right.infer_type();

                // If types match, use that type
                if left_type == right_type {
                    left_type
                } else {
                    // Type mismatch - return Unknown
                    // In a more sophisticated implementation, we might do type promotion here
                    DType::Unknown
                }
            }

            // Unary operations that preserve type
            AstNode::Recip(operand) => operand.infer_type(),

            // Mathematical operations that typically return F32
            AstNode::Sqrt(_) | AstNode::Log2(_) | AstNode::Exp2(_) | AstNode::Sin(_) => DType::F32,
        }
    }
}

// Helper functions for constructing AST nodes
pub mod helper;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_from_f32() {
        let lit: Literal = 3.14f32.into();
        match lit {
            Literal::F32(v) => assert_eq!(v, 3.14),
            _ => panic!("Expected F32 literal"),
        }

        let lit = Literal::from(2.5f32);
        match lit {
            Literal::F32(v) => assert_eq!(v, 2.5),
            _ => panic!("Expected F32 literal"),
        }
    }

    #[test]
    fn test_literal_from_isize() {
        let lit: Literal = 42isize.into();
        match lit {
            Literal::Isize(v) => assert_eq!(v, 42),
            _ => panic!("Expected Isize literal"),
        }

        let lit = Literal::from(-10isize);
        match lit {
            Literal::Isize(v) => assert_eq!(v, -10),
            _ => panic!("Expected Isize literal"),
        }
    }

    #[test]
    fn test_literal_from_usize() {
        let lit: Literal = 100usize.into();
        match lit {
            Literal::Usize(v) => assert_eq!(v, 100),
            _ => panic!("Expected Usize literal"),
        }

        let lit = Literal::from(256usize);
        match lit {
            Literal::Usize(v) => assert_eq!(v, 256),
            _ => panic!("Expected Usize literal"),
        }
    }

    #[test]
    fn test_literal_dtype() {
        let f32_lit = Literal::F32(3.14);
        assert_eq!(f32_lit.dtype(), DType::F32);

        let isize_lit = Literal::Isize(42);
        assert_eq!(isize_lit.dtype(), DType::Isize);

        let usize_lit = Literal::Usize(100);
        assert_eq!(usize_lit.dtype(), DType::Usize);
    }

    #[test]
    fn test_children_const() {
        use crate::ast::helper::*;

        let node = const_f32(3.14);
        let children = node.children();
        assert_eq!(children.len(), 0);
    }

    #[test]
    fn test_children_binary_ops() {
        use crate::ast::helper::*;

        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let node = a + b;
        let children = node.children();
        assert_eq!(children.len(), 2);

        let node = const_isize(3) * const_isize(4);
        let children = node.children();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_children_unary_ops() {
        use crate::ast::helper::*;

        let node = sqrt(const_f32(4.0));
        let children = node.children();
        assert_eq!(children.len(), 1);

        let node = sin(const_f32(1.0));
        let children = node.children();
        assert_eq!(children.len(), 1);

        let node = recip(const_f32(2.0));
        let children = node.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_children_cast() {
        use crate::ast::helper::*;

        let node = cast(const_f32(3.14), DType::Isize);
        let children = node.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_children_composite() {
        use crate::ast::helper::*;

        // (a + b) * c
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let c = const_f32(3.0);
        let product = (a + b) * c;

        let children = product.children();
        assert_eq!(children.len(), 2);

        // The first child should be the sum node
        let sum_children = children[0].children();
        assert_eq!(sum_children.len(), 2);
    }

    #[test]
    fn test_infer_type_const() {
        use crate::ast::helper::*;

        let node = const_f32(3.14);
        assert_eq!(node.infer_type(), DType::F32);

        let node = const_isize(42);
        assert_eq!(node.infer_type(), DType::Isize);

        let node = const_usize(100);
        assert_eq!(node.infer_type(), DType::Usize);
    }

    #[test]
    fn test_infer_type_binary_ops() {
        use crate::ast::helper::*;

        // Same types should return that type
        let node = const_f32(1.0) + const_f32(2.0);
        assert_eq!(node.infer_type(), DType::F32);

        let node = const_isize(3) * const_isize(4);
        assert_eq!(node.infer_type(), DType::Isize);

        // Mixed types should return Unknown
        let node = const_f32(1.0) + const_isize(2);
        assert_eq!(node.infer_type(), DType::Unknown);
    }

    #[test]
    fn test_infer_type_unary_ops() {
        use crate::ast::helper::*;

        // Recip preserves type
        let node = recip(const_f32(2.0));
        assert_eq!(node.infer_type(), DType::F32);

        // Math operations return F32
        let node = sqrt(const_f32(4.0));
        assert_eq!(node.infer_type(), DType::F32);

        let node = sin(const_f32(1.0));
        assert_eq!(node.infer_type(), DType::F32);

        let node = log2(const_f32(8.0));
        assert_eq!(node.infer_type(), DType::F32);

        let node = exp2(const_f32(3.0));
        assert_eq!(node.infer_type(), DType::F32);
    }

    #[test]
    fn test_infer_type_cast() {
        use crate::ast::helper::*;

        let node = cast(const_f32(3.14), DType::Isize);
        assert_eq!(node.infer_type(), DType::Isize);

        let node = cast(const_isize(42), DType::F32);
        assert_eq!(node.infer_type(), DType::F32);
    }

    #[test]
    fn test_infer_type_composite() {
        use crate::ast::helper::*;

        // (a + b) * c where all are F32
        let a = const_f32(1.0);
        let b = const_f32(2.0);
        let c = const_f32(3.0);
        let expr = (a + b) * c;
        assert_eq!(expr.infer_type(), DType::F32);

        // sqrt(a + b) where a, b are F32
        let a = const_f32(4.0);
        let b = const_f32(5.0);
        let expr = sqrt(a + b);
        assert_eq!(expr.infer_type(), DType::F32);

        // Complex expression with cast
        let a = const_isize(10);
        let b = const_isize(20);
        let casted = cast(a + b, DType::F32);
        let result = sqrt(casted);
        assert_eq!(result.infer_type(), DType::F32);
    }
}

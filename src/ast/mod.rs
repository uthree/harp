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

// Operator overloading for AstNode
pub mod ops;

// Helper functions for constructing AST nodes
pub mod helper;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;

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
        let node = AstNode::Const(3.14f32.into());
        let children = node.children();
        assert_eq!(children.len(), 0);
    }

    #[test]
    fn test_children_binary_ops() {
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let node = a + b;
        let children = node.children();
        assert_eq!(children.len(), 2);

        let node = AstNode::Const(3isize.into()) * AstNode::Const(4isize.into());
        let children = node.children();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_children_unary_ops() {
        let node = sqrt(AstNode::Const(4.0f32.into()));
        let children = node.children();
        assert_eq!(children.len(), 1);

        let node = sin(AstNode::Const(1.0f32.into()));
        let children = node.children();
        assert_eq!(children.len(), 1);

        let node = recip(AstNode::Const(2.0f32.into()));
        let children = node.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_children_cast() {
        let node = cast(AstNode::Const(3.14f32.into()), DType::Isize);
        let children = node.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_children_composite() {
        // (a + b) * c
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let c = AstNode::Const(3.0f32.into());
        let product = (a + b) * c;

        let children = product.children();
        assert_eq!(children.len(), 2);

        // The first child should be the sum node
        let sum_children = children[0].children();
        assert_eq!(sum_children.len(), 2);
    }

    #[test]
    fn test_infer_type_const() {
        let node = AstNode::Const(3.14f32.into());
        assert_eq!(node.infer_type(), DType::F32);

        let node = AstNode::Const(42isize.into());
        assert_eq!(node.infer_type(), DType::Isize);

        let node = AstNode::Const(100usize.into());
        assert_eq!(node.infer_type(), DType::Usize);
    }

    #[test]
    fn test_infer_type_binary_ops() {
        // Same types should return that type
        let node = AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into());
        assert_eq!(node.infer_type(), DType::F32);

        let node = AstNode::Const(3isize.into()) * AstNode::Const(4isize.into());
        assert_eq!(node.infer_type(), DType::Isize);

        // Mixed types should return Unknown
        let node = AstNode::Const(1.0f32.into()) + AstNode::Const(2isize.into());
        assert_eq!(node.infer_type(), DType::Unknown);
    }

    #[test]
    fn test_infer_type_unary_ops() {
        // Recip preserves type
        let node = recip(AstNode::Const(2.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        // Math operations return F32
        let node = sqrt(AstNode::Const(4.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        let node = sin(AstNode::Const(1.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        let node = log2(AstNode::Const(8.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        let node = exp2(AstNode::Const(3.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);
    }

    #[test]
    fn test_infer_type_cast() {
        let node = cast(AstNode::Const(3.14f32.into()), DType::Isize);
        assert_eq!(node.infer_type(), DType::Isize);

        let node = cast(AstNode::Const(42isize.into()), DType::F32);
        assert_eq!(node.infer_type(), DType::F32);
    }

    #[test]
    fn test_infer_type_composite() {
        // (a + b) * c where all are F32
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let c = AstNode::Const(3.0f32.into());
        let expr = (a + b) * c;
        assert_eq!(expr.infer_type(), DType::F32);

        // sqrt(a + b) where a, b are F32
        let a = AstNode::Const(4.0f32.into());
        let b = AstNode::Const(5.0f32.into());
        let expr = sqrt(a + b);
        assert_eq!(expr.infer_type(), DType::F32);

        // Complex expression with cast
        let a = AstNode::Const(10isize.into());
        let b = AstNode::Const(20isize.into());
        let casted = cast(a + b, DType::F32);
        let result = sqrt(casted);
        assert_eq!(result.infer_type(), DType::F32);
    }
}

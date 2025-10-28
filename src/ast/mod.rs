// Operator overloading for AstNode
pub mod ops;
// Helper functions for constructing AST nodes
pub mod helper;
pub mod pat;

#[derive(Clone, Debug, PartialEq)]
pub enum AstNode {
    // Pattern matching wildcard - パターンマッチング用ワイルドカード
    Wildcard(String),

    // arithmetics - 算術演算
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
    CallFunction,

    // Variables - 変数
    Var(String),

    // Memory operations - メモリ操作（バッファー用）
    Load {
        ptr: Box<AstNode>,    // ポインタ（Ptr<T>型の式）
        offset: Box<AstNode>, // オフセット（Usize型の式）
        count: usize,         // 読み込む要素数（コンパイル時定数、1ならスカラー）
    },
    Store {
        ptr: Box<AstNode>,    // ポインタ（Ptr<T>型の式）
        offset: Box<AstNode>, // オフセット（Usize型の式）
        value: Box<AstNode>,  // 書き込む値（スカラーまたはVec型）
    },

    // Assignment - 変数への代入（スタック/レジスタ用）
    Assign {
        var: String,         // 変数名
        value: Box<AstNode>, // 代入する値
    },
    // TODO: 制御構文
    Range {
        start: Box<AstNode>,
        step: Box<AstNode>,
        stop: Box<AstNode>,
        body: Vec<AstNode>,
    },

    // TODO: 関数と呼び出し
    Program {},

    Function {},
}

pub struct Scope {}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Isize,                  // signed integer
    Usize,                  // unsigned integer (for array indexing)
    F32,                    // float
    Ptr(Box<DType>),        // pointer for memory buffer, 値を渡す時は参照を渡す。
    Vec(Box<DType>, usize), // fixed size vector for SIMD, 値は渡す時にコピーされる
    Tuple(Vec<DType>),
    Unknown,
    // TODO: boolなどの追加
    // TODO: 将来的にf16とか対応させたい
}

#[derive(Debug, Clone, PartialEq)]
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

impl DType {
    /// Convert this type to a vector type with the given size
    pub fn to_vec(&self, size: usize) -> DType {
        DType::Vec(Box::new(self.clone()), size)
    }

    /// Convert this type to a pointer type
    pub fn to_ptr(&self) -> DType {
        DType::Ptr(Box::new(self.clone()))
    }

    /// If this is a Vec type, return the element type and size
    /// Returns None if this is not a Vec type
    pub fn from_vec(&self) -> Option<(&DType, usize)> {
        match self {
            DType::Vec(elem_type, size) => Some((elem_type.as_ref(), *size)),
            _ => None,
        }
    }

    /// If this is a Ptr type, return the pointee type
    /// Returns None if this is not a Ptr type
    pub fn from_ptr(&self) -> Option<&DType> {
        match self {
            DType::Ptr(pointee) => Some(pointee.as_ref()),
            _ => None,
        }
    }

    /// Get the element type if this is a Vec, otherwise return self
    pub fn element_type(&self) -> &DType {
        match self {
            DType::Vec(elem_type, _) => elem_type.as_ref(),
            _ => self,
        }
    }

    /// Get the pointee type if this is a Ptr, otherwise return self
    pub fn deref_type(&self) -> &DType {
        match self {
            DType::Ptr(pointee) => pointee.as_ref(),
            _ => self,
        }
    }

    /// Check if this is a Vec type
    pub fn is_vec(&self) -> bool {
        matches!(self, DType::Vec(_, _))
    }

    /// Check if this is a Ptr type
    pub fn is_ptr(&self) -> bool {
        matches!(self, DType::Ptr(_))
    }
}

impl AstNode {
    /// Get child nodes of this AST node
    pub fn children(&self) -> Vec<&AstNode> {
        match self {
            AstNode::Wildcard(_) | AstNode::Const(_) | AstNode::Var(_) => vec![],
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
            AstNode::Load { ptr, offset, .. } => vec![ptr.as_ref(), offset.as_ref()],
            AstNode::Store { ptr, offset, value } => {
                vec![ptr.as_ref(), offset.as_ref(), value.as_ref()]
            }
            AstNode::Assign { value, .. } => vec![value.as_ref()],
            // 未実装のノード
            AstNode::CallFunction
            | AstNode::Range { .. }
            | AstNode::Program {}
            | AstNode::Function {} => {
                todo!("Not yet implemented")
            }
        }
    }

    /// Apply a function to all child nodes and construct a new node with the results
    /// This is useful for recursive transformations of the AST
    pub fn map_children<F>(&self, f: &F) -> Self
    where
        F: Fn(&AstNode) -> AstNode,
    {
        match self {
            AstNode::Wildcard(_) | AstNode::Const(_) | AstNode::Var(_) => self.clone(),
            AstNode::Add(left, right) => AstNode::Add(Box::new(f(left)), Box::new(f(right))),
            AstNode::Mul(left, right) => AstNode::Mul(Box::new(f(left)), Box::new(f(right))),
            AstNode::Max(left, right) => AstNode::Max(Box::new(f(left)), Box::new(f(right))),
            AstNode::Rem(left, right) => AstNode::Rem(Box::new(f(left)), Box::new(f(right))),
            AstNode::Idiv(left, right) => AstNode::Idiv(Box::new(f(left)), Box::new(f(right))),
            AstNode::Recip(operand) => AstNode::Recip(Box::new(f(operand))),
            AstNode::Sqrt(operand) => AstNode::Sqrt(Box::new(f(operand))),
            AstNode::Log2(operand) => AstNode::Log2(Box::new(f(operand))),
            AstNode::Exp2(operand) => AstNode::Exp2(Box::new(f(operand))),
            AstNode::Sin(operand) => AstNode::Sin(Box::new(f(operand))),
            AstNode::Cast(operand, dtype) => AstNode::Cast(Box::new(f(operand)), dtype.clone()),
            AstNode::Load { ptr, offset, count } => AstNode::Load {
                ptr: Box::new(f(ptr)),
                offset: Box::new(f(offset)),
                count: *count,
            },
            AstNode::Store { ptr, offset, value } => AstNode::Store {
                ptr: Box::new(f(ptr)),
                offset: Box::new(f(offset)),
                value: Box::new(f(value)),
            },
            AstNode::Assign { var, value } => AstNode::Assign {
                var: var.clone(),
                value: Box::new(f(value)),
            },
            // 未実装のノード
            AstNode::CallFunction
            | AstNode::Range { .. }
            | AstNode::Program {}
            | AstNode::Function {} => {
                todo!("Not yet implemented")
            }
        }
    }

    /// Recursively infer the type of this AST node by traversing child nodes
    pub fn infer_type(&self) -> DType {
        match self {
            AstNode::Wildcard(_) => DType::Unknown,
            AstNode::Const(lit) => lit.dtype(),
            AstNode::Cast(_, dtype) => dtype.clone(),
            AstNode::Var(_) => DType::Unknown, // 変数の型はコンテキストに依存

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

            // Memory operations
            AstNode::Load { ptr, count, .. } => {
                let ptr_type = ptr.infer_type();
                let pointee_type = ptr_type.deref_type().clone();
                if *count == 1 {
                    pointee_type // スカラー
                } else {
                    pointee_type.to_vec(*count) // Vec型
                }
            }
            AstNode::Store { .. } => DType::Tuple(vec![]), // Storeは値を返さない（unit型）

            // Assignment
            AstNode::Assign { value, .. } => value.infer_type(), // 代入された値の型を返す

            // 未実装のノード
            AstNode::CallFunction
            | AstNode::Range { .. }
            | AstNode::Program {}
            | AstNode::Function {} => {
                todo!("Not yet implemented")
            }
        }
    }
}

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

    #[test]
    fn test_dtype_to_vec() {
        let base_type = DType::F32;
        let vec_type = base_type.to_vec(4);

        match vec_type {
            DType::Vec(elem_type, size) => {
                assert_eq!(*elem_type, DType::F32);
                assert_eq!(size, 4);
            }
            _ => panic!("Expected Vec type"),
        }
    }

    #[test]
    fn test_dtype_from_vec() {
        let vec_type = DType::F32.to_vec(8);

        let result = vec_type.from_vec();
        assert!(result.is_some());

        let (elem_type, size) = result.unwrap();
        assert_eq!(elem_type, &DType::F32);
        assert_eq!(size, 8);

        // Non-vec type should return None
        let scalar = DType::F32;
        assert!(scalar.from_vec().is_none());
    }

    #[test]
    fn test_dtype_to_ptr() {
        let base_type = DType::F32;
        let ptr_type = base_type.to_ptr();

        match ptr_type {
            DType::Ptr(pointee) => {
                assert_eq!(*pointee, DType::F32);
            }
            _ => panic!("Expected Ptr type"),
        }
    }

    #[test]
    fn test_dtype_from_ptr() {
        let ptr_type = DType::F32.to_ptr();

        let result = ptr_type.from_ptr();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), &DType::F32);

        // Non-ptr type should return None
        let scalar = DType::F32;
        assert!(scalar.from_ptr().is_none());
    }

    #[test]
    fn test_dtype_element_type() {
        // Vec should return element type
        let vec_type = DType::F32.to_vec(4);
        assert_eq!(vec_type.element_type(), &DType::F32);

        // Non-vec should return self
        let scalar = DType::Isize;
        assert_eq!(scalar.element_type(), &DType::Isize);
    }

    #[test]
    fn test_dtype_deref_type() {
        // Ptr should return pointee type
        let ptr_type = DType::F32.to_ptr();
        assert_eq!(ptr_type.deref_type(), &DType::F32);

        // Non-ptr should return self
        let scalar = DType::Isize;
        assert_eq!(scalar.deref_type(), &DType::Isize);
    }

    #[test]
    fn test_dtype_is_vec() {
        let vec_type = DType::F32.to_vec(4);
        assert!(vec_type.is_vec());

        let scalar = DType::F32;
        assert!(!scalar.is_vec());
    }

    #[test]
    fn test_dtype_is_ptr() {
        let ptr_type = DType::F32.to_ptr();
        assert!(ptr_type.is_ptr());

        let scalar = DType::F32;
        assert!(!scalar.is_ptr());
    }

    #[test]
    fn test_dtype_nested_types() {
        // Vec of Ptr
        let ptr_type = DType::F32.to_ptr();
        let vec_of_ptr = ptr_type.to_vec(4);

        assert!(vec_of_ptr.is_vec());
        let (elem, size) = vec_of_ptr.from_vec().unwrap();
        assert_eq!(size, 4);
        assert!(elem.is_ptr());

        // Ptr to Vec
        let vec_type = DType::F32.to_vec(8);
        let ptr_to_vec = vec_type.to_ptr();

        assert!(ptr_to_vec.is_ptr());
        let pointee = ptr_to_vec.from_ptr().unwrap();
        assert!(pointee.is_vec());
    }

    #[test]
    fn test_var_node() {
        let var = AstNode::Var("x".to_string());
        assert_eq!(var.children().len(), 0);
        assert_eq!(var.infer_type(), DType::Unknown);
    }

    #[test]
    fn test_load_scalar() {
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Var("input0".to_string())),
            offset: Box::new(AstNode::Const(0usize.into())),
            count: 1,
        };

        // children should include ptr and offset
        let children = load.children();
        assert_eq!(children.len(), 2);

        // Type inference: Var returns Unknown, so deref_type returns Unknown
        // This test demonstrates the structure, actual type depends on context
        let inferred = load.infer_type();
        assert_eq!(inferred, DType::Unknown);
    }

    #[test]
    fn test_load_vector() {
        // Create a proper pointer type for testing
        let ptr_node = AstNode::Cast(
            Box::new(AstNode::Var("buffer".to_string())),
            DType::F32.to_ptr(),
        );

        let load = AstNode::Load {
            ptr: Box::new(ptr_node),
            offset: Box::new(AstNode::Const(0usize.into())),
            count: 4,
        };

        // Type should be Vec<F32, 4>
        let inferred = load.infer_type();
        assert_eq!(inferred, DType::F32.to_vec(4));

        // children should include ptr and offset
        let children = load.children();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_store() {
        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var("output0".to_string())),
            offset: Box::new(AstNode::Const(0usize.into())),
            value: Box::new(AstNode::Const(3.14f32.into())),
        };

        // children should include ptr, offset, and value
        let children = store.children();
        assert_eq!(children.len(), 3);

        // Store returns unit type (empty tuple)
        let inferred = store.infer_type();
        assert_eq!(inferred, DType::Tuple(vec![]));
    }

    #[test]
    fn test_assign() {
        let assign = AstNode::Assign {
            var: "alu0".to_string(),
            value: Box::new(AstNode::Const(42isize.into())),
        };

        // children should include only value
        let children = assign.children();
        assert_eq!(children.len(), 1);

        // Assign returns the type of the value
        let inferred = assign.infer_type();
        assert_eq!(inferred, DType::Isize);
    }

    #[test]
    fn test_load_store_map_children() {
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Const(1isize.into())),
            offset: Box::new(AstNode::Const(2isize.into())),
            count: 4,
        };

        // Map children: multiply each constant by 2
        let mapped = load.map_children(&|node| match node {
            AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n * 2)),
            _ => node.clone(),
        });

        if let AstNode::Load { ptr, offset, count } = mapped {
            assert_eq!(*ptr, AstNode::Const(Literal::Isize(2)));
            assert_eq!(*offset, AstNode::Const(Literal::Isize(4)));
            assert_eq!(count, 4);
        } else {
            panic!("Expected Load node");
        }
    }

    #[test]
    fn test_assign_map_children() {
        let assign = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Const(10isize.into())),
        };

        // Map children: increment constant
        let mapped = assign.map_children(&|node| match node {
            AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n + 1)),
            _ => node.clone(),
        });

        if let AstNode::Assign { var, value } = mapped {
            assert_eq!(var, "x");
            assert_eq!(*value, AstNode::Const(Literal::Isize(11)));
        } else {
            panic!("Expected Assign node");
        }
    }
}

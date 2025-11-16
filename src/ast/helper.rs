use super::{AstNode, DType, Literal, Scope};

// Convenience free functions for AST construction

/// Macro to generate binary operation helper functions
macro_rules! impl_binary_helper {
    ($fn_name:ident, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name(a: AstNode, b: AstNode) -> AstNode {
            AstNode::$variant(Box::new(a), Box::new(b))
        }
    };
}

/// Macro to generate unary operation helper functions
macro_rules! impl_unary_helper {
    ($fn_name:ident, $variant:ident, $doc:expr) => {
        #[doc = $doc]
        pub fn $fn_name(a: AstNode) -> AstNode {
            AstNode::$variant(Box::new(a))
        }
    };
}

// Binary operation helpers
impl_binary_helper!(max, Max, "Create a max node: max(a, b)");
impl_binary_helper!(idiv, Idiv, "Create an integer division node: a / b");
impl_binary_helper!(rem, Rem, "Create a remainder node: a % b");

// Unary operation helpers
impl_unary_helper!(recip, Recip, "Create a reciprocal node: 1 / a");
impl_unary_helper!(sqrt, Sqrt, "Create a square root node: sqrt(a)");
impl_unary_helper!(log2, Log2, "Create a log2 node: log2(a)");
impl_unary_helper!(exp2, Exp2, "Create an exp2 node: 2^a");
impl_unary_helper!(sin, Sin, "Create a sine node: sin(a)");

/// Create a cast node: cast a to dtype
pub fn cast(a: AstNode, dtype: DType) -> AstNode {
    AstNode::Cast(Box::new(a), dtype)
}

/// Create a variable reference node
pub fn var(name: impl Into<String>) -> AstNode {
    AstNode::Var(name.into())
}

/// Create a wildcard node for pattern matching or input capture
pub fn wildcard(name: impl Into<String>) -> AstNode {
    AstNode::Wildcard(name.into())
}

/// Create a load node for scalar memory access
pub fn load(ptr: AstNode, offset: AstNode, dtype: DType) -> AstNode {
    AstNode::Load {
        ptr: Box::new(ptr),
        offset: Box::new(offset),
        count: 1,
        dtype,
    }
}

/// Create a load node for vector memory access (SIMD)
pub fn load_vec(ptr: AstNode, offset: AstNode, count: usize, dtype: DType) -> AstNode {
    AstNode::Load {
        ptr: Box::new(ptr),
        offset: Box::new(offset),
        count,
        dtype,
    }
}

/// Create a store node
pub fn store(ptr: AstNode, offset: AstNode, value: AstNode) -> AstNode {
    AstNode::Store {
        ptr: Box::new(ptr),
        offset: Box::new(offset),
        value: Box::new(value),
    }
}

/// Create an assignment node
pub fn assign(var: impl Into<String>, value: AstNode) -> AstNode {
    AstNode::Assign {
        var: var.into(),
        value: Box::new(value),
    }
}

/// Create a barrier node for synchronization in parallel execution
pub fn barrier() -> AstNode {
    AstNode::Barrier
}

/// Broadcast a scalar value to a vector type (for SIMD)
///
/// This creates a cast that converts a scalar to a vector where all elements have the same value.
/// Example: broadcast(2.0, 4) creates float4(2.0, 2.0, 2.0, 2.0)
pub fn broadcast(value: AstNode, width: usize) -> AstNode {
    let value_type = value.infer_type();
    let vec_type = value_type.to_vec(width);
    AstNode::Cast(Box::new(value), vec_type)
}

/// Create a function node
///
/// # Arguments
/// * `name` - Function name (can be None for anonymous functions)
/// * `kind` - Function kind (Normal or Kernel)
/// * `params` - Parameter declarations
/// * `return_type` - Return type
/// * `body` - Function body (typically a Block node)
pub fn function(
    name: Option<impl Into<String>>,
    kind: super::FunctionKind,
    params: Vec<super::VarDecl>,
    return_type: DType,
    body: AstNode,
) -> AstNode {
    AstNode::Function {
        name: name.map(|n| n.into()),
        params,
        return_type,
        body: Box::new(body),
        kind,
    }
}

/// Create a program node
///
/// # Arguments
/// * `functions` - List of AstNode::Function
/// * `entry_point` - Name of the entry point function
pub fn program(functions: Vec<AstNode>, entry_point: impl Into<String>) -> AstNode {
    AstNode::Program {
        functions,
        entry_point: entry_point.into(),
    }
}

/// Create a range (for loop) node
///
/// # Arguments
/// * `var` - Loop variable name
/// * `start` - Start value
/// * `step` - Step value
/// * `stop` - Stop value (exclusive)
/// * `body` - Loop body
pub fn range(
    var: impl Into<String>,
    start: AstNode,
    step: AstNode,
    stop: AstNode,
    body: AstNode,
) -> AstNode {
    AstNode::Range {
        var: var.into(),
        start: Box::new(start),
        step: Box::new(step),
        stop: Box::new(stop),
        body: Box::new(body),
    }
}

/// Create a block node with statements and scope
///
/// # Arguments
/// * `statements` - List of statements in the block
/// * `scope` - Scope for the block
pub fn block(statements: Vec<AstNode>, scope: Scope) -> AstNode {
    AstNode::Block {
        statements,
        scope: Box::new(scope),
    }
}

/// Create an empty block
pub fn empty_block() -> AstNode {
    AstNode::Block {
        statements: vec![],
        scope: Box::new(Scope::new()),
    }
}

/// Create an integer constant node
pub fn const_int(value: isize) -> AstNode {
    AstNode::Const(Literal::Int(value))
}

/// Create a float constant node (f32)
pub fn const_f32(value: f32) -> AstNode {
    AstNode::Const(Literal::F32(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Literal;

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_const_creation() {
        // Test constant creation using Into
        let f32_node = AstNode::Const(3.14f32.into());
        match f32_node {
            AstNode::Const(Literal::F32(v)) => assert_eq!(v, 3.14),
            _ => panic!("Expected F32 constant"),
        }

        let isize_node = AstNode::Const(42isize.into());
        match isize_node {
            AstNode::Const(Literal::Int(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Isize constant"),
        }

        let usize_node = AstNode::Const(100usize.into());
        match usize_node {
            AstNode::Const(Literal::Int(v)) => assert_eq!(v, 100),
            _ => panic!("Expected Usize constant"),
        }
    }

    #[test]
    fn test_binary_ops() {
        // Test binary operation using operator overloading
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());

        let add_node = a.clone() + b.clone();
        match add_node {
            AstNode::Add(left, right) => match (*left, *right) {
                (AstNode::Const(Literal::F32(l)), AstNode::Const(Literal::F32(r))) => {
                    assert_eq!(l, 1.0);
                    assert_eq!(r, 2.0);
                }
                _ => panic!("Expected F32 constants in Add node"),
            },
            _ => panic!("Expected Add node"),
        }

        let mul_node = a.clone() * b.clone();
        match mul_node {
            AstNode::Mul(_, _) => {}
            _ => panic!("Expected Mul node"),
        }

        let max_node = max(a.clone(), b.clone());
        match max_node {
            AstNode::Max(_, _) => {}
            _ => panic!("Expected Max node"),
        }

        let rem_node = a.clone() % b.clone();
        match rem_node {
            AstNode::Rem(_, _) => {}
            _ => panic!("Expected Rem node"),
        }

        let idiv_node = idiv(a.clone(), b.clone());
        match idiv_node {
            AstNode::Idiv(_, _) => {}
            _ => panic!("Expected Idiv node"),
        }
    }

    #[test]
    fn test_unary_ops() {
        // Test unary operation helpers
        let a = AstNode::Const(4.0f32.into());

        let recip_node = recip(a.clone());
        match recip_node {
            AstNode::Recip(_) => {}
            _ => panic!("Expected Recip node"),
        }

        let sqrt_node = sqrt(a.clone());
        match sqrt_node {
            AstNode::Sqrt(_) => {}
            _ => panic!("Expected Sqrt node"),
        }

        let log2_node = log2(a.clone());
        match log2_node {
            AstNode::Log2(_) => {}
            _ => panic!("Expected Log2 node"),
        }

        let exp2_node = exp2(a.clone());
        match exp2_node {
            AstNode::Exp2(_) => {}
            _ => panic!("Expected Exp2 node"),
        }

        let sin_node = sin(a.clone());
        match sin_node {
            AstNode::Sin(_) => {}
            _ => panic!("Expected Sin node"),
        }
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_cast() {
        let a = AstNode::Const(3.14f32.into());
        let cast_node = cast(a, DType::Int);
        match cast_node {
            AstNode::Cast(_, dtype) => match dtype {
                DType::Int => {}
                _ => panic!("Expected Int dtype"),
            },
            _ => panic!("Expected Cast node"),
        }
    }

    #[test]
    fn test_composite_expression() {
        // Test building a composite expression: (a + b) * c using operator overloading
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let c = AstNode::Const(3.0f32.into());

        let product = (a + b) * c;

        match product {
            AstNode::Mul(left, right) => match (*left, *right) {
                (AstNode::Add(_, _), AstNode::Const(Literal::F32(v))) => {
                    assert_eq!(v, 3.0);
                }
                _ => panic!("Expected Add node and F32 constant"),
            },
            _ => panic!("Expected Mul node"),
        }
    }

    #[test]
    fn test_var_helper() {
        let var_node = var("x");
        match var_node {
            AstNode::Var(name) => assert_eq!(name, "x"),
            _ => panic!("Expected Var node"),
        }

        // Test with String
        let var_node2 = var("buffer".to_string());
        match var_node2 {
            AstNode::Var(name) => assert_eq!(name, "buffer"),
            _ => panic!("Expected Var node"),
        }
    }

    #[test]
    fn test_load_helper() {
        let ptr = var("input0");
        let offset = AstNode::Const(0usize.into());
        let load_node = load(ptr, offset, DType::F32);

        match load_node {
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype: _,
            } => {
                assert_eq!(count, 1);
                match *ptr {
                    AstNode::Var(name) => assert_eq!(name, "input0"),
                    _ => panic!("Expected Var node for ptr"),
                }
                match *offset {
                    AstNode::Const(Literal::Int(v)) => assert_eq!(v, 0),
                    _ => panic!("Expected Usize constant for offset"),
                }
            }
            _ => panic!("Expected Load node"),
        }
    }

    #[test]
    fn test_load_vec_helper() {
        let ptr = cast(var("buffer"), DType::F32.to_ptr());
        let offset = var("i");
        let load_node = load_vec(ptr, offset, 4, DType::F32.to_vec(4));

        match load_node {
            AstNode::Load { count, .. } => {
                assert_eq!(count, 4);
            }
            _ => panic!("Expected Load node"),
        }

        // Test type inference
        let inferred = load_node.infer_type();
        assert_eq!(inferred, DType::F32.to_vec(4));
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_store_helper() {
        let ptr = var("output0");
        let offset = AstNode::Const(0usize.into());
        let value = AstNode::Const(3.14f32.into());
        let store_node = store(ptr, offset, value);

        match store_node {
            AstNode::Store { ptr, offset, value } => {
                match *ptr {
                    AstNode::Var(name) => assert_eq!(name, "output0"),
                    _ => panic!("Expected Var node for ptr"),
                }
                match *offset {
                    AstNode::Const(Literal::Int(v)) => assert_eq!(v, 0),
                    _ => panic!("Expected Usize constant for offset"),
                }
                match *value {
                    AstNode::Const(Literal::F32(v)) => assert_eq!(v, 3.14),
                    _ => panic!("Expected F32 constant for value"),
                }
            }
            _ => panic!("Expected Store node"),
        }
    }

    #[test]
    fn test_assign_helper() {
        let value = AstNode::Const(42isize.into());
        let assign_node = assign("alu0", value);

        match assign_node {
            AstNode::Assign { var, value } => {
                assert_eq!(var, "alu0");
                match *value {
                    AstNode::Const(Literal::Int(v)) => assert_eq!(v, 42),
                    _ => panic!("Expected Isize constant for value"),
                }
            }
            _ => panic!("Expected Assign node"),
        }

        // Test with String
        let assign_node2 = assign("temp".to_string(), AstNode::Const(10isize.into()));
        match assign_node2 {
            AstNode::Assign { var, .. } => assert_eq!(var, "temp"),
            _ => panic!("Expected Assign node"),
        }
    }

    #[test]
    fn test_memory_operation_combination() {
        // Test realistic memory operation: output[i] = input[i] * 2
        let input_ptr = var("input");
        let output_ptr = var("output");
        let i = var("i");

        let loaded = load(input_ptr, i.clone(), DType::F32);
        let doubled = loaded * AstNode::Const(2.0f32.into());
        let stored = store(output_ptr, i, doubled);

        match stored {
            AstNode::Store { .. } => {}
            _ => panic!("Expected Store node"),
        }
    }

    #[test]
    fn test_barrier_helper() {
        let barrier_node = barrier();
        match barrier_node {
            AstNode::Barrier => {}
            _ => panic!("Expected Barrier node"),
        }
    }

    #[test]
    fn test_function_helper() {
        use crate::ast::{DType, FunctionKind, Mutability, VarDecl, VarKind};

        let params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }];

        let body = AstNode::Return {
            value: Box::new(var("x")),
        };

        let func = function(
            Some("test_func"),
            FunctionKind::Normal,
            params.clone(),
            DType::F32,
            body,
        );

        match func {
            AstNode::Function {
                name,
                params: func_params,
                return_type,
                kind,
                ..
            } => {
                assert_eq!(name, Some("test_func".to_string()));
                assert_eq!(func_params.len(), 1);
                assert_eq!(func_params[0].name, "x");
                assert_eq!(return_type, DType::F32);
                assert_eq!(kind, FunctionKind::Normal);
            }
            _ => panic!("Expected Function node"),
        }
    }

    #[test]
    fn test_program_helper() {
        use crate::ast::{DType, FunctionKind, Mutability, VarDecl, VarKind};

        let params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }];

        let body = AstNode::Return {
            value: Box::new(var("x")),
        };

        let main_func = function(Some("main"), FunctionKind::Normal, params, DType::F32, body);

        let prog = program(vec![main_func], "main");

        match prog {
            AstNode::Program {
                functions,
                entry_point,
            } => {
                assert_eq!(functions.len(), 1);
                assert_eq!(entry_point, "main");
            }
            _ => panic!("Expected Program node"),
        }
    }

    #[test]
    fn test_get_function() {
        use crate::ast::{DType, FunctionKind, Mutability, VarDecl, VarKind};

        let params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }];

        let body = AstNode::Return {
            value: Box::new(var("x")),
        };

        let main_func = function(
            Some("main"),
            FunctionKind::Normal,
            params.clone(),
            DType::F32,
            body.clone(),
        );

        let helper_func = function(
            Some("helper"),
            FunctionKind::Normal,
            params,
            DType::F32,
            body,
        );

        let prog = program(vec![main_func, helper_func], "main");

        // Get function by name
        let found = prog.get_function("helper");
        assert!(found.is_some());
        match found.unwrap() {
            AstNode::Function { name, .. } => {
                assert_eq!(name, &Some("helper".to_string()));
            }
            _ => panic!("Expected Function node"),
        }

        // Get entry point
        let entry = prog.get_entry();
        assert!(entry.is_some());
        match entry.unwrap() {
            AstNode::Function { name, .. } => {
                assert_eq!(name, &Some("main".to_string()));
            }
            _ => panic!("Expected Function node"),
        }

        // Try to get non-existent function
        assert!(prog.get_function("nonexistent").is_none());
    }
}

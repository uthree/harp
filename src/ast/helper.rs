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
impl_binary_helper!(add, Add, "Create an add node: a + b");
impl_binary_helper!(mul, Mul, "Create a multiply node: a * b");
impl_binary_helper!(max, Max, "Create a max node: max(a, b)");
impl_binary_helper!(idiv, Idiv, "Create an integer division node: a / b");
impl_binary_helper!(rem, Rem, "Create a remainder node: a % b");

// Comparison operation helpers
impl_binary_helper!(lt, Lt, "Create a less-than comparison: a < b");
impl_binary_helper!(le, Le, "Create a less-than-or-equal comparison: a <= b");
impl_binary_helper!(gt, Gt, "Create a greater-than comparison: a > b");
impl_binary_helper!(ge, Ge, "Create a greater-than-or-equal comparison: a >= b");
impl_binary_helper!(eq, Eq, "Create an equality comparison: a == b");
impl_binary_helper!(ne, Ne, "Create a not-equal comparison: a != b");

// Bitwise operation helpers
impl_binary_helper!(bitand, BitwiseAnd, "Create a bitwise AND: a & b");
impl_binary_helper!(bitor, BitwiseOr, "Create a bitwise OR: a | b");
impl_binary_helper!(bitxor, BitwiseXor, "Create a bitwise XOR: a ^ b");
impl_unary_helper!(bitnot, BitwiseNot, "Create a bitwise NOT: !a");
impl_binary_helper!(shl, LeftShift, "Create a left shift: a << b");
impl_binary_helper!(shr, RightShift, "Create a right shift: a >> b");

// Unary operation helpers
impl_unary_helper!(recip, Recip, "Create a reciprocal node: 1 / a");
impl_unary_helper!(sqrt, Sqrt, "Create a square root node: sqrt(a)");
impl_unary_helper!(log2, Log2, "Create a log2 node: log2(a)");
impl_unary_helper!(exp2, Exp2, "Create an exp2 node: 2^a");
impl_unary_helper!(sin, Sin, "Create a sine node: sin(a)");
impl_unary_helper!(floor, Floor, "Create a floor node: floor(a)");

/// Create a negation node: -a
/// Note: Uses Mul with -1.0 since there's no Neg variant in AstNode
pub fn neg(a: AstNode) -> AstNode {
    AstNode::Mul(
        Box::new(AstNode::Const(super::Literal::F32(-1.0))),
        Box::new(a),
    )
}

/// Create a fused multiply-add node: fma(a, b, c) = a * b + c
/// This is more accurate and potentially faster than separate multiply and add operations.
pub fn fma(a: AstNode, b: AstNode, c: AstNode) -> AstNode {
    AstNode::Fma {
        a: Box::new(a),
        b: Box::new(b),
        c: Box::new(c),
    }
}

/// Create an atomic add node for parallel reduction
/// Atomically adds value to the memory location at ptr[offset] and returns the old value.
pub fn atomic_add(ptr: AstNode, offset: AstNode, value: AstNode, dtype: DType) -> AstNode {
    AstNode::AtomicAdd {
        ptr: Box::new(ptr),
        offset: Box::new(offset),
        value: Box::new(value),
        dtype,
    }
}

/// Create an atomic max node for parallel reduction
/// Atomically computes max of current value and new value at ptr[offset], returns the old value.
pub fn atomic_max(ptr: AstNode, offset: AstNode, value: AstNode, dtype: DType) -> AstNode {
    AstNode::AtomicMax {
        ptr: Box::new(ptr),
        offset: Box::new(offset),
        value: Box::new(value),
        dtype,
    }
}

/// Create a random number node: generates uniform random value in [0, 1)
pub fn rand() -> AstNode {
    AstNode::Rand
}

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

/// Create a function node (normal function)
///
/// # Arguments
/// * `name` - Function name (can be None for anonymous functions)
/// * `params` - Parameter declarations
/// * `return_type` - Return type
/// * `body` - Function body (typically a Block node)
pub fn function(
    name: Option<impl Into<String>>,
    params: Vec<super::VarDecl>,
    return_type: DType,
    body: AstNode,
) -> AstNode {
    AstNode::Function {
        name: name.map(|n| n.into()),
        params,
        return_type,
        body: Box::new(body),
    }
}

/// Create a kernel node (GPU kernel function) with 3D dispatch configuration
///
/// # Arguments
/// * `name` - Kernel name (can be None for anonymous kernels)
/// * `params` - Parameter declarations
/// * `return_type` - Return type (usually void/unit)
/// * `body` - Kernel body (typically a Block node)
/// * `default_grid_size` - Default grid size (x, y, z) for CallKernel generation
/// * `default_thread_group_size` - Default thread group size (x, y, z) for CallKernel generation
pub fn kernel(
    name: Option<impl Into<String>>,
    params: Vec<super::VarDecl>,
    return_type: DType,
    body: AstNode,
    default_grid_size: [AstNode; 3],
    default_thread_group_size: [AstNode; 3],
) -> AstNode {
    AstNode::Kernel {
        name: name.map(|n| n.into()),
        params,
        return_type,
        body: Box::new(body),
        default_grid_size: [
            Box::new(default_grid_size[0].clone()),
            Box::new(default_grid_size[1].clone()),
            Box::new(default_grid_size[2].clone()),
        ],
        default_thread_group_size: [
            Box::new(default_thread_group_size[0].clone()),
            Box::new(default_thread_group_size[1].clone()),
            Box::new(default_thread_group_size[2].clone()),
        ],
    }
}

/// Create a kernel node with 1D dispatch configuration (convenience function)
///
/// Sets y and z dimensions to 1 automatically.
pub fn kernel_1d(
    name: Option<impl Into<String>>,
    params: Vec<super::VarDecl>,
    return_type: DType,
    body: AstNode,
    default_grid_size: AstNode,
    default_thread_group_size: AstNode,
) -> AstNode {
    let one = const_int(1);
    kernel(
        name,
        params,
        return_type,
        body,
        [default_grid_size, one.clone(), one.clone()],
        [default_thread_group_size, one.clone(), one],
    )
}

/// Create a kernel call node with 3D dispatch configuration
///
/// # Arguments
/// * `name` - Name of the kernel to call
/// * `args` - Arguments (buffer pointers, etc.)
/// * `grid_size` - Grid size (x, y, z) - number of thread groups
/// * `thread_group_size` - Thread group size (x, y, z) - threads per group
pub fn call_kernel(
    name: impl Into<String>,
    args: Vec<AstNode>,
    grid_size: [AstNode; 3],
    thread_group_size: [AstNode; 3],
) -> AstNode {
    AstNode::CallKernel {
        name: name.into(),
        args,
        grid_size: [
            Box::new(grid_size[0].clone()),
            Box::new(grid_size[1].clone()),
            Box::new(grid_size[2].clone()),
        ],
        thread_group_size: [
            Box::new(thread_group_size[0].clone()),
            Box::new(thread_group_size[1].clone()),
            Box::new(thread_group_size[2].clone()),
        ],
    }
}

/// Create a kernel call node with 1D dispatch configuration (convenience function)
///
/// Sets y and z dimensions to 1 automatically.
pub fn call_kernel_1d(
    name: impl Into<String>,
    args: Vec<AstNode>,
    grid_size: AstNode,
    thread_group_size: AstNode,
) -> AstNode {
    let one = const_int(1);
    call_kernel(
        name,
        args,
        [grid_size, one.clone(), one.clone()],
        [thread_group_size, one.clone(), one],
    )
}

/// Create a program node with empty execution waves
///
/// # Arguments
/// * `functions` - List of AstNode::Function or AstNode::Kernel
pub fn program(functions: Vec<AstNode>) -> AstNode {
    AstNode::Program {
        functions,
        execution_waves: vec![],
    }
}

/// Create a program node with execution waves
///
/// # Arguments
/// * `functions` - List of AstNode::Function or AstNode::Kernel
/// * `execution_waves` - Execution waves (groups of parallel-executable kernel calls)
pub fn program_with_execution_waves(
    functions: Vec<AstNode>,
    execution_waves: Vec<Vec<crate::ast::AstKernelCallInfo>>,
) -> AstNode {
    AstNode::Program {
        functions,
        execution_waves,
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

/// Create an if-then node (without else branch)
///
/// # Arguments
/// * `condition` - Condition expression (Bool type)
/// * `then_body` - Body to execute if condition is true
pub fn if_then(condition: AstNode, then_body: AstNode) -> AstNode {
    AstNode::If {
        condition: Box::new(condition),
        then_body: Box::new(then_body),
        else_body: None,
    }
}

/// Create an if-then-else node
///
/// # Arguments
/// * `condition` - Condition expression (Bool type)
/// * `then_body` - Body to execute if condition is true
/// * `else_body` - Body to execute if condition is false
pub fn if_then_else(condition: AstNode, then_body: AstNode, else_body: AstNode) -> AstNode {
    AstNode::If {
        condition: Box::new(condition),
        then_body: Box::new(then_body),
        else_body: Some(Box::new(else_body)),
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
pub fn const_int(value: i64) -> AstNode {
    AstNode::Const(Literal::I64(value))
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

        let isize_node = AstNode::Const(42i64.into());
        match isize_node {
            AstNode::Const(Literal::I64(v)) => assert_eq!(v, 42),
            _ => panic!("Expected Isize constant"),
        }

        let usize_node = AstNode::Const(100usize.into());
        match usize_node {
            AstNode::Const(Literal::I64(v)) => assert_eq!(v, 100),
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
        let cast_node = cast(a, DType::I64);
        match cast_node {
            AstNode::Cast(_, dtype) => match dtype {
                DType::I64 => {}
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
                    AstNode::Const(Literal::I64(v)) => assert_eq!(v, 0),
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
                    AstNode::Const(Literal::I64(v)) => assert_eq!(v, 0),
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
        let value = AstNode::Const(42i64.into());
        let assign_node = assign("alu0", value);

        match assign_node {
            AstNode::Assign { var, value } => {
                assert_eq!(var, "alu0");
                match *value {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(v, 42),
                    _ => panic!("Expected Isize constant for value"),
                }
            }
            _ => panic!("Expected Assign node"),
        }

        // Test with String
        let assign_node2 = assign("temp".to_string(), AstNode::Const(10i64.into()));
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
        use crate::ast::{DType, Mutability, VarDecl, VarKind};

        let params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }];

        let body = AstNode::Return {
            value: Box::new(var("x")),
        };

        let func = function(Some("test_func"), params.clone(), DType::F32, body);

        match func {
            AstNode::Function {
                name,
                params: func_params,
                return_type,
                ..
            } => {
                assert_eq!(name, Some("test_func".to_string()));
                assert_eq!(func_params.len(), 1);
                assert_eq!(func_params[0].name, "x");
                assert_eq!(return_type, DType::F32);
            }
            _ => panic!("Expected Function node"),
        }
    }

    #[test]
    fn test_kernel_helper() {
        use crate::ast::{DType, Literal, Mutability, VarDecl, VarKind};

        let params = vec![VarDecl {
            name: "buffer".to_string(),
            dtype: DType::Ptr(Box::new(DType::F32)),
            mutability: Mutability::Mutable,
            kind: VarKind::Normal,
        }];

        let body = empty_block();

        // Test 3D kernel helper
        let kern = kernel(
            Some("compute_kernel"),
            params.clone(),
            DType::Tuple(vec![]),
            body.clone(),
            [const_int(16), const_int(16), const_int(1)],
            [const_int(64), const_int(1), const_int(1)],
        );

        match kern {
            AstNode::Kernel {
                name,
                params: kern_params,
                return_type,
                default_grid_size,
                default_thread_group_size,
                ..
            } => {
                assert_eq!(name, Some("compute_kernel".to_string()));
                assert_eq!(kern_params.len(), 1);
                assert_eq!(kern_params[0].name, "buffer");
                assert_eq!(return_type, DType::Tuple(vec![]));
                // Check grid size
                match default_grid_size[0].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 16),
                    _ => panic!("Expected Int constant for grid_size[0]"),
                }
                // Check thread group size
                match default_thread_group_size[0].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 64),
                    _ => panic!("Expected Int constant for thread_group_size[0]"),
                }
            }
            _ => panic!("Expected Kernel node"),
        }

        // Test 1D kernel helper
        let kern_1d = kernel_1d(
            Some("simple_kernel"),
            params,
            DType::Tuple(vec![]),
            body,
            const_int(1024),
            const_int(64),
        );

        match kern_1d {
            AstNode::Kernel {
                default_grid_size,
                default_thread_group_size,
                ..
            } => {
                // Y and Z should be 1
                match default_grid_size[1].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 1),
                    _ => panic!("Expected Int constant 1 for grid_size[1]"),
                }
                match default_thread_group_size[1].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 1),
                    _ => panic!("Expected Int constant 1 for thread_group_size[1]"),
                }
            }
            _ => panic!("Expected Kernel node"),
        }
    }

    #[test]
    fn test_call_kernel_helper() {
        use crate::ast::Literal;

        // Test 3D call_kernel
        let call = call_kernel(
            "my_kernel",
            vec![var("a"), var("b")],
            [const_int(8), const_int(8), const_int(1)],
            [const_int(32), const_int(32), const_int(1)],
        );

        match call {
            AstNode::CallKernel {
                name,
                args,
                grid_size,
                thread_group_size,
            } => {
                assert_eq!(name, "my_kernel");
                assert_eq!(args.len(), 2);
                match grid_size[0].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 8),
                    _ => panic!("Expected Int constant for grid_size[0]"),
                }
                match thread_group_size[0].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 32),
                    _ => panic!("Expected Int constant for thread_group_size[0]"),
                }
            }
            _ => panic!("Expected CallKernel node"),
        }

        // Test 1D call_kernel
        let call_1d = call_kernel_1d("simple", vec![var("x")], const_int(100), const_int(64));

        match call_1d {
            AstNode::CallKernel {
                grid_size,
                thread_group_size,
                ..
            } => {
                // Y and Z should be 1
                match grid_size[1].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 1),
                    _ => panic!("Expected Int constant 1 for grid_size[1]"),
                }
                match thread_group_size[2].as_ref() {
                    AstNode::Const(Literal::I64(v)) => assert_eq!(*v, 1),
                    _ => panic!("Expected Int constant 1 for thread_group_size[2]"),
                }
            }
            _ => panic!("Expected CallKernel node"),
        }
    }

    #[test]
    fn test_program_helper() {
        use crate::ast::{DType, Mutability, VarDecl, VarKind};

        let params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }];

        let body = AstNode::Return {
            value: Box::new(var("x")),
        };

        let main_func = function(Some("main"), params, DType::F32, body);

        let prog = program(vec![main_func]);

        match prog {
            AstNode::Program { functions, .. } => {
                assert_eq!(functions.len(), 1);
            }
            _ => panic!("Expected Program node"),
        }
    }

    #[test]
    fn test_get_function() {
        use crate::ast::{DType, Mutability, VarDecl, VarKind};

        let params = vec![VarDecl {
            name: "x".to_string(),
            dtype: DType::F32,
            mutability: Mutability::Immutable,
            kind: VarKind::Normal,
        }];

        let body = AstNode::Return {
            value: Box::new(var("x")),
        };

        let main_func = function(Some("main"), params.clone(), DType::F32, body.clone());

        let helper_func = function(Some("helper"), params, DType::F32, body);

        let prog = program(vec![main_func, helper_func]);

        // Get function by name
        let found = prog.get_function("helper");
        assert!(found.is_some());
        match found.unwrap() {
            AstNode::Function { name, .. } => {
                assert_eq!(name.clone(), Some("helper".to_string()));
            }
            _ => panic!("Expected Function node"),
        }

        // Try to get non-existent function
        assert!(prog.get_function("nonexistent").is_none());
    }
}

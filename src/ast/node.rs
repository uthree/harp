use super::{ConstLiteral, DType, KernelScope, Scope};

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

    // comparison ops (return Bool)
    LessThan(Box<Self>, Box<Self>), // x < y
    Eq(Box<Self>, Box<Self>),       // x == y

    // conditional selection
    Select {
        cond: Box<Self>,      // Bool型の条件
        true_val: Box<Self>,  // 条件が真の場合の値
        false_val: Box<Self>, // 条件が偽の場合の値
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
    Load {
        target: Box<Self>,
        index: Box<Self>,
        vector_width: usize, // number of elements to load (1=scalar, 2,4,8,...=vector)
    }, // load value(s) from memory location (target[index..index+vector_width])
    Store {
        target: Box<Self>,
        index: Box<Self>,
        value: Box<Self>,
        vector_width: usize, // number of elements to store (1=scalar, 2,4,8,...=vector)
    }, // store value(s) to memory location (target[index..index+vector_width] = value)

    Range {
        // Forループ (start から max-1 まで、stepずつインクリメント)
        counter_name: String, // ループカウンタの変数名
        start: Box<Self>,     // 開始値（デフォルトは0）
        max: Box<Self>,       // 終了値
        step: Box<Self>,      // インクリメント量（デフォルトは1）
        body: Box<Self>,
        unroll: Option<usize>, // #pragma unroll相当のヒント (None=no unroll, Some(0)=full unroll, Some(n)=unroll n times)
    },

    If {
        condition: Box<Self>,           // Bool型の条件式
        then_branch: Box<Self>,         // 条件が真の場合に実行
        else_branch: Option<Box<Self>>, // 条件が偽の場合に実行（オプション）
    },

    Drop(String), // drop (local) variable explicitly

    Barrier, // Synchronization barrier for parallel execution (separates computation generations)

    // Function definition
    Function {
        name: String,
        scope: Scope,
        statements: Vec<AstNode>,
        arguments: Vec<(String, DType)>,
        return_type: DType,
    },

    // GPU Kernel definition (parallelizable function)
    Kernel {
        name: String,
        scope: KernelScope, // Kernel has both regular variables and thread ID declarations
        statements: Vec<AstNode>,
        arguments: Vec<(String, DType)>,
        return_type: DType,
        global_size: [Box<AstNode>; 3], // Global work size for each dimension [x, y, z]
        local_size: [Box<AstNode>; 3],  // Local work group size for each dimension [x, y, z]
    },

    // Call a kernel with grid configuration
    CallKernel {
        name: String,
        args: Vec<AstNode>,
        global_size: [Box<AstNode>; 3], // Global work size for each dimension [x, y, z]
        local_size: [Box<AstNode>; 3],  // Local work group size for each dimension [x, y, z]
    },

    // Program definition
    Program {
        functions: Vec<AstNode>,
        entry_point: String,
    },

    // for pattern matching
    Capture(usize),
}

impl AstNode {
    /// Create a Program node
    pub fn program(functions: Vec<AstNode>, entry_point: impl Into<String>) -> Self {
        AstNode::Program {
            functions,
            entry_point: entry_point.into(),
        }
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

    /// Create a Kernel node
    pub fn kernel(
        name: impl Into<String>,
        arguments: Vec<(String, DType)>,
        return_type: DType,
        scope: KernelScope,
        statements: Vec<AstNode>,
        global_size: [Box<AstNode>; 3],
        local_size: [Box<AstNode>; 3],
    ) -> Self {
        AstNode::Kernel {
            name: name.into(),
            scope,
            statements,
            arguments,
            return_type,
            global_size,
            local_size,
        }
    }

    /// Create a CallKernel node
    pub fn call_kernel(
        name: impl Into<String>,
        args: Vec<AstNode>,
        global_size: [Box<AstNode>; 3],
        local_size: [Box<AstNode>; 3],
    ) -> Self {
        AstNode::CallKernel {
            name: name.into(),
            args,
            global_size,
            local_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        helper::*, ConstLiteral, KernelScope, ThreadIdDecl, ThreadIdType, VariableDecl,
    };

    #[test]
    fn test_kernel_creation() {
        // Create a simple kernel with thread IDs (as 3D vectors)
        let mut scope = KernelScope::new();
        scope.thread_ids.push(ThreadIdDecl {
            name: "global_id".to_string(),
            id_type: ThreadIdType::GlobalId,
        });

        let kernel = AstNode::kernel(
            "test_kernel",
            vec![("input".to_string(), DType::Ptr(Box::new(DType::F32)))],
            DType::Void,
            scope,
            vec![barrier()],
            [
                Box::new(const_val(ConstLiteral::Isize(1024))),
                Box::new(const_val(ConstLiteral::Isize(1))),
                Box::new(const_val(ConstLiteral::Isize(1))),
            ],
            [
                Box::new(const_val(ConstLiteral::Isize(256))),
                Box::new(const_val(ConstLiteral::Isize(1))),
                Box::new(const_val(ConstLiteral::Isize(1))),
            ],
        );

        // Verify kernel structure
        match kernel {
            AstNode::Kernel {
                name,
                scope,
                global_size,
                ..
            } => {
                assert_eq!(name, "test_kernel");
                assert_eq!(scope.thread_ids.len(), 1);
                assert_eq!(scope.thread_ids[0].name, "global_id");
                assert_eq!(scope.thread_ids[0].id_type, ThreadIdType::GlobalId);

                // Verify global_size
                assert_eq!(*global_size[0], const_val(ConstLiteral::Isize(1024)));
            }
            _ => panic!("Expected Kernel node"),
        }
    }

    #[test]
    fn test_call_kernel_creation() {
        let call = AstNode::call_kernel(
            "test_kernel",
            vec![var("input_ptr")],
            [
                Box::new(const_val(ConstLiteral::Isize(1024))),
                Box::new(const_val(ConstLiteral::Isize(1))),
                Box::new(const_val(ConstLiteral::Isize(1))),
            ],
            [
                Box::new(const_val(ConstLiteral::Isize(256))),
                Box::new(const_val(ConstLiteral::Isize(1))),
                Box::new(const_val(ConstLiteral::Isize(1))),
            ],
        );

        // Verify CallKernel structure
        match call {
            AstNode::CallKernel {
                name,
                args,
                global_size,
                local_size,
            } => {
                assert_eq!(name, "test_kernel");
                assert_eq!(args.len(), 1);
                assert_eq!(*global_size[0], const_val(ConstLiteral::Isize(1024)));
                assert_eq!(*local_size[0], const_val(ConstLiteral::Isize(256)));
            }
            _ => panic!("Expected CallKernel node"),
        }
    }

    #[test]
    fn test_kernel_scope_methods() {
        let mut scope = KernelScope::new();

        // Add thread IDs (as 3D vectors)
        scope.thread_ids.push(ThreadIdDecl {
            name: "global_id".to_string(),
            id_type: ThreadIdType::GlobalId,
        });
        scope.thread_ids.push(ThreadIdDecl {
            name: "local_id".to_string(),
            id_type: ThreadIdType::LocalId,
        });

        // Test get_thread_id_name
        assert_eq!(
            scope.get_thread_id_name(&ThreadIdType::GlobalId),
            Some("global_id")
        );
        assert_eq!(
            scope.get_thread_id_name(&ThreadIdType::LocalId),
            Some("local_id")
        );
        assert_eq!(scope.get_thread_id_name(&ThreadIdType::GroupId), None);

        // Test has_name_conflict
        assert!(scope.has_name_conflict("global_id"));
        assert!(scope.has_name_conflict("local_id"));
        assert!(!scope.has_name_conflict("other_var"));

        // Add regular variable
        scope.declarations.push(VariableDecl {
            name: "temp".to_string(),
            dtype: DType::F32,
            constant: false,
            size_expr: None,
        });
        assert!(scope.has_name_conflict("temp"));

        // Test all_thread_id_names
        let names = scope.all_thread_id_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"global_id"));
        assert!(names.contains(&"local_id"));

        // Test thread ID dtype
        let dtype = KernelScope::get_thread_id_dtype();
        assert_eq!(dtype, DType::Vec(Box::new(DType::Usize), 3));
    }

    #[test]
    fn test_kernel_children() {
        let mut scope = KernelScope::new();
        scope.thread_ids.push(ThreadIdDecl {
            name: "global_id".to_string(),
            id_type: ThreadIdType::GlobalId,
        });

        let kernel = AstNode::kernel(
            "test_kernel",
            vec![],
            DType::Void,
            scope,
            vec![barrier(), assign("x", const_val(ConstLiteral::F32(1.0)))],
            [
                Box::new(const_val(ConstLiteral::Isize(1024))),
                Box::new(const_val(ConstLiteral::Isize(1))),
                Box::new(const_val(ConstLiteral::Isize(1))),
            ],
            [
                Box::new(const_val(ConstLiteral::Isize(256))),
                Box::new(const_val(ConstLiteral::Isize(1))),
                Box::new(const_val(ConstLiteral::Isize(1))),
            ],
        );

        // Children should include statements + grid size expressions (2 statements + 6 grid sizes = 8 children)
        let children = kernel.children();
        assert_eq!(children.len(), 8);
    }
}

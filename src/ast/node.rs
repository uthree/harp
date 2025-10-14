use super::{ConstLiteral, DType, Scope};

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
}

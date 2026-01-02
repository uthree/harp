mod dtype;

#[derive(Debug, Clone)]
pub enum Node {
    // Const and variables
    Var(String),

    // math operators
    Neg(Box<Node>),
    Recip(Box<Node>),
    Sqrt(Box<Node>),
    Sin(Box<Node>),
    Log2(Box<Node>),
    Exp2(Box<Node>),
    Add(Box<Node>, Box<Node>),
    Mul(Box<Node>, Box<Node>),
    Max(Box<Node>, Box<Node>),
    Rem(Box<Node>, Box<Node>),
    Idiv(Box<Node>, Box<Node>),

    // Read value from buffer
    Load {
        buffer: Box<Node>,
        offset: Box<Node>,
        size: usize,
    },

    // Statements
    Store {
        // Write value to buffer
        buffer: Box<Node>,
        offset: Box<Node>,
        value: Box<Node>,
    },

    // loop statement
    Loop {
        start: Box<Node>,
        stop: Box<Node>,
        step: Box<Node>,
        body: Box<Node>,
    },

    // block
    Block(Vec<Node>),

    // Functions
    Function {},
    Kernel {},
}

struct Scope {}

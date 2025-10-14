use super::AstNode;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Default, Eq, Hash)]
pub enum DType {
    #[default]
    F32, // float
    Usize, // size_t
    Isize, // ssize_t
    Bool,  // boolean
    Void,

    Ptr(Box<Self>),        // pointer
    Vec(Box<Self>, usize), // fixed-size array (for SIMD vectorization)
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::F32 => write!(f, "F32"),
            DType::Usize => write!(f, "Usize"),
            DType::Isize => write!(f, "Isize"),
            DType::Bool => write!(f, "Bool"),
            DType::Void => write!(f, "Void"),
            DType::Ptr(inner) => write!(f, "Ptr<{}>", inner),
            DType::Vec(inner, size) => write!(f, "Vec<{}, {}>", inner, size),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstLiteral {
    F32(f32),
    Usize(usize),
    Isize(isize),
    Bool(bool),
}

// f32はEqを実装していないので手動でEqとHashを実装
impl Eq for ConstLiteral {}

impl std::hash::Hash for ConstLiteral {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ConstLiteral::F32(f) => {
                0u8.hash(state);
                f.to_bits().hash(state);
            }
            ConstLiteral::Usize(u) => {
                1u8.hash(state);
                u.hash(state);
            }
            ConstLiteral::Isize(i) => {
                2u8.hash(state);
                i.hash(state);
            }
            ConstLiteral::Bool(b) => {
                3u8.hash(state);
                b.hash(state);
            }
        }
    }
}

/// Variable declaration in a scope
#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub dtype: DType,
    pub constant: bool,
    pub size_expr: Option<Box<AstNode>>, // For dynamic arrays, the size expression
}

/// Scope containing variable declarations
#[derive(Debug, Clone, PartialEq)]
pub struct Scope {
    pub declarations: Vec<VariableDecl>,
}

/// Type of thread ID variable for GPU parallelization
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ThreadIdType {
    GlobalId, // Global thread ID (across all work groups)
    LocalId,  // Local thread ID (within a work group)
    GroupId,  // Work group ID
}

/// Thread ID variable declaration for GPU kernels
/// Thread IDs are 3-dimensional vectors (accessed as array[0..2] for x,y,z)
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadIdDecl {
    pub name: String,
    pub id_type: ThreadIdType,
}

/// Kernel scope containing both regular variables and thread ID declarations
#[derive(Debug, Clone, PartialEq)]
pub struct KernelScope {
    pub declarations: Vec<VariableDecl>,
    pub thread_ids: Vec<ThreadIdDecl>,
}

impl Default for KernelScope {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelScope {
    /// Create a new empty kernel scope
    pub fn new() -> Self {
        Self {
            declarations: Vec::new(),
            thread_ids: Vec::new(),
        }
    }

    /// Get thread ID variable name by type
    pub fn get_thread_id_name(&self, id_type: &ThreadIdType) -> Option<&str> {
        self.thread_ids
            .iter()
            .find(|decl| decl.id_type == *id_type)
            .map(|decl| decl.name.as_str())
    }

    /// Check if a name conflicts with any thread ID or regular variable
    pub fn has_name_conflict(&self, name: &str) -> bool {
        self.thread_ids.iter().any(|decl| decl.name == name)
            || self.declarations.iter().any(|decl| decl.name == name)
    }

    /// Get all thread ID variable names
    pub fn all_thread_id_names(&self) -> Vec<&str> {
        self.thread_ids
            .iter()
            .map(|decl| decl.name.as_str())
            .collect()
    }

    /// Get the data type for a thread ID (always Vec<Usize, 3>)
    pub fn get_thread_id_dtype() -> DType {
        DType::Vec(Box::new(DType::Usize), 3)
    }
}

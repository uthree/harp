//! DSL error types

use std::fmt;

/// DSL parsing and compilation errors
#[derive(Debug)]
pub enum DslError {
    /// Parsing error from pest
    ParseError(String),
    /// Unknown dtype
    UnknownDType(String),
    /// Unknown binary operator
    UnknownBinOp(String),
    /// Unknown comparison operator
    UnknownCompOp(String),
    /// Undefined variable reference
    UndefinedVariable(String),
    /// Unknown function
    UnknownFunction(String),
    /// Invalid argument count
    InvalidArgCount {
        func: String,
        expected: usize,
        got: usize,
    },
    /// Invalid argument type
    InvalidArgType {
        func: String,
        arg: String,
        expected: String,
        got: String,
    },
    /// Shape mismatch
    ShapeMismatch {
        context: String,
        expected: Vec<i64>,
        got: Vec<i64>,
    },
    /// Type mismatch
    TypeMismatch {
        context: String,
        expected: String,
        got: String,
    },
    /// Unresolved dynamic dimension
    UnresolvedDynamicDim(String),
    /// IO error
    IoError(std::io::Error),
}

impl fmt::Display for DslError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DslError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            DslError::UnknownDType(dt) => write!(f, "Unknown data type: {}", dt),
            DslError::UnknownBinOp(op) => write!(f, "Unknown binary operator: {}", op),
            DslError::UnknownCompOp(op) => write!(f, "Unknown comparison operator: {}", op),
            DslError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            DslError::UnknownFunction(name) => write!(f, "Unknown function: {}", name),
            DslError::InvalidArgCount {
                func,
                expected,
                got,
            } => write!(
                f,
                "Invalid argument count for '{}': expected {}, got {}",
                func, expected, got
            ),
            DslError::InvalidArgType {
                func,
                arg,
                expected,
                got,
            } => write!(
                f,
                "Invalid argument type for '{}' in '{}': expected {}, got {}",
                arg, func, expected, got
            ),
            DslError::ShapeMismatch {
                context,
                expected,
                got,
            } => write!(
                f,
                "Shape mismatch in {}: expected {:?}, got {:?}",
                context, expected, got
            ),
            DslError::TypeMismatch {
                context,
                expected,
                got,
            } => write!(
                f,
                "Type mismatch in {}: expected {}, got {}",
                context, expected, got
            ),
            DslError::UnresolvedDynamicDim(name) => {
                write!(f, "Unresolved dynamic dimension: {}", name)
            }
            DslError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for DslError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DslError::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for DslError {
    fn from(e: std::io::Error) -> Self {
        DslError::IoError(e)
    }
}

pub type DslResult<T> = Result<T, DslError>;

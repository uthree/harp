//! Error types for the Harp DSL

use thiserror::Error;

/// Main error type for DSL operations
#[derive(Debug, Error)]
pub enum DslError {
    /// Parse error with location information
    #[error("Parse error at line {line}, column {column}: {message}")]
    ParseError {
        line: usize,
        column: usize,
        message: String,
    },

    /// Type checking error
    #[error("Type error: {0}")]
    TypeError(String),

    /// Shape mismatch error
    #[error("Shape error: {0}")]
    ShapeError(String),

    /// Undefined variable error
    #[error("Undefined variable: {0}")]
    UndefinedVariable(String),

    /// Duplicate definition error
    #[error("Duplicate definition: {0}")]
    DuplicateDefinition(String),

    /// Unsupported operation error
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Invalid argument error
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),

    /// Compilation error
    #[error("Compilation error: {0}")]
    CompilationError(String),
}

impl DslError {
    /// Create a parse error from pest error
    pub fn from_pest_error(err: pest::error::Error<crate::parser::Rule>) -> Self {
        let (line, column) = match err.line_col {
            pest::error::LineColLocation::Pos((l, c)) => (l, c),
            pest::error::LineColLocation::Span((l, c), _) => (l, c),
        };
        DslError::ParseError {
            line,
            column,
            message: err.variant.message().to_string(),
        }
    }
}

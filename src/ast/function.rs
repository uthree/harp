use super::{AstNode, DType, Scope};

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub(crate) name: String,
    pub(crate) scope: Scope,
    pub(crate) statements: Vec<AstNode>,
    pub(crate) arguments: Vec<(String, DType)>,
    pub(crate) return_type: DType,
}

impl Function {
    pub fn new(
        name: String,
        arguments: Vec<(String, DType)>,
        return_type: DType,
        scope: Scope,
        statements: Vec<AstNode>,
    ) -> Self {
        Self {
            name,
            arguments,
            return_type,
            scope,
            statements,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    pub fn statements(&self) -> &[AstNode] {
        &self.statements
    }

    pub fn arguments(&self) -> &[(String, DType)] {
        &self.arguments
    }

    pub fn return_type(&self) -> &DType {
        &self.return_type
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub(crate) functions: Vec<AstNode>,
    pub(crate) entry_point: String,
}

//! 関数とプログラムの定義

use super::AstNode;
use super::scope::{Scope, VarDecl, VarKind};
use super::types::DType;
use std::collections::HashMap;

/// 関数定義（ASTノードではなく、構造体としての関数表現）
///
/// 注意: GPUカーネルは `AstNode::Kernel` を使用してください。
/// この構造体は通常関数の定義に使用します。
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub params: Vec<VarDecl>, // 引数リスト
    pub return_type: DType,   // 返り値の型
    pub body: Box<AstNode>,   // 関数本体（Blockノード）
}

/// プログラム全体の構造
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub functions: HashMap<String, Function>, // 関数定義の集合
    pub entry_point: String,                  // エントリーポイントの関数名
}

impl Function {
    /// Create a new function with parameters and return type
    pub fn new(
        params: Vec<VarDecl>,
        return_type: DType,
        body_statements: Vec<AstNode>,
    ) -> Result<Self, String> {
        // Create scope with parameters declared
        let mut scope = Scope::new();
        for param in &params {
            // VarKindがNormalの場合のみスコープに宣言
            // ThreadId/GroupIdなどは組み込み値なのでスコープには登録しない
            if param.kind == VarKind::Normal {
                scope.declare(
                    param.name.clone(),
                    param.dtype.clone(),
                    param.mutability.clone(),
                )?;
            }
        }

        // Create Block node with the body statements
        let body = Box::new(AstNode::Block {
            statements: body_statements,
            scope: Box::new(scope.clone()),
        });

        Ok(Function {
            params,
            return_type,
            body,
        })
    }

    /// Check if the function body is valid within its scope
    pub fn check_body(&self) -> Result<(), String> {
        // bodyはBlockノードであるべき
        if let AstNode::Block { scope, .. } = self.body.as_ref() {
            self.body.check_scope(scope)?;
        } else {
            return Err("Function body must be a Block node".to_string());
        }
        Ok(())
    }

    /// Infer the return type from the function body
    /// Looks for Return statements and checks consistency
    pub fn infer_return_type(&self) -> DType {
        if let AstNode::Block { statements, .. } = self.body.as_ref() {
            for node in statements {
                if let AstNode::Return { value } = node {
                    return value.infer_type();
                }
            }
        }
        // If no explicit return, return unit type
        DType::Tuple(vec![])
    }
}

impl Program {
    /// Create a new empty program
    pub fn new(entry_point: String) -> Self {
        Program {
            functions: HashMap::new(),
            entry_point,
        }
    }

    /// Add a function to the program
    pub fn add_function(&mut self, name: String, function: Function) -> Result<(), String> {
        if self.functions.contains_key(&name) {
            return Err(format!("Function '{}' is already defined", name));
        }
        self.functions.insert(name, function);
        Ok(())
    }

    /// Get a function by name
    pub fn get_function(&self, name: &str) -> Option<&Function> {
        self.functions.get(name)
    }

    /// Check if a function exists
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Get the entry point function
    pub fn get_entry(&self) -> Option<&Function> {
        self.get_function(&self.entry_point)
    }

    /// Validate the entire program
    /// - Check that entry point exists
    /// - Check all function bodies
    /// - Check that all called functions exist
    pub fn validate(&self) -> Result<(), String> {
        // Check entry point exists
        if !self.has_function(&self.entry_point) {
            return Err(format!(
                "Entry point function '{}' not found",
                self.entry_point
            ));
        }

        // Check all function bodies
        for (name, function) in &self.functions {
            function
                .check_body()
                .map_err(|e| format!("Error in function '{}': {}", name, e))?;
        }

        // TODO: Check that all called functions exist (requires traversing AST)

        Ok(())
    }
}

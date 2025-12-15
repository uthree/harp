//! 関数とプログラムの定義

use super::AstNode;
use super::scope::{Scope, VarDecl, VarKind};
use super::types::DType;
use crate::graph::shape::Expr;
use std::collections::HashMap;

/// カーネル呼び出しメタデータ
///
/// ホスト側でカーネルを順次実行するための情報を保持します。
/// サブグラフが個別カーネルとして生成される場合に使用されます。
#[derive(Clone, Debug, PartialEq)]
pub struct KernelCall {
    /// 呼び出すカーネル名
    pub kernel_name: String,
    /// 入力バッファ名リスト
    pub inputs: Vec<String>,
    /// 出力バッファ名リスト
    pub outputs: Vec<String>,
    /// グリッドサイズ（ワークアイテム数）
    pub grid_size: Vec<Expr>,
    /// スレッドグループサイズ（ローカルワークサイズ）
    pub thread_group_size: Vec<Expr>,
}

impl KernelCall {
    /// 新しいKernelCallを作成
    pub fn new(
        kernel_name: impl Into<String>,
        inputs: Vec<String>,
        outputs: Vec<String>,
        grid_size: Vec<Expr>,
        thread_group_size: Vec<Expr>,
    ) -> Self {
        Self {
            kernel_name: kernel_name.into(),
            inputs,
            outputs,
            grid_size,
            thread_group_size,
        }
    }
}

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
///
/// 複数のカーネル関数を保持するコンテナです。
/// カーネルの実行順序はホスト側（CompiledProgram）で管理されます。
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub functions: HashMap<String, Function>, // 関数定義の集合
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
    pub fn new() -> Self {
        Program {
            functions: HashMap::new(),
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

    /// Get all function names
    pub fn function_names(&self) -> impl Iterator<Item = &String> {
        self.functions.keys()
    }

    /// Validate the entire program
    /// - Check all function bodies
    /// - Check that all called functions exist
    pub fn validate(&self) -> Result<(), String> {
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

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

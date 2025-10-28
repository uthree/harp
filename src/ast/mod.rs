// Operator overloading for AstNode
pub mod ops;
// Helper functions for constructing AST nodes
pub mod helper;
pub mod pat;

use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub enum AstNode {
    // Pattern matching wildcard - パターンマッチング用ワイルドカード
    Wildcard(String),

    // arithmetics - 算術演算
    Const(Literal),
    Add(Box<AstNode>, Box<AstNode>),
    Mul(Box<AstNode>, Box<AstNode>),
    Max(Box<AstNode>, Box<AstNode>),
    Rem(Box<AstNode>, Box<AstNode>),
    Idiv(Box<AstNode>, Box<AstNode>),
    Recip(Box<AstNode>),
    Sqrt(Box<AstNode>),
    Log2(Box<AstNode>),
    Exp2(Box<AstNode>),
    Sin(Box<AstNode>),
    Cast(Box<AstNode>, DType),

    // Variables - 変数
    Var(String),

    // Memory operations - メモリ操作（バッファー用）
    Load {
        ptr: Box<AstNode>,    // ポインタ（Ptr<T>型の式）
        offset: Box<AstNode>, // オフセット（Usize型の式）
        count: usize,         // 読み込む要素数（コンパイル時定数、1ならスカラー）
    },
    Store {
        ptr: Box<AstNode>,    // ポインタ（Ptr<T>型の式）
        offset: Box<AstNode>, // オフセット（Usize型の式）
        value: Box<AstNode>,  // 書き込む値（スカラーまたはVec型）
    },

    // Assignment - 変数への代入（スタック/レジスタ用）
    Assign {
        var: String,         // 変数名
        value: Box<AstNode>, // 代入する値
    },

    // Block - 文のブロックとスコープ
    Block {
        statements: Vec<AstNode>, // 文のリスト
        scope: Box<Scope>,        // ブロックのスコープ
    },

    // Control flow - 制御構文
    Range {
        var: String,         // ループ変数名
        start: Box<AstNode>, // 開始値
        step: Box<AstNode>,  // ステップ
        stop: Box<AstNode>,  // 終了値
        body: Box<AstNode>,  // ループ本体（Blockノード）
    },

    // Function call - 関数呼び出し
    Call {
        name: String,       // 関数名
        args: Vec<AstNode>, // 引数リスト
    },

    // Return statement - 返り値
    Return {
        value: Box<AstNode>, // 返す値
    },
}

/// 関数定義
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub params: Vec<(String, DType)>, // 引数リスト: (変数名, 型)
    pub return_type: DType,           // 返り値の型
    pub body: Box<AstNode>,           // 関数本体（Blockノード）
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
        params: Vec<(String, DType)>,
        return_type: DType,
        body_statements: Vec<AstNode>,
    ) -> Result<Self, String> {
        // Create scope with parameters declared
        let mut scope = Scope::new();
        for (param_name, param_type) in &params {
            scope.declare(
                param_name.clone(),
                param_type.clone(),
                Mutability::Immutable, // パラメータは基本的にimmutable
                AccessRegion::ThreadLocal,
            )?;
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

// Scope management for variable access control
#[derive(Clone, Debug, PartialEq)]
pub struct Scope {
    // 変数名 -> 宣言情報
    variables: HashMap<String, VarDecl>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    /// Create a new empty scope
    pub fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            parent: None,
        }
    }

    /// Create a new scope with a parent
    pub fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// Declare a variable in this scope
    pub fn declare(
        &mut self,
        name: String,
        dtype: DType,
        mutability: Mutability,
        region: AccessRegion,
    ) -> Result<(), String> {
        if self.variables.contains_key(&name) {
            return Err(format!(
                "Variable '{}' already declared in this scope",
                name
            ));
        }
        self.variables.insert(
            name,
            VarDecl {
                dtype,
                mutability,
                region,
            },
        );
        Ok(())
    }

    /// Look up a variable in this scope or parent scopes
    fn lookup(&self, name: &str) -> Option<&VarDecl> {
        self.variables
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }

    /// Check if a variable can be read
    pub fn check_read(&self, name: &str) -> Result<DType, String> {
        self.lookup(name)
            .map(|decl| decl.dtype.clone())
            .ok_or_else(|| format!("Undeclared variable: '{}'", name))
    }

    /// Check if a variable can be written to
    pub fn check_write(&self, name: &str, value_type: &DType) -> Result<(), String> {
        let decl = self
            .lookup(name)
            .ok_or_else(|| format!("Undeclared variable: '{}'", name))?;

        // Mutability check
        if decl.mutability == Mutability::Immutable {
            return Err(format!("Cannot write to immutable variable: '{}'", name));
        }

        // Type check
        if &decl.dtype != value_type {
            return Err(format!(
                "Type mismatch for '{}': expected {:?}, found {:?}",
                name, decl.dtype, value_type
            ));
        }

        Ok(())
    }

    /// Check if two variables can be accessed in parallel
    pub fn can_access_parallel(&self, var1: &str, var2: &str) -> bool {
        let Some(decl1) = self.lookup(var1) else {
            return false;
        };
        let Some(decl2) = self.lookup(var2) else {
            return false;
        };

        // 両方がImmutableなら常に並列OK
        if decl1.mutability == Mutability::Immutable && decl2.mutability == Mutability::Immutable {
            return true;
        }

        // 両方がThreadLocalなら並列OK
        if decl1.region == AccessRegion::ThreadLocal && decl2.region == AccessRegion::ThreadLocal {
            return true;
        }

        // ShardedByで分割されている場合
        match (&decl1.region, &decl2.region) {
            (AccessRegion::ShardedBy(axes1), AccessRegion::ShardedBy(axes2)) => {
                // 異なる軸でシャーディングされていればOK
                axes1.iter().any(|a1| axes2.iter().any(|a2| a1 != a2))
            }
            _ => false,
        }
    }

    /// Get a reference to a variable declaration
    pub fn get(&self, name: &str) -> Option<&VarDecl> {
        self.lookup(name)
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VarDecl {
    pub dtype: DType,
    pub mutability: Mutability,
    pub region: AccessRegion,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Mutability {
    Immutable, // 読み取り専用（複数スレッドから安全にアクセス可）
    Mutable,   // 書き込み可能（単一スレッドのみ）
}

#[derive(Clone, Debug, PartialEq)]
pub enum AccessRegion {
    ThreadLocal,           // スレッドローカル（競合なし）
    Shared,                // 共有メモリ（読み取り専用なら安全）
    ShardedBy(Vec<usize>), // 特定の軸でシャーディング（軸番号のリスト）
}

#[derive(Debug, Clone, PartialEq)]
pub enum DType {
    Isize,                  // signed integer
    Usize,                  // unsigned integer (for array indexing)
    F32,                    // float
    Ptr(Box<DType>),        // pointer for memory buffer, 値を渡す時は参照を渡す。
    Vec(Box<DType>, usize), // fixed size vector for SIMD, 値は渡す時にコピーされる
    Tuple(Vec<DType>),
    Unknown,
    // TODO: boolなどの追加
    // TODO: 将来的にf16とか対応させたい
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Isize(isize),
    Usize(usize),
    F32(f32),
}

// Conversion from numeric types to Literal
impl From<f32> for Literal {
    fn from(value: f32) -> Self {
        Literal::F32(value)
    }
}

impl From<isize> for Literal {
    fn from(value: isize) -> Self {
        Literal::Isize(value)
    }
}

impl From<usize> for Literal {
    fn from(value: usize) -> Self {
        Literal::Usize(value)
    }
}

impl Literal {
    /// Get the DType of this literal
    pub fn dtype(&self) -> DType {
        match self {
            Literal::F32(_) => DType::F32,
            Literal::Isize(_) => DType::Isize,
            Literal::Usize(_) => DType::Usize,
        }
    }
}

impl DType {
    /// Convert this type to a vector type with the given size
    pub fn to_vec(&self, size: usize) -> DType {
        DType::Vec(Box::new(self.clone()), size)
    }

    /// Convert this type to a pointer type
    pub fn to_ptr(&self) -> DType {
        DType::Ptr(Box::new(self.clone()))
    }

    /// If this is a Vec type, return the element type and size
    /// Returns None if this is not a Vec type
    pub fn from_vec(&self) -> Option<(&DType, usize)> {
        match self {
            DType::Vec(elem_type, size) => Some((elem_type.as_ref(), *size)),
            _ => None,
        }
    }

    /// If this is a Ptr type, return the pointee type
    /// Returns None if this is not a Ptr type
    pub fn from_ptr(&self) -> Option<&DType> {
        match self {
            DType::Ptr(pointee) => Some(pointee.as_ref()),
            _ => None,
        }
    }

    /// Get the element type if this is a Vec, otherwise return self
    pub fn element_type(&self) -> &DType {
        match self {
            DType::Vec(elem_type, _) => elem_type.as_ref(),
            _ => self,
        }
    }

    /// Get the pointee type if this is a Ptr, otherwise return self
    pub fn deref_type(&self) -> &DType {
        match self {
            DType::Ptr(pointee) => pointee.as_ref(),
            _ => self,
        }
    }

    /// Check if this is a Vec type
    pub fn is_vec(&self) -> bool {
        matches!(self, DType::Vec(_, _))
    }

    /// Check if this is a Ptr type
    pub fn is_ptr(&self) -> bool {
        matches!(self, DType::Ptr(_))
    }
}

impl AstNode {
    /// Get child nodes of this AST node
    pub fn children(&self) -> Vec<&AstNode> {
        match self {
            AstNode::Wildcard(_) | AstNode::Const(_) | AstNode::Var(_) => vec![],
            AstNode::Add(left, right)
            | AstNode::Mul(left, right)
            | AstNode::Max(left, right)
            | AstNode::Rem(left, right)
            | AstNode::Idiv(left, right) => vec![left.as_ref(), right.as_ref()],
            AstNode::Recip(operand)
            | AstNode::Sqrt(operand)
            | AstNode::Log2(operand)
            | AstNode::Exp2(operand)
            | AstNode::Sin(operand)
            | AstNode::Cast(operand, _) => vec![operand.as_ref()],
            AstNode::Load { ptr, offset, .. } => vec![ptr.as_ref(), offset.as_ref()],
            AstNode::Store { ptr, offset, value } => {
                vec![ptr.as_ref(), offset.as_ref(), value.as_ref()]
            }
            AstNode::Assign { value, .. } => vec![value.as_ref()],
            AstNode::Block { statements, .. } => {
                statements.iter().map(|node| node as &AstNode).collect()
            }
            AstNode::Range {
                start,
                step,
                stop,
                body,
                ..
            } => vec![start.as_ref(), step.as_ref(), stop.as_ref(), body.as_ref()],
            AstNode::Call { args, .. } => args.iter().map(|node| node as &AstNode).collect(),
            AstNode::Return { value } => vec![value.as_ref()],
        }
    }

    /// Apply a function to all child nodes and construct a new node with the results
    /// This is useful for recursive transformations of the AST
    pub fn map_children<F>(&self, f: &F) -> Self
    where
        F: Fn(&AstNode) -> AstNode,
    {
        match self {
            AstNode::Wildcard(_) | AstNode::Const(_) | AstNode::Var(_) => self.clone(),
            AstNode::Add(left, right) => AstNode::Add(Box::new(f(left)), Box::new(f(right))),
            AstNode::Mul(left, right) => AstNode::Mul(Box::new(f(left)), Box::new(f(right))),
            AstNode::Max(left, right) => AstNode::Max(Box::new(f(left)), Box::new(f(right))),
            AstNode::Rem(left, right) => AstNode::Rem(Box::new(f(left)), Box::new(f(right))),
            AstNode::Idiv(left, right) => AstNode::Idiv(Box::new(f(left)), Box::new(f(right))),
            AstNode::Recip(operand) => AstNode::Recip(Box::new(f(operand))),
            AstNode::Sqrt(operand) => AstNode::Sqrt(Box::new(f(operand))),
            AstNode::Log2(operand) => AstNode::Log2(Box::new(f(operand))),
            AstNode::Exp2(operand) => AstNode::Exp2(Box::new(f(operand))),
            AstNode::Sin(operand) => AstNode::Sin(Box::new(f(operand))),
            AstNode::Cast(operand, dtype) => AstNode::Cast(Box::new(f(operand)), dtype.clone()),
            AstNode::Load { ptr, offset, count } => AstNode::Load {
                ptr: Box::new(f(ptr)),
                offset: Box::new(f(offset)),
                count: *count,
            },
            AstNode::Store { ptr, offset, value } => AstNode::Store {
                ptr: Box::new(f(ptr)),
                offset: Box::new(f(offset)),
                value: Box::new(f(value)),
            },
            AstNode::Assign { var, value } => AstNode::Assign {
                var: var.clone(),
                value: Box::new(f(value)),
            },
            AstNode::Block { statements, scope } => AstNode::Block {
                statements: statements.iter().map(f).collect(),
                scope: scope.clone(),
            },
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => AstNode::Range {
                var: var.clone(),
                start: Box::new(f(start)),
                step: Box::new(f(step)),
                stop: Box::new(f(stop)),
                body: Box::new(f(body)),
            },
            AstNode::Call { name, args } => AstNode::Call {
                name: name.clone(),
                args: args.iter().map(f).collect(),
            },
            AstNode::Return { value } => AstNode::Return {
                value: Box::new(f(value)),
            },
        }
    }

    /// Recursively infer the type of this AST node by traversing child nodes
    pub fn infer_type(&self) -> DType {
        match self {
            AstNode::Wildcard(_) => DType::Unknown,
            AstNode::Const(lit) => lit.dtype(),
            AstNode::Cast(_, dtype) => dtype.clone(),
            AstNode::Var(_) => DType::Unknown, // 変数の型はコンテキストに依存

            // Binary operations - infer from operands
            AstNode::Add(left, right)
            | AstNode::Mul(left, right)
            | AstNode::Max(left, right)
            | AstNode::Rem(left, right)
            | AstNode::Idiv(left, right) => {
                let left_type = left.infer_type();
                let right_type = right.infer_type();

                // If types match, use that type
                if left_type == right_type {
                    left_type
                } else {
                    // Type mismatch - return Unknown
                    // In a more sophisticated implementation, we might do type promotion here
                    DType::Unknown
                }
            }

            // Unary operations that preserve type
            AstNode::Recip(operand) => operand.infer_type(),

            // Mathematical operations that typically return F32
            AstNode::Sqrt(_) | AstNode::Log2(_) | AstNode::Exp2(_) | AstNode::Sin(_) => DType::F32,

            // Memory operations
            AstNode::Load { ptr, count, .. } => {
                let ptr_type = ptr.infer_type();
                let pointee_type = ptr_type.deref_type().clone();
                if *count == 1 {
                    pointee_type // スカラー
                } else {
                    pointee_type.to_vec(*count) // Vec型
                }
            }
            AstNode::Store { .. } => DType::Tuple(vec![]), // Storeは値を返さない（unit型）

            // Assignment
            AstNode::Assign { value, .. } => value.infer_type(), // 代入された値の型を返す

            // Block - 最後の文の型を返す（空ならunit型）
            AstNode::Block { statements, .. } => statements
                .last()
                .map(|node| node.infer_type())
                .unwrap_or(DType::Tuple(vec![])),

            // Range - ループは値を返さない（unit型）
            AstNode::Range { .. } => DType::Tuple(vec![]),

            // Call - 関数呼び出しの型は関数定義から推論する必要がある
            // ここでは関数定義を参照できないので、Unknownを返す
            // Programコンテキストで適切に型チェックする
            AstNode::Call { .. } => DType::Unknown,

            // Return - 返す値の型を返す
            AstNode::Return { value } => value.infer_type(),
        }
    }

    /// Check if this AST node is valid within the given scope
    /// This performs local checks without traversing the entire tree
    pub fn check_scope(&self, scope: &Scope) -> Result<(), String> {
        match self {
            AstNode::Var(name) => {
                // 変数が読み取り可能かチェック
                scope.check_read(name)?;
                Ok(())
            }
            AstNode::Assign { var, value } => {
                // 値の型を推論
                let value_type = value.infer_type();
                // 書き込み可能かチェック
                scope.check_write(var, &value_type)?;
                // 値の部分も再帰的にチェック
                value.check_scope(scope)?;
                Ok(())
            }
            AstNode::Load { ptr, offset, .. } => {
                // ポインタとオフセットをチェック
                ptr.check_scope(scope)?;
                offset.check_scope(scope)?;
                Ok(())
            }
            AstNode::Store { ptr, offset, value } => {
                // 全ての子ノードをチェック
                ptr.check_scope(scope)?;
                offset.check_scope(scope)?;
                value.check_scope(scope)?;
                Ok(())
            }
            // 二項演算・単項演算は子ノードをチェック
            AstNode::Add(left, right)
            | AstNode::Mul(left, right)
            | AstNode::Max(left, right)
            | AstNode::Rem(left, right)
            | AstNode::Idiv(left, right) => {
                left.check_scope(scope)?;
                right.check_scope(scope)?;
                Ok(())
            }
            AstNode::Recip(operand)
            | AstNode::Sqrt(operand)
            | AstNode::Log2(operand)
            | AstNode::Exp2(operand)
            | AstNode::Sin(operand)
            | AstNode::Cast(operand, _) => {
                operand.check_scope(scope)?;
                Ok(())
            }
            // 定数とワイルドカードはスコープに依存しない
            AstNode::Const(_) | AstNode::Wildcard(_) => Ok(()),
            // Block - ブロック内の文をチェック
            AstNode::Block {
                statements,
                scope: block_scope,
            } => {
                for node in statements {
                    node.check_scope(block_scope)?;
                }
                Ok(())
            }
            // Range - ループ
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
            } => {
                // start, step, stopを外側のスコープでチェック
                start.check_scope(scope)?;
                step.check_scope(scope)?;
                stop.check_scope(scope)?;

                // bodyはBlockノードであるべきで、その中でループ変数がチェックされる
                body.check_scope(scope)?;

                // bodyがBlockの場合、ループ変数がそのスコープに宣言されているかチェック
                if let AstNode::Block {
                    scope: inner_scope, ..
                } = body.as_ref()
                {
                    inner_scope.check_read(var)?;
                }

                Ok(())
            }
            // Call - 引数のスコープチェック（関数名の存在確認はProgramレベルで行う）
            AstNode::Call { args, .. } => {
                for arg in args {
                    arg.check_scope(scope)?;
                }
                Ok(())
            }
            // Return - 返す値のスコープチェック
            AstNode::Return { value } => {
                value.check_scope(scope)?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::*;

    #[test]
    fn test_literal_from_f32() {
        let lit: Literal = 3.14f32.into();
        match lit {
            Literal::F32(v) => assert_eq!(v, 3.14),
            _ => panic!("Expected F32 literal"),
        }

        let lit = Literal::from(2.5f32);
        match lit {
            Literal::F32(v) => assert_eq!(v, 2.5),
            _ => panic!("Expected F32 literal"),
        }
    }

    #[test]
    fn test_literal_from_isize() {
        let lit: Literal = 42isize.into();
        match lit {
            Literal::Isize(v) => assert_eq!(v, 42),
            _ => panic!("Expected Isize literal"),
        }

        let lit = Literal::from(-10isize);
        match lit {
            Literal::Isize(v) => assert_eq!(v, -10),
            _ => panic!("Expected Isize literal"),
        }
    }

    #[test]
    fn test_literal_from_usize() {
        let lit: Literal = 100usize.into();
        match lit {
            Literal::Usize(v) => assert_eq!(v, 100),
            _ => panic!("Expected Usize literal"),
        }

        let lit = Literal::from(256usize);
        match lit {
            Literal::Usize(v) => assert_eq!(v, 256),
            _ => panic!("Expected Usize literal"),
        }
    }

    #[test]
    fn test_literal_dtype() {
        let f32_lit = Literal::F32(3.14);
        assert_eq!(f32_lit.dtype(), DType::F32);

        let isize_lit = Literal::Isize(42);
        assert_eq!(isize_lit.dtype(), DType::Isize);

        let usize_lit = Literal::Usize(100);
        assert_eq!(usize_lit.dtype(), DType::Usize);
    }

    #[test]
    fn test_children_const() {
        let node = AstNode::Const(3.14f32.into());
        let children = node.children();
        assert_eq!(children.len(), 0);
    }

    #[test]
    fn test_children_binary_ops() {
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let node = a + b;
        let children = node.children();
        assert_eq!(children.len(), 2);

        let node = AstNode::Const(3isize.into()) * AstNode::Const(4isize.into());
        let children = node.children();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_children_unary_ops() {
        let node = sqrt(AstNode::Const(4.0f32.into()));
        let children = node.children();
        assert_eq!(children.len(), 1);

        let node = sin(AstNode::Const(1.0f32.into()));
        let children = node.children();
        assert_eq!(children.len(), 1);

        let node = recip(AstNode::Const(2.0f32.into()));
        let children = node.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_children_cast() {
        let node = cast(AstNode::Const(3.14f32.into()), DType::Isize);
        let children = node.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_children_composite() {
        // (a + b) * c
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let c = AstNode::Const(3.0f32.into());
        let product = (a + b) * c;

        let children = product.children();
        assert_eq!(children.len(), 2);

        // The first child should be the sum node
        let sum_children = children[0].children();
        assert_eq!(sum_children.len(), 2);
    }

    #[test]
    fn test_infer_type_const() {
        let node = AstNode::Const(3.14f32.into());
        assert_eq!(node.infer_type(), DType::F32);

        let node = AstNode::Const(42isize.into());
        assert_eq!(node.infer_type(), DType::Isize);

        let node = AstNode::Const(100usize.into());
        assert_eq!(node.infer_type(), DType::Usize);
    }

    #[test]
    fn test_infer_type_binary_ops() {
        // Same types should return that type
        let node = AstNode::Const(1.0f32.into()) + AstNode::Const(2.0f32.into());
        assert_eq!(node.infer_type(), DType::F32);

        let node = AstNode::Const(3isize.into()) * AstNode::Const(4isize.into());
        assert_eq!(node.infer_type(), DType::Isize);

        // Mixed types should return Unknown
        let node = AstNode::Const(1.0f32.into()) + AstNode::Const(2isize.into());
        assert_eq!(node.infer_type(), DType::Unknown);
    }

    #[test]
    fn test_infer_type_unary_ops() {
        // Recip preserves type
        let node = recip(AstNode::Const(2.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        // Math operations return F32
        let node = sqrt(AstNode::Const(4.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        let node = sin(AstNode::Const(1.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        let node = log2(AstNode::Const(8.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);

        let node = exp2(AstNode::Const(3.0f32.into()));
        assert_eq!(node.infer_type(), DType::F32);
    }

    #[test]
    fn test_infer_type_cast() {
        let node = cast(AstNode::Const(3.14f32.into()), DType::Isize);
        assert_eq!(node.infer_type(), DType::Isize);

        let node = cast(AstNode::Const(42isize.into()), DType::F32);
        assert_eq!(node.infer_type(), DType::F32);
    }

    #[test]
    fn test_infer_type_composite() {
        // (a + b) * c where all are F32
        let a = AstNode::Const(1.0f32.into());
        let b = AstNode::Const(2.0f32.into());
        let c = AstNode::Const(3.0f32.into());
        let expr = (a + b) * c;
        assert_eq!(expr.infer_type(), DType::F32);

        // sqrt(a + b) where a, b are F32
        let a = AstNode::Const(4.0f32.into());
        let b = AstNode::Const(5.0f32.into());
        let expr = sqrt(a + b);
        assert_eq!(expr.infer_type(), DType::F32);

        // Complex expression with cast
        let a = AstNode::Const(10isize.into());
        let b = AstNode::Const(20isize.into());
        let casted = cast(a + b, DType::F32);
        let result = sqrt(casted);
        assert_eq!(result.infer_type(), DType::F32);
    }

    #[test]
    fn test_dtype_to_vec() {
        let base_type = DType::F32;
        let vec_type = base_type.to_vec(4);

        match vec_type {
            DType::Vec(elem_type, size) => {
                assert_eq!(*elem_type, DType::F32);
                assert_eq!(size, 4);
            }
            _ => panic!("Expected Vec type"),
        }
    }

    #[test]
    fn test_dtype_from_vec() {
        let vec_type = DType::F32.to_vec(8);

        let result = vec_type.from_vec();
        assert!(result.is_some());

        let (elem_type, size) = result.unwrap();
        assert_eq!(elem_type, &DType::F32);
        assert_eq!(size, 8);

        // Non-vec type should return None
        let scalar = DType::F32;
        assert!(scalar.from_vec().is_none());
    }

    #[test]
    fn test_dtype_to_ptr() {
        let base_type = DType::F32;
        let ptr_type = base_type.to_ptr();

        match ptr_type {
            DType::Ptr(pointee) => {
                assert_eq!(*pointee, DType::F32);
            }
            _ => panic!("Expected Ptr type"),
        }
    }

    #[test]
    fn test_dtype_from_ptr() {
        let ptr_type = DType::F32.to_ptr();

        let result = ptr_type.from_ptr();
        assert!(result.is_some());
        assert_eq!(result.unwrap(), &DType::F32);

        // Non-ptr type should return None
        let scalar = DType::F32;
        assert!(scalar.from_ptr().is_none());
    }

    #[test]
    fn test_dtype_element_type() {
        // Vec should return element type
        let vec_type = DType::F32.to_vec(4);
        assert_eq!(vec_type.element_type(), &DType::F32);

        // Non-vec should return self
        let scalar = DType::Isize;
        assert_eq!(scalar.element_type(), &DType::Isize);
    }

    #[test]
    fn test_dtype_deref_type() {
        // Ptr should return pointee type
        let ptr_type = DType::F32.to_ptr();
        assert_eq!(ptr_type.deref_type(), &DType::F32);

        // Non-ptr should return self
        let scalar = DType::Isize;
        assert_eq!(scalar.deref_type(), &DType::Isize);
    }

    #[test]
    fn test_dtype_is_vec() {
        let vec_type = DType::F32.to_vec(4);
        assert!(vec_type.is_vec());

        let scalar = DType::F32;
        assert!(!scalar.is_vec());
    }

    #[test]
    fn test_dtype_is_ptr() {
        let ptr_type = DType::F32.to_ptr();
        assert!(ptr_type.is_ptr());

        let scalar = DType::F32;
        assert!(!scalar.is_ptr());
    }

    #[test]
    fn test_dtype_nested_types() {
        // Vec of Ptr
        let ptr_type = DType::F32.to_ptr();
        let vec_of_ptr = ptr_type.to_vec(4);

        assert!(vec_of_ptr.is_vec());
        let (elem, size) = vec_of_ptr.from_vec().unwrap();
        assert_eq!(size, 4);
        assert!(elem.is_ptr());

        // Ptr to Vec
        let vec_type = DType::F32.to_vec(8);
        let ptr_to_vec = vec_type.to_ptr();

        assert!(ptr_to_vec.is_ptr());
        let pointee = ptr_to_vec.from_ptr().unwrap();
        assert!(pointee.is_vec());
    }

    #[test]
    fn test_var_node() {
        let var = AstNode::Var("x".to_string());
        assert_eq!(var.children().len(), 0);
        assert_eq!(var.infer_type(), DType::Unknown);
    }

    #[test]
    fn test_load_scalar() {
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Var("input0".to_string())),
            offset: Box::new(AstNode::Const(0usize.into())),
            count: 1,
        };

        // children should include ptr and offset
        let children = load.children();
        assert_eq!(children.len(), 2);

        // Type inference: Var returns Unknown, so deref_type returns Unknown
        // This test demonstrates the structure, actual type depends on context
        let inferred = load.infer_type();
        assert_eq!(inferred, DType::Unknown);
    }

    #[test]
    fn test_load_vector() {
        // Create a proper pointer type for testing
        let ptr_node = AstNode::Cast(
            Box::new(AstNode::Var("buffer".to_string())),
            DType::F32.to_ptr(),
        );

        let load = AstNode::Load {
            ptr: Box::new(ptr_node),
            offset: Box::new(AstNode::Const(0usize.into())),
            count: 4,
        };

        // Type should be Vec<F32, 4>
        let inferred = load.infer_type();
        assert_eq!(inferred, DType::F32.to_vec(4));

        // children should include ptr and offset
        let children = load.children();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_store() {
        let store = AstNode::Store {
            ptr: Box::new(AstNode::Var("output0".to_string())),
            offset: Box::new(AstNode::Const(0usize.into())),
            value: Box::new(AstNode::Const(3.14f32.into())),
        };

        // children should include ptr, offset, and value
        let children = store.children();
        assert_eq!(children.len(), 3);

        // Store returns unit type (empty tuple)
        let inferred = store.infer_type();
        assert_eq!(inferred, DType::Tuple(vec![]));
    }

    #[test]
    fn test_assign() {
        let assign = AstNode::Assign {
            var: "alu0".to_string(),
            value: Box::new(AstNode::Const(42isize.into())),
        };

        // children should include only value
        let children = assign.children();
        assert_eq!(children.len(), 1);

        // Assign returns the type of the value
        let inferred = assign.infer_type();
        assert_eq!(inferred, DType::Isize);
    }

    #[test]
    fn test_load_store_map_children() {
        let load = AstNode::Load {
            ptr: Box::new(AstNode::Const(1isize.into())),
            offset: Box::new(AstNode::Const(2isize.into())),
            count: 4,
        };

        // Map children: multiply each constant by 2
        let mapped = load.map_children(&|node| match node {
            AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n * 2)),
            _ => node.clone(),
        });

        if let AstNode::Load { ptr, offset, count } = mapped {
            assert_eq!(*ptr, AstNode::Const(Literal::Isize(2)));
            assert_eq!(*offset, AstNode::Const(Literal::Isize(4)));
            assert_eq!(count, 4);
        } else {
            panic!("Expected Load node");
        }
    }

    #[test]
    fn test_assign_map_children() {
        let assign = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Const(10isize.into())),
        };

        // Map children: increment constant
        let mapped = assign.map_children(&|node| match node {
            AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n + 1)),
            _ => node.clone(),
        });

        if let AstNode::Assign { var, value } = mapped {
            assert_eq!(var, "x");
            assert_eq!(*value, AstNode::Const(Literal::Isize(11)));
        } else {
            panic!("Expected Assign node");
        }
    }

    // Scope tests
    #[test]
    fn test_scope_declare() {
        let mut scope = Scope::new();

        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        assert!(scope.get("x").is_some());
        assert_eq!(scope.get("x").unwrap().dtype, DType::F32);
    }

    #[test]
    fn test_scope_duplicate_declare() {
        let mut scope = Scope::new();

        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let result = scope.declare(
            "x".to_string(),
            DType::Isize,
            Mutability::Mutable,
            AccessRegion::ThreadLocal,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_scope_check_read() {
        let mut scope = Scope::new();

        scope
            .declare(
                "input".to_string(),
                DType::F32.to_ptr(),
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        assert!(scope.check_read("input").is_ok());
        assert!(scope.check_read("undefined").is_err());
    }

    #[test]
    fn test_scope_check_write_immutable() {
        let mut scope = Scope::new();

        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let result = scope.check_write("x", &DType::F32);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("immutable"));
    }

    #[test]
    fn test_scope_check_write_mutable() {
        let mut scope = Scope::new();

        scope
            .declare(
                "output".to_string(),
                DType::F32,
                Mutability::Mutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        assert!(scope.check_write("output", &DType::F32).is_ok());
    }

    #[test]
    fn test_scope_check_write_type_mismatch() {
        let mut scope = Scope::new();

        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Mutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let result = scope.check_write("x", &DType::Isize);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Type mismatch"));
    }

    #[test]
    fn test_scope_parent_lookup() {
        let mut parent = Scope::new();
        parent
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        let child = Scope::with_parent(parent);

        // 親スコープの変数にアクセスできる
        assert!(child.check_read("x").is_ok());
    }

    #[test]
    fn test_scope_can_access_parallel_immutable() {
        let mut scope = Scope::new();

        scope
            .declare(
                "input1".to_string(),
                DType::F32.to_ptr(),
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        scope
            .declare(
                "input2".to_string(),
                DType::F32.to_ptr(),
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        // 両方immutableなので並列OK
        assert!(scope.can_access_parallel("input1", "input2"));
    }

    #[test]
    fn test_scope_can_access_parallel_thread_local() {
        let mut scope = Scope::new();

        scope
            .declare(
                "temp1".to_string(),
                DType::F32,
                Mutability::Mutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        scope
            .declare(
                "temp2".to_string(),
                DType::F32,
                Mutability::Mutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        // 両方ThreadLocalなので並列OK
        assert!(scope.can_access_parallel("temp1", "temp2"));
    }

    #[test]
    fn test_scope_can_access_parallel_sharded() {
        let mut scope = Scope::new();

        scope
            .declare(
                "output1".to_string(),
                DType::F32.to_ptr(),
                Mutability::Mutable,
                AccessRegion::ShardedBy(vec![0]),
            )
            .unwrap();

        scope
            .declare(
                "output2".to_string(),
                DType::F32.to_ptr(),
                Mutability::Mutable,
                AccessRegion::ShardedBy(vec![1]),
            )
            .unwrap();

        // 異なる軸でシャーディングされているので並列OK
        assert!(scope.can_access_parallel("output1", "output2"));
    }

    #[test]
    fn test_scope_cannot_access_parallel_mutable_shared() {
        let mut scope = Scope::new();

        scope
            .declare(
                "output".to_string(),
                DType::F32.to_ptr(),
                Mutability::Mutable,
                AccessRegion::Shared,
            )
            .unwrap();

        scope
            .declare(
                "input".to_string(),
                DType::F32.to_ptr(),
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        // 片方がMutableでSharedなので並列NG
        assert!(!scope.can_access_parallel("output", "input"));
    }

    #[test]
    fn test_check_scope_var() {
        let mut scope = Scope::new();

        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let var_node = AstNode::Var("x".to_string());
        assert!(var_node.check_scope(&scope).is_ok());

        let undefined_var = AstNode::Var("undefined".to_string());
        assert!(undefined_var.check_scope(&scope).is_err());
    }

    #[test]
    fn test_check_scope_assign() {
        let mut scope = Scope::new();

        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Mutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let assign_node = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Const(3.14f32.into())),
        };

        assert!(assign_node.check_scope(&scope).is_ok());
    }

    #[test]
    fn test_check_scope_assign_immutable() {
        let mut scope = Scope::new();

        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let assign_node = AstNode::Assign {
            var: "x".to_string(),
            value: Box::new(AstNode::Const(3.14f32.into())),
        };

        let result = assign_node.check_scope(&scope);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("immutable"));
    }

    #[test]
    fn test_check_scope_complex_expression() {
        let mut scope = Scope::new();

        scope
            .declare(
                "input".to_string(),
                DType::F32.to_ptr(),
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        scope
            .declare(
                "output".to_string(),
                DType::F32.to_ptr(),
                Mutability::Mutable,
                AccessRegion::ShardedBy(vec![0]),
            )
            .unwrap();

        scope
            .declare(
                "i".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        // output[i] = input[i] * 2.0
        let expr = AstNode::Store {
            ptr: Box::new(AstNode::Var("output".to_string())),
            offset: Box::new(AstNode::Var("i".to_string())),
            value: Box::new(AstNode::Mul(
                Box::new(AstNode::Load {
                    ptr: Box::new(AstNode::Var("input".to_string())),
                    offset: Box::new(AstNode::Var("i".to_string())),
                    count: 1,
                }),
                Box::new(AstNode::Const(2.0f32.into())),
            )),
        };

        assert!(expr.check_scope(&scope).is_ok());
    }

    // Range tests
    #[test]
    fn test_range_basic() {
        let mut scope = Scope::new();
        scope
            .declare(
                "i".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0usize.into())),
            step: Box::new(AstNode::Const(1usize.into())),
            stop: Box::new(AstNode::Const(10usize.into())),
            body: Box::new(AstNode::Block {
                statements: vec![],
                scope: Box::new(scope),
            }),
        };

        // Rangeはunit型を返す
        assert_eq!(range.infer_type(), DType::Tuple(vec![]));
    }

    #[test]
    fn test_range_children() {
        let mut scope = Scope::new();
        scope
            .declare(
                "i".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0usize.into())),
            step: Box::new(AstNode::Const(1usize.into())),
            stop: Box::new(AstNode::Const(10usize.into())),
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Const(1.0f32.into()), AstNode::Const(2.0f32.into())],
                scope: Box::new(scope),
            }),
        };

        let children = range.children();
        // start, step, stop, body = 4個
        assert_eq!(children.len(), 4);
    }

    #[test]
    fn test_range_with_scope() {
        let mut outer_scope = Scope::new();
        outer_scope
            .declare(
                "N".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        outer_scope
            .declare(
                "input".to_string(),
                DType::F32.to_ptr(),
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        outer_scope
            .declare(
                "output".to_string(),
                DType::F32.to_ptr(),
                Mutability::Mutable,
                AccessRegion::ShardedBy(vec![0]),
            )
            .unwrap();

        // ループ内のスコープ
        let mut loop_scope = Scope::with_parent(outer_scope.clone());
        loop_scope
            .declare(
                "i".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        // for i in 0..N: output[i] = input[i] * 2
        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0usize.into())),
            step: Box::new(AstNode::Const(1usize.into())),
            stop: Box::new(AstNode::Var("N".to_string())),
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Store {
                    ptr: Box::new(AstNode::Var("output".to_string())),
                    offset: Box::new(AstNode::Var("i".to_string())),
                    value: Box::new(AstNode::Mul(
                        Box::new(AstNode::Load {
                            ptr: Box::new(AstNode::Var("input".to_string())),
                            offset: Box::new(AstNode::Var("i".to_string())),
                            count: 1,
                        }),
                        Box::new(AstNode::Const(2.0f32.into())),
                    )),
                }],
                scope: Box::new(loop_scope),
            }),
        };

        // スコープチェック
        assert!(range.check_scope(&outer_scope).is_ok());
    }

    #[test]
    fn test_range_scope_check_undefined_loop_var() {
        let mut outer_scope = Scope::new();
        outer_scope
            .declare(
                "N".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        // ループスコープにループ変数を宣言しない
        let loop_scope = Scope::with_parent(outer_scope.clone());

        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0usize.into())),
            step: Box::new(AstNode::Const(1usize.into())),
            stop: Box::new(AstNode::Var("N".to_string())),
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Var("i".to_string())],
                scope: Box::new(loop_scope),
            }),
        };

        // ループ変数が宣言されていないのでエラー
        let result = range.check_scope(&outer_scope);
        assert!(result.is_err());
    }

    #[test]
    fn test_range_nested() {
        let mut outer_scope = Scope::new();
        outer_scope
            .declare(
                "N".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::Shared,
            )
            .unwrap();

        // 外側のループスコープ
        let mut outer_loop_scope = Scope::with_parent(outer_scope.clone());
        outer_loop_scope
            .declare(
                "i".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        // 内側のループスコープ
        let mut inner_loop_scope = Scope::with_parent(outer_loop_scope.clone());
        inner_loop_scope
            .declare(
                "j".to_string(),
                DType::Usize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        // for j in 0..N: use i and j
        let inner_range = AstNode::Range {
            var: "j".to_string(),
            start: Box::new(AstNode::Const(0usize.into())),
            step: Box::new(AstNode::Const(1usize.into())),
            stop: Box::new(AstNode::Var("N".to_string())),
            body: Box::new(AstNode::Block {
                statements: vec![AstNode::Var("i".to_string()), AstNode::Var("j".to_string())],
                scope: Box::new(inner_loop_scope),
            }),
        };

        // for i in 0..N: ...
        let outer_range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(AstNode::Const(0usize.into())),
            step: Box::new(AstNode::Const(1usize.into())),
            stop: Box::new(AstNode::Var("N".to_string())),
            body: Box::new(AstNode::Block {
                statements: vec![inner_range],
                scope: Box::new(outer_loop_scope),
            }),
        };

        // ネストしたループのスコープチェック
        assert!(outer_range.check_scope(&outer_scope).is_ok());
    }

    // Block tests
    #[test]
    fn test_block_basic() {
        let mut scope = Scope::new();
        scope
            .declare(
                "x".to_string(),
                DType::Isize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let block = AstNode::Block {
            statements: vec![
                AstNode::Var("x".to_string()),
                AstNode::Const(42isize.into()),
            ],
            scope: Box::new(scope),
        };

        // Blockは最後の文の型を返す
        assert_eq!(block.infer_type(), DType::Isize);
    }

    #[test]
    fn test_block_children() {
        let block = AstNode::Block {
            statements: vec![AstNode::Const(1isize.into()), AstNode::Const(2isize.into())],
            scope: Box::new(Scope::new()),
        };

        let children = block.children();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_block_check_scope() {
        let mut scope = Scope::new();
        scope
            .declare(
                "a".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let block = AstNode::Block {
            statements: vec![var("a"), AstNode::Const(1.0f32.into())],
            scope: Box::new(scope),
        };

        // スコープチェックが成功するはず
        let outer_scope = Scope::new();
        assert!(block.check_scope(&outer_scope).is_ok());
    }

    // Call tests
    #[test]
    fn test_call_children() {
        let call = AstNode::Call {
            name: "add".to_string(),
            args: vec![AstNode::Const(1isize.into()), AstNode::Const(2isize.into())],
        };

        let children = call.children();
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_call_map_children() {
        let call = AstNode::Call {
            name: "mul".to_string(),
            args: vec![AstNode::Const(3isize.into()), AstNode::Const(4isize.into())],
        };

        let mapped = call.map_children(&|node| match node {
            AstNode::Const(Literal::Isize(n)) => AstNode::Const(Literal::Isize(n * 2)),
            _ => node.clone(),
        });

        if let AstNode::Call { name, args } = mapped {
            assert_eq!(name, "mul");
            assert_eq!(args.len(), 2);
            assert_eq!(args[0], AstNode::Const(Literal::Isize(6)));
            assert_eq!(args[1], AstNode::Const(Literal::Isize(8)));
        } else {
            panic!("Expected Call node");
        }
    }

    #[test]
    fn test_call_check_scope() {
        let mut scope = Scope::new();
        scope
            .declare(
                "x".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();
        scope
            .declare(
                "y".to_string(),
                DType::F32,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let call = AstNode::Call {
            name: "add".to_string(),
            args: vec![var("x"), var("y")],
        };

        assert!(call.check_scope(&scope).is_ok());
    }

    // Return tests
    #[test]
    fn test_return_children() {
        let ret = AstNode::Return {
            value: Box::new(AstNode::Const(42isize.into())),
        };

        let children = ret.children();
        assert_eq!(children.len(), 1);
    }

    #[test]
    fn test_return_infer_type() {
        let ret = AstNode::Return {
            value: Box::new(AstNode::Const(3.14f32.into())),
        };

        assert_eq!(ret.infer_type(), DType::F32);
    }

    #[test]
    fn test_return_check_scope() {
        let mut scope = Scope::new();
        scope
            .declare(
                "result".to_string(),
                DType::Isize,
                Mutability::Immutable,
                AccessRegion::ThreadLocal,
            )
            .unwrap();

        let ret = AstNode::Return {
            value: Box::new(var("result")),
        };

        assert!(ret.check_scope(&scope).is_ok());
    }

    // Function tests
    #[test]
    fn test_function_new() {
        let params = vec![("a".to_string(), DType::F32), ("b".to_string(), DType::F32)];
        let return_type = DType::F32;
        let body = vec![AstNode::Return {
            value: Box::new(var("a") + var("b")),
        }];

        let func = Function::new(params, return_type, body);
        assert!(func.is_ok());

        let func = func.unwrap();
        assert_eq!(func.params.len(), 2);
        assert_eq!(func.return_type, DType::F32);

        // bodyはBlock nodeになっている
        match &*func.body {
            AstNode::Block { statements, .. } => {
                assert_eq!(statements.len(), 1);
            }
            _ => panic!("Expected Block node for function body"),
        }
    }

    #[test]
    fn test_function_check_body() {
        let params = vec![("x".to_string(), DType::Isize)];
        let return_type = DType::Isize;
        let body = vec![AstNode::Return {
            value: Box::new(var("x") * AstNode::Const(2isize.into())),
        }];

        let func = Function::new(params, return_type, body).unwrap();
        assert!(func.check_body().is_ok());
    }

    #[test]
    fn test_function_infer_return_type() {
        let params = vec![];
        let return_type = DType::F32;
        let body = vec![AstNode::Return {
            value: Box::new(AstNode::Const(1.0f32.into())),
        }];

        let func = Function::new(params, return_type, body).unwrap();
        assert_eq!(func.infer_return_type(), DType::F32);
    }

    // Program tests
    #[test]
    fn test_program_new() {
        let program = Program::new("main".to_string());
        assert_eq!(program.entry_point, "main");
        assert_eq!(program.functions.len(), 0);
    }

    #[test]
    fn test_program_add_function() {
        let mut program = Program::new("main".to_string());

        let func = Function::new(vec![], DType::Tuple(vec![]), vec![]).unwrap();
        assert!(program.add_function("main".to_string(), func).is_ok());
        assert_eq!(program.functions.len(), 1);
    }

    #[test]
    fn test_program_get_function() {
        let mut program = Program::new("main".to_string());
        let func = Function::new(vec![], DType::Tuple(vec![]), vec![]).unwrap();
        program.add_function("main".to_string(), func).unwrap();

        assert!(program.get_function("main").is_some());
        assert!(program.get_function("nonexistent").is_none());
    }

    #[test]
    fn test_program_validate() {
        let mut program = Program::new("main".to_string());

        // エントリーポイントがない場合はエラー
        assert!(program.validate().is_err());

        // エントリーポイントを追加
        let func = Function::new(vec![], DType::Tuple(vec![]), vec![]).unwrap();
        program.add_function("main".to_string(), func).unwrap();

        // 成功するはず
        assert!(program.validate().is_ok());
    }

    #[test]
    fn test_program_with_function_call() {
        let mut program = Program::new("main".to_string());

        // helper関数: double(x) = x * 2
        let double_func = Function::new(
            vec![("x".to_string(), DType::Isize)],
            DType::Isize,
            vec![AstNode::Return {
                value: Box::new(var("x") * AstNode::Const(2isize.into())),
            }],
        )
        .unwrap();
        program
            .add_function("double".to_string(), double_func)
            .unwrap();

        // main関数: Call double(5)
        let main_func = Function::new(
            vec![],
            DType::Isize,
            vec![AstNode::Call {
                name: "double".to_string(),
                args: vec![AstNode::Const(5isize.into())],
            }],
        )
        .unwrap();
        program.add_function("main".to_string(), main_func).unwrap();

        // プログラムの検証
        assert!(program.validate().is_ok());
        assert!(program.has_function("double"));
        assert!(program.has_function("main"));
    }
}

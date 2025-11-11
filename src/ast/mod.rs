// Operator overloading for AstNode
pub mod ops;
// Helper functions for constructing AST nodes
pub mod helper;
pub mod pat;
pub mod renderer;

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

    // Barrier - 同期バリア（並列実行の同期点）
    Barrier,

    // Function definition - 関数定義
    Function {
        name: Option<String>, // 関数名（Program内ではこのフィールドは使用されず、匿名関数も可能）
        params: Vec<VarDecl>, // 引数リスト
        return_type: DType,   // 返り値の型
        body: Box<AstNode>,   // 関数本体（通常はBlock）
        kind: FunctionKind,   // 関数の種類（Normal or Kernel）
    },

    // Program - プログラム全体
    Program {
        functions: Vec<AstNode>, // AstNode::Function のリスト
        entry_point: String,     // エントリーポイントの関数名
    },
}

/// 関数の種類
#[derive(Clone, Debug, PartialEq)]
pub enum FunctionKind {
    /// 通常の関数（CPU上で逐次実行）
    Normal,
    /// GPUカーネル（並列実行される）
    /// 内部の数値は並列実行の次元数（1D, 2D, 3Dなど）
    Kernel(usize),
}

/// 関数定義
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub params: Vec<VarDecl>, // 引数リスト
    pub return_type: DType,   // 返り値の型
    pub body: Box<AstNode>,   // 関数本体（Blockノード）
    pub kind: FunctionKind,   // 関数の種類（通常 or カーネル）
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
        kind: FunctionKind,
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
                    param.region.clone(),
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
            kind,
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
            name.clone(),
            VarDecl {
                name,
                dtype,
                mutability,
                region,
                kind: VarKind::Normal, // 通常の変数宣言
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
    pub name: String,
    pub dtype: DType,
    pub mutability: Mutability,
    pub region: AccessRegion,
    pub kind: VarKind, // 変数の種類
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

/// 変数の種類
#[derive(Clone, Debug, PartialEq)]
pub enum VarKind {
    Normal,           // 通常の変数/引数
    ThreadId(usize),  // スレッドID（軸番号）
    GroupId(usize),   // グループID（軸番号）
    GroupSize(usize), // グループサイズ（軸番号）
    GridSize(usize),  // グリッドサイズ（軸番号）
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
    /// バイト単位でのサイズを取得
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::Isize => std::mem::size_of::<isize>(),
            DType::Usize => std::mem::size_of::<usize>(),
            DType::F32 => std::mem::size_of::<f32>(),
            DType::Ptr(_) => std::mem::size_of::<usize>(), // ポインタはusizeと同じサイズ
            DType::Vec(elem_type, size) => elem_type.size_in_bytes() * size,
            DType::Tuple(types) => types.iter().map(|t| t.size_in_bytes()).sum(),
            DType::Unknown => 0,
        }
    }

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
            AstNode::Barrier => vec![],
            AstNode::Function { body, .. } => vec![body.as_ref()],
            AstNode::Program { functions, .. } => {
                functions.iter().map(|node| node as &AstNode).collect()
            }
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
            AstNode::Barrier => AstNode::Barrier,
            AstNode::Function {
                name,
                params,
                return_type,
                body,
                kind,
            } => AstNode::Function {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(f(body)),
                kind: kind.clone(),
            },
            AstNode::Program {
                functions,
                entry_point,
            } => AstNode::Program {
                functions: functions.iter().map(f).collect(),
                entry_point: entry_point.clone(),
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

            // Barrier - 同期バリアは値を返さない（unit型）
            AstNode::Barrier => DType::Tuple(vec![]),

            // Function - 関数自体の型は返り値の型
            AstNode::Function { return_type, .. } => return_type.clone(),

            // Program - プログラム全体の型はエントリーポイントの返り値の型
            AstNode::Program { .. } => {
                if let Some(entry) = self.get_entry() {
                    entry.infer_type()
                } else {
                    DType::Unknown
                }
            }
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
            // Barrier - 同期バリアはスコープに依存しない
            AstNode::Barrier => Ok(()),
            // Function - 関数本体のスコープチェック（パラメータは関数のスコープに含まれる）
            AstNode::Function { body, .. } => body.check_scope(scope),
            // Program - 各関数のスコープチェック
            AstNode::Program { functions, .. } => {
                for func in functions {
                    func.check_scope(scope)?;
                }
                Ok(())
            }
        }
    }

    /// Get a function from a Program by name
    ///
    /// Returns None if this is not a Program or if the function is not found
    pub fn get_function(&self, name: &str) -> Option<&AstNode> {
        match self {
            AstNode::Program { functions, .. } => functions
                .iter()
                .find(|f| matches!(f, AstNode::Function { name: Some(n), .. } if n == name)),
            _ => None,
        }
    }

    /// Get the entry point function from a Program
    ///
    /// Returns None if this is not a Program or if the entry point is not found
    pub fn get_entry(&self) -> Option<&AstNode> {
        match self {
            AstNode::Program { entry_point, .. } => self.get_function(entry_point),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;

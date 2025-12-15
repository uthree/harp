//! スコープ管理と変数宣言

use super::types::DType;
use indexmap::IndexMap;

/// 変数宣言
#[derive(Clone, Debug, PartialEq)]
pub struct VarDecl {
    pub name: String,
    pub dtype: DType,
    pub mutability: Mutability,
    pub kind: VarKind, // 変数の種類
}

/// 変数の可変性
#[derive(Clone, Debug, PartialEq)]
pub enum Mutability {
    Immutable, // 読み取り専用（複数スレッドから安全にアクセス可）
    Mutable,   // 書き込み可能（単一スレッドのみ）
}

/// 変数の種類
///
/// GPUカーネルの並列化には GroupId + LocalId を使用します。
/// グローバルインデックスは `group_id * local_size + local_id` で計算できます。
#[derive(Clone, Debug, PartialEq)]
pub enum VarKind {
    Normal,           // 通常の変数/引数
    GroupId(usize),   // グループID（軸番号）- get_group_id(axis)
    LocalId(usize),   // ローカルスレッドID（軸番号）- get_local_id(axis)
    GroupSize(usize), // グループサイズ（軸番号）- get_local_size(axis)
    GridSize(usize),  // グリッドサイズ（軸番号）- get_global_size(axis)
}

/// スコープ管理（変数アクセス制御）
#[derive(Clone, Debug, PartialEq)]
pub struct Scope {
    // 変数名 -> 宣言情報（IndexMapを使用して宣言順序を保持）
    variables: IndexMap<String, VarDecl>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    /// Create a new empty scope
    pub fn new() -> Self {
        Scope {
            variables: IndexMap::new(),
            parent: None,
        }
    }

    /// Create a new scope with a parent
    pub fn with_parent(parent: Scope) -> Self {
        Scope {
            variables: IndexMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    /// Declare a variable in this scope
    pub fn declare(
        &mut self,
        name: String,
        dtype: DType,
        mutability: Mutability,
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

    /// Get a reference to a variable declaration
    pub fn get(&self, name: &str) -> Option<&VarDecl> {
        self.lookup(name)
    }

    /// Get an iterator over local variables declared in this scope (not including parent scopes)
    pub fn local_variables(&self) -> impl Iterator<Item = &VarDecl> {
        self.variables.values()
    }
}

impl Default for Scope {
    fn default() -> Self {
        Self::new()
    }
}

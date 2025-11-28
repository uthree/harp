// Operator overloading for AstNode
pub mod ops;
// Helper functions for constructing AST nodes
pub mod helper;
pub mod pat;
pub mod program;
pub mod renderer;
pub mod scope;
pub mod types;

// Re-export types for backwards compatibility
pub use program::{Function, FunctionKind, Program};
pub use scope::{Mutability, Scope, VarDecl, VarKind};
pub use types::{DType, Literal};

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

    // Random number generation - 乱数生成
    /// 0〜1の一様乱数を生成（F32型）
    Rand,

    // bitwise operations - ビット演算
    BitwiseAnd(Box<AstNode>, Box<AstNode>),
    BitwiseOr(Box<AstNode>, Box<AstNode>),
    BitwiseXor(Box<AstNode>, Box<AstNode>),
    BitwiseNot(Box<AstNode>),
    LeftShift(Box<AstNode>, Box<AstNode>),
    RightShift(Box<AstNode>, Box<AstNode>),

    // Variables - 変数
    Var(String),

    // Memory operations - メモリ操作（バッファー用）
    Load {
        ptr: Box<AstNode>,    // ポインタ（Ptr<T>型の式）
        offset: Box<AstNode>, // オフセット（Usize型の式）
        count: usize,         // 読み込む要素数（コンパイル時定数、1ならスカラー）
        dtype: DType,         // 読み込む要素の型
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

    // Memory allocation - メモリ確保
    Allocate {
        dtype: Box<DType>,  // 確保する要素の型
        size: Box<AstNode>, // 確保する要素数
    },
    Deallocate {
        ptr: Box<AstNode>, // 解放するポインタ
    },

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

impl AstNode {
    /// Get child nodes of this AST node
    pub fn children(&self) -> Vec<&AstNode> {
        match self {
            AstNode::Wildcard(_) | AstNode::Const(_) | AstNode::Var(_) | AstNode::Rand => vec![],
            AstNode::Add(left, right)
            | AstNode::Mul(left, right)
            | AstNode::Max(left, right)
            | AstNode::Rem(left, right)
            | AstNode::Idiv(left, right)
            | AstNode::BitwiseAnd(left, right)
            | AstNode::BitwiseOr(left, right)
            | AstNode::BitwiseXor(left, right)
            | AstNode::LeftShift(left, right)
            | AstNode::RightShift(left, right) => vec![left.as_ref(), right.as_ref()],
            AstNode::Recip(operand)
            | AstNode::Sqrt(operand)
            | AstNode::Log2(operand)
            | AstNode::Exp2(operand)
            | AstNode::Sin(operand)
            | AstNode::BitwiseNot(operand)
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
            AstNode::Allocate { size, .. } => vec![size.as_ref()],
            AstNode::Deallocate { ptr } => vec![ptr.as_ref()],
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
            AstNode::Wildcard(_) | AstNode::Const(_) | AstNode::Var(_) | AstNode::Rand => {
                self.clone()
            }
            AstNode::Add(left, right) => AstNode::Add(Box::new(f(left)), Box::new(f(right))),
            AstNode::Mul(left, right) => AstNode::Mul(Box::new(f(left)), Box::new(f(right))),
            AstNode::Max(left, right) => AstNode::Max(Box::new(f(left)), Box::new(f(right))),
            AstNode::Rem(left, right) => AstNode::Rem(Box::new(f(left)), Box::new(f(right))),
            AstNode::Idiv(left, right) => AstNode::Idiv(Box::new(f(left)), Box::new(f(right))),
            AstNode::BitwiseAnd(left, right) => {
                AstNode::BitwiseAnd(Box::new(f(left)), Box::new(f(right)))
            }
            AstNode::BitwiseOr(left, right) => {
                AstNode::BitwiseOr(Box::new(f(left)), Box::new(f(right)))
            }
            AstNode::BitwiseXor(left, right) => {
                AstNode::BitwiseXor(Box::new(f(left)), Box::new(f(right)))
            }
            AstNode::LeftShift(left, right) => {
                AstNode::LeftShift(Box::new(f(left)), Box::new(f(right)))
            }
            AstNode::RightShift(left, right) => {
                AstNode::RightShift(Box::new(f(left)), Box::new(f(right)))
            }
            AstNode::Recip(operand) => AstNode::Recip(Box::new(f(operand))),
            AstNode::Sqrt(operand) => AstNode::Sqrt(Box::new(f(operand))),
            AstNode::Log2(operand) => AstNode::Log2(Box::new(f(operand))),
            AstNode::Exp2(operand) => AstNode::Exp2(Box::new(f(operand))),
            AstNode::Sin(operand) => AstNode::Sin(Box::new(f(operand))),
            AstNode::BitwiseNot(operand) => AstNode::BitwiseNot(Box::new(f(operand))),
            AstNode::Cast(operand, dtype) => AstNode::Cast(Box::new(f(operand)), dtype.clone()),
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => AstNode::Load {
                ptr: Box::new(f(ptr)),
                offset: Box::new(f(offset)),
                count: *count,
                dtype: dtype.clone(),
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
            AstNode::Allocate { dtype, size } => AstNode::Allocate {
                dtype: dtype.clone(),
                size: Box::new(f(size)),
            },
            AstNode::Deallocate { ptr } => AstNode::Deallocate {
                ptr: Box::new(f(ptr)),
            },
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
            | AstNode::Idiv(left, right)
            | AstNode::BitwiseAnd(left, right)
            | AstNode::BitwiseOr(left, right)
            | AstNode::BitwiseXor(left, right)
            | AstNode::LeftShift(left, right)
            | AstNode::RightShift(left, right) => {
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
            AstNode::Recip(operand) | AstNode::BitwiseNot(operand) => operand.infer_type(),

            // Mathematical operations that typically return F32
            AstNode::Sqrt(_) | AstNode::Log2(_) | AstNode::Exp2(_) | AstNode::Sin(_) => DType::F32,

            // Random number generation returns F32
            AstNode::Rand => DType::F32,

            // Memory operations
            AstNode::Load { dtype, .. } => dtype.clone(),
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

            // Allocate - ポインタを返す
            AstNode::Allocate { dtype, .. } => DType::Ptr(dtype.clone()),

            // Deallocate - 値を返さない（unit型）
            AstNode::Deallocate { .. } => DType::Tuple(vec![]),

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
            | AstNode::Idiv(left, right)
            | AstNode::BitwiseAnd(left, right)
            | AstNode::BitwiseOr(left, right)
            | AstNode::BitwiseXor(left, right)
            | AstNode::LeftShift(left, right)
            | AstNode::RightShift(left, right) => {
                left.check_scope(scope)?;
                right.check_scope(scope)?;
                Ok(())
            }
            AstNode::Recip(operand)
            | AstNode::Sqrt(operand)
            | AstNode::Log2(operand)
            | AstNode::Exp2(operand)
            | AstNode::Sin(operand)
            | AstNode::BitwiseNot(operand)
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
            // Rand - 乱数生成はスコープに依存しない
            AstNode::Rand => Ok(()),
            // Allocate - サイズ式のスコープチェック
            AstNode::Allocate { size, .. } => {
                size.check_scope(scope)?;
                Ok(())
            }
            // Deallocate - ポインタのスコープチェック
            AstNode::Deallocate { ptr } => {
                ptr.check_scope(scope)?;
                Ok(())
            }
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

    /// Substitute Wildcard nodes with provided mappings
    ///
    /// This recursively traverses the AST and replaces Wildcard("name") with the
    /// corresponding AstNode from the mapping. Used for fused operations where
    /// Wildcard("0"), Wildcard("1"), etc. represent input sources.
    ///
    /// # Example
    /// ```
    /// use harp::ast::{AstNode, helper::*};
    /// use std::collections::HashMap;
    ///
    /// // Create an expression: Wildcard("0") + Wildcard("1")
    /// let expr = wildcard("0") + wildcard("1");
    ///
    /// // Create mappings
    /// let mut mappings = HashMap::new();
    /// mappings.insert("0".to_string(), const_f32(2.0));
    /// mappings.insert("1".to_string(), const_f32(3.0));
    ///
    /// // Substitute wildcards
    /// let result = expr.substitute(&mappings);
    ///
    /// // Result is: const_f32(2.0) + const_f32(3.0)
    /// ```
    pub fn substitute(&self, mappings: &std::collections::HashMap<String, AstNode>) -> AstNode {
        match self {
            AstNode::Wildcard(name) => {
                if let Some(replacement) = mappings.get(name) {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            _ => self.map_children(&|child| child.substitute(mappings)),
        }
    }

    /// 変数ノード（Var）を指定されたマッピングで置換します。
    ///
    /// マッピングに含まれる変数名に一致するVarノードを、対応するAstNodeに置換します。
    /// Wildcardも同時に置換されます。
    pub fn substitute_vars(
        &self,
        mappings: &std::collections::HashMap<String, AstNode>,
    ) -> AstNode {
        match self {
            AstNode::Var(name) => {
                if let Some(replacement) = mappings.get(name) {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            AstNode::Wildcard(name) => {
                if let Some(replacement) = mappings.get(name) {
                    replacement.clone()
                } else {
                    self.clone()
                }
            }
            _ => self.map_children(&|child| child.substitute_vars(mappings)),
        }
    }
}

#[cfg(test)]
mod tests;

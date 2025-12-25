// Operator overloading for AstNode
pub mod ops;
// Helper functions for constructing AST nodes
pub mod helper;
pub mod pat;
pub mod renderer;
pub mod scope;
pub mod types;

// Re-export commonly used types
pub use scope::{Mutability, Scope, VarDecl, VarKind};
pub use types::{DType, Literal};

use crate::tensor::shape::Expr;

/// カーネル呼び出し情報（AST層）
///
/// AstNode::Program内でカーネルの実行順序と呼び出し情報を管理します。
/// grid_size/local_sizeはExprで表現され、simplify()で評価できます。
#[derive(Clone, Debug, PartialEq)]
pub struct AstKernelCallInfo {
    /// カーネル名（AstNode::Kernel.nameに対応）
    pub kernel_name: String,
    /// 入力バッファ名のリスト
    pub inputs: Vec<String>,
    /// 出力バッファ名のリスト
    pub outputs: Vec<String>,
    /// グリッドサイズ（ワークアイテム数）
    pub grid_size: [Expr; 3],
    /// ローカルサイズ（スレッドグループサイズ）
    pub local_size: [Expr; 3],
}

impl AstKernelCallInfo {
    /// 新しいAstKernelCallInfoを作成
    pub fn new(
        kernel_name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
        grid_size: [Expr; 3],
        local_size: [Expr; 3],
    ) -> Self {
        Self {
            kernel_name,
            inputs,
            outputs,
            grid_size,
            local_size,
        }
    }

    /// デフォルトのdispatchサイズで作成
    pub fn with_default_dispatch(
        kernel_name: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    ) -> Self {
        Self {
            kernel_name,
            inputs,
            outputs,
            grid_size: [Expr::Const(1), Expr::Const(1), Expr::Const(1)],
            local_size: [Expr::Const(1), Expr::Const(1), Expr::Const(1)],
        }
    }
}

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
    Floor(Box<AstNode>),
    Cast(Box<AstNode>, DType),

    /// Fused Multiply-Add: fma(a, b, c) = a * b + c
    /// 1命令で実行されるため、精度が向上し高速
    Fma {
        a: Box<AstNode>,
        b: Box<AstNode>,
        c: Box<AstNode>,
    },

    /// アトミック加算（並列Reduce用）
    /// グローバルメモリ上の値にアトミックに加算し、古い値を返す
    AtomicAdd {
        ptr: Box<AstNode>,    // グローバルメモリへのポインタ
        offset: Box<AstNode>, // オフセット
        value: Box<AstNode>,  // 加算する値
        dtype: DType,         // 値の型
    },

    /// アトミック最大値（並列Reduce用）
    /// グローバルメモリ上の値とのアトミックmax、古い値を返す
    AtomicMax {
        ptr: Box<AstNode>,
        offset: Box<AstNode>,
        value: Box<AstNode>,
        dtype: DType,
    },

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

    // Comparison operations - 比較演算（Bool型を返す）
    Lt(Box<AstNode>, Box<AstNode>), // <
    Le(Box<AstNode>, Box<AstNode>), // <=
    Gt(Box<AstNode>, Box<AstNode>), // >
    Ge(Box<AstNode>, Box<AstNode>), // >=
    Eq(Box<AstNode>, Box<AstNode>), // ==
    Ne(Box<AstNode>, Box<AstNode>), // !=

    // Control flow - 制御構文
    Range {
        var: String,         // ループ変数名
        start: Box<AstNode>, // 開始値
        step: Box<AstNode>,  // ステップ
        stop: Box<AstNode>,  // 終了値
        body: Box<AstNode>,  // ループ本体（Blockノード）
    },

    /// 条件分岐
    If {
        condition: Box<AstNode>,         // 条件式（Bool型）
        then_body: Box<AstNode>,         // 条件が真の場合の処理
        else_body: Option<Box<AstNode>>, // 条件が偽の場合の処理（オプション）
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

    // Function definition - 通常関数定義
    Function {
        name: Option<String>, // 関数名（Program内ではこのフィールドは使用されず、匿名関数も可能）
        params: Vec<VarDecl>, // 引数リスト
        return_type: DType,   // 返り値の型
        body: Box<AstNode>,   // 関数本体（通常はBlock）
    },

    // Kernel definition - GPUカーネル定義
    Kernel {
        name: Option<String>, // カーネル名
        params: Vec<VarDecl>, // 引数リスト
        return_type: DType,   // 返り値の型（通常はvoid）
        body: Box<AstNode>,   // カーネル本体（通常はBlock）
        // 推奨dispatch設定（CallKernel生成時のデフォルト/ヒント）
        default_grid_size: [Box<AstNode>; 3], // グリッド数 (x, y, z)
        default_thread_group_size: [Box<AstNode>; 3], // スレッドグループサイズ (x, y, z)
    },

    // Kernel call - GPUカーネル呼び出し
    CallKernel {
        name: String,                         // 呼び出すカーネル名
        args: Vec<AstNode>,                   // 引数（バッファポインタ等）
        grid_size: [Box<AstNode>; 3],         // グリッド数 (x, y, z)
        thread_group_size: [Box<AstNode>; 3], // スレッドグループサイズ (x, y, z)
    },

    // Program - プログラム全体
    /// カーネル関数群を保持するプログラムノード
    ///
    /// Graphの最適化・Lowering後、複数のKernelノードが1つのProgramにまとめられます。
    /// `execution_waves`でカーネルの実行順序を管理します。
    ///
    /// ## 実行モデル
    ///
    /// ```text
    /// execution_waves = [
    ///     // Wave 0: 並列実行可能なカーネル群
    ///     [KernelA(x1→y1), KernelA(x2→y2)],
    ///     // <implicit barrier>
    ///     // Wave 1: Wave 0の結果に依存
    ///     [KernelB(y1,y2→z)],
    /// ]
    /// ```
    Program {
        /// AstNode::Function または AstNode::Kernel のリスト
        functions: Vec<AstNode>,
        /// 実行波（二重ネスト配列）
        ///
        /// - 内側のVec: 並列実行可能なカーネル呼び出し群
        /// - 外側のVec: 順次実行される波（間に暗黙のバリア）
        execution_waves: Vec<Vec<AstKernelCallInfo>>,
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
            | AstNode::RightShift(left, right)
            | AstNode::Lt(left, right)
            | AstNode::Le(left, right)
            | AstNode::Gt(left, right)
            | AstNode::Ge(left, right)
            | AstNode::Eq(left, right)
            | AstNode::Ne(left, right) => vec![left.as_ref(), right.as_ref()],
            AstNode::Recip(operand)
            | AstNode::Sqrt(operand)
            | AstNode::Log2(operand)
            | AstNode::Exp2(operand)
            | AstNode::Sin(operand)
            | AstNode::Floor(operand)
            | AstNode::BitwiseNot(operand)
            | AstNode::Cast(operand, _) => vec![operand.as_ref()],
            AstNode::Fma { a, b, c } => vec![a.as_ref(), b.as_ref(), c.as_ref()],
            AstNode::AtomicAdd {
                ptr, offset, value, ..
            }
            | AstNode::AtomicMax {
                ptr, offset, value, ..
            } => vec![ptr.as_ref(), offset.as_ref(), value.as_ref()],
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
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                let mut children = vec![condition.as_ref(), then_body.as_ref()];
                if let Some(else_b) = else_body {
                    children.push(else_b.as_ref());
                }
                children
            }
            AstNode::Call { args, .. } => args.iter().map(|node| node as &AstNode).collect(),
            AstNode::Return { value } => vec![value.as_ref()],
            AstNode::Barrier => vec![],
            AstNode::Allocate { size, .. } => vec![size.as_ref()],
            AstNode::Deallocate { ptr } => vec![ptr.as_ref()],
            AstNode::Function { body, .. } => vec![body.as_ref()],
            AstNode::Kernel {
                body,
                default_grid_size,
                default_thread_group_size,
                ..
            } => {
                let mut children = vec![body.as_ref()];
                for dim in default_grid_size {
                    children.push(dim.as_ref());
                }
                for dim in default_thread_group_size {
                    children.push(dim.as_ref());
                }
                children
            }
            AstNode::CallKernel {
                args,
                grid_size,
                thread_group_size,
                ..
            } => {
                let mut children: Vec<&AstNode> = args.iter().collect();
                for dim in grid_size {
                    children.push(dim.as_ref());
                }
                for dim in thread_group_size {
                    children.push(dim.as_ref());
                }
                children
            }
            AstNode::Program {
                functions,
                execution_waves: _,
            } => {
                // execution_waves内のgrid_size/local_sizeはExpr型なのでAstNode子ノードではない
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
            AstNode::Lt(left, right) => AstNode::Lt(Box::new(f(left)), Box::new(f(right))),
            AstNode::Le(left, right) => AstNode::Le(Box::new(f(left)), Box::new(f(right))),
            AstNode::Gt(left, right) => AstNode::Gt(Box::new(f(left)), Box::new(f(right))),
            AstNode::Ge(left, right) => AstNode::Ge(Box::new(f(left)), Box::new(f(right))),
            AstNode::Eq(left, right) => AstNode::Eq(Box::new(f(left)), Box::new(f(right))),
            AstNode::Ne(left, right) => AstNode::Ne(Box::new(f(left)), Box::new(f(right))),
            AstNode::Recip(operand) => AstNode::Recip(Box::new(f(operand))),
            AstNode::Sqrt(operand) => AstNode::Sqrt(Box::new(f(operand))),
            AstNode::Log2(operand) => AstNode::Log2(Box::new(f(operand))),
            AstNode::Exp2(operand) => AstNode::Exp2(Box::new(f(operand))),
            AstNode::Sin(operand) => AstNode::Sin(Box::new(f(operand))),
            AstNode::Floor(operand) => AstNode::Floor(Box::new(f(operand))),
            AstNode::BitwiseNot(operand) => AstNode::BitwiseNot(Box::new(f(operand))),
            AstNode::Cast(operand, dtype) => AstNode::Cast(Box::new(f(operand)), dtype.clone()),
            AstNode::Fma { a, b, c } => AstNode::Fma {
                a: Box::new(f(a)),
                b: Box::new(f(b)),
                c: Box::new(f(c)),
            },
            AstNode::AtomicAdd {
                ptr,
                offset,
                value,
                dtype,
            } => AstNode::AtomicAdd {
                ptr: Box::new(f(ptr)),
                offset: Box::new(f(offset)),
                value: Box::new(f(value)),
                dtype: dtype.clone(),
            },
            AstNode::AtomicMax {
                ptr,
                offset,
                value,
                dtype,
            } => AstNode::AtomicMax {
                ptr: Box::new(f(ptr)),
                offset: Box::new(f(offset)),
                value: Box::new(f(value)),
                dtype: dtype.clone(),
            },
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
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => AstNode::If {
                condition: Box::new(f(condition)),
                then_body: Box::new(f(then_body)),
                else_body: else_body.as_ref().map(|e| Box::new(f(e))),
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
            } => AstNode::Function {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(f(body)),
            },
            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => AstNode::Kernel {
                name: name.clone(),
                params: params.clone(),
                return_type: return_type.clone(),
                body: Box::new(f(body)),
                default_grid_size: [
                    Box::new(f(&default_grid_size[0])),
                    Box::new(f(&default_grid_size[1])),
                    Box::new(f(&default_grid_size[2])),
                ],
                default_thread_group_size: [
                    Box::new(f(&default_thread_group_size[0])),
                    Box::new(f(&default_thread_group_size[1])),
                    Box::new(f(&default_thread_group_size[2])),
                ],
            },
            AstNode::CallKernel {
                name,
                args,
                grid_size,
                thread_group_size,
            } => AstNode::CallKernel {
                name: name.clone(),
                args: args.iter().map(f).collect(),
                grid_size: [
                    Box::new(f(&grid_size[0])),
                    Box::new(f(&grid_size[1])),
                    Box::new(f(&grid_size[2])),
                ],
                thread_group_size: [
                    Box::new(f(&thread_group_size[0])),
                    Box::new(f(&thread_group_size[1])),
                    Box::new(f(&thread_group_size[2])),
                ],
            },
            AstNode::Program {
                functions,
                execution_waves,
            } => AstNode::Program {
                functions: functions.iter().map(f).collect(),
                // execution_wavesはExpr型なのでmap_childrenの対象外
                execution_waves: execution_waves.clone(),
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

            // Comparison operations - always return Bool
            AstNode::Lt(_, _)
            | AstNode::Le(_, _)
            | AstNode::Gt(_, _)
            | AstNode::Ge(_, _)
            | AstNode::Eq(_, _)
            | AstNode::Ne(_, _) => DType::Bool,

            // Unary operations that preserve type
            AstNode::Recip(operand) | AstNode::BitwiseNot(operand) => operand.infer_type(),

            // Mathematical operations that typically return F32
            AstNode::Sqrt(_)
            | AstNode::Log2(_)
            | AstNode::Exp2(_)
            | AstNode::Sin(_)
            | AstNode::Floor(_) => DType::F32,

            // Fused Multiply-Add - returns same type as operands (typically F32)
            AstNode::Fma { a, .. } => a.infer_type(),

            // Atomic operations - return the old value at the memory location
            AstNode::AtomicAdd { dtype, .. } | AstNode::AtomicMax { dtype, .. } => dtype.clone(),

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

            // If - 条件分岐は値を返さない（unit型）
            // （式としてのif-elseはthen_bodyの型を返すべきだが、
            //   ここでは制御構文として扱い、副作用のみを期待）
            AstNode::If { .. } => DType::Tuple(vec![]),

            // Call - 関数呼び出しの型は関数定義から推論する必要がある
            // ここでは関数定義を参照できないので、Unknownを返す
            // Programコンテキストで適切に型チェックする
            AstNode::Call { .. } => DType::Unknown,

            // CallKernel - カーネル呼び出しは値を返さない（unit型）
            AstNode::CallKernel { .. } => DType::Tuple(vec![]),

            // Return - 返す値の型を返す
            AstNode::Return { value } => value.infer_type(),

            // Barrier - 同期バリアは値を返さない（unit型）
            AstNode::Barrier => DType::Tuple(vec![]),

            // Allocate - ポインタを返す
            AstNode::Allocate { dtype, .. } => DType::Ptr(dtype.clone()),

            // Deallocate - 値を返さない（unit型）
            AstNode::Deallocate { .. } => DType::Tuple(vec![]),

            // Function/Kernel - 関数自体の型は返り値の型
            AstNode::Function { return_type, .. } | AstNode::Kernel { return_type, .. } => {
                return_type.clone()
            }

            // Program - プログラム全体の型はvoid（カーネル群の集合）
            AstNode::Program { .. } => DType::Tuple(vec![]),
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
            | AstNode::RightShift(left, right)
            | AstNode::Lt(left, right)
            | AstNode::Le(left, right)
            | AstNode::Gt(left, right)
            | AstNode::Ge(left, right)
            | AstNode::Eq(left, right)
            | AstNode::Ne(left, right) => {
                left.check_scope(scope)?;
                right.check_scope(scope)?;
                Ok(())
            }
            AstNode::Recip(operand)
            | AstNode::Sqrt(operand)
            | AstNode::Log2(operand)
            | AstNode::Exp2(operand)
            | AstNode::Sin(operand)
            | AstNode::Floor(operand)
            | AstNode::BitwiseNot(operand)
            | AstNode::Cast(operand, _) => {
                operand.check_scope(scope)?;
                Ok(())
            }
            // Fused Multiply-Add
            AstNode::Fma { a, b, c } => {
                a.check_scope(scope)?;
                b.check_scope(scope)?;
                c.check_scope(scope)?;
                Ok(())
            }
            // Atomic operations
            AstNode::AtomicAdd {
                ptr, offset, value, ..
            }
            | AstNode::AtomicMax {
                ptr, offset, value, ..
            } => {
                ptr.check_scope(scope)?;
                offset.check_scope(scope)?;
                value.check_scope(scope)?;
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
            // If - 条件分岐
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                condition.check_scope(scope)?;
                then_body.check_scope(scope)?;
                if let Some(else_b) = else_body {
                    else_b.check_scope(scope)?;
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
            // CallKernel - 引数とdispatch設定のスコープチェック
            AstNode::CallKernel {
                args,
                grid_size,
                thread_group_size,
                ..
            } => {
                for arg in args {
                    arg.check_scope(scope)?;
                }
                for dim in grid_size {
                    dim.check_scope(scope)?;
                }
                for dim in thread_group_size {
                    dim.check_scope(scope)?;
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
            // Function/Kernel - 関数本体のスコープチェック（パラメータは関数のスコープに含まれる）
            AstNode::Function { body, .. } | AstNode::Kernel { body, .. } => {
                body.check_scope(scope)
            }
            // Program - 各関数のスコープチェック
            AstNode::Program { functions, .. } => {
                for func in functions {
                    func.check_scope(scope)?;
                }
                Ok(())
            }
        }
    }

    /// Get a function or kernel from a Program by name
    ///
    /// Returns None if this is not a Program or if the function/kernel is not found
    pub fn get_function(&self, name: &str) -> Option<&AstNode> {
        match self {
            AstNode::Program { functions, .. } => functions.iter().find(|f| {
                matches!(f, AstNode::Function { name: Some(n), .. } | AstNode::Kernel { name: Some(n), .. } if n == name)
            }),
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

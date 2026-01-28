//! 変数展開（Variable Expansion / Inlining）のためのSuggester実装
//!
//! 一時変数への代入を、その変数の使用箇所に展開します。
//! CSE（共通部分式除去）の逆操作として機能します。
//!
//! # 例
//! ```text
//! // 展開前
//! x = (a * (a + 1))
//! y = x + x
//!
//! // 展開後
//! y = (a * (a + 1)) + (a * (a + 1))
//! ```
//!
//! # 用途
//! - CSE適用後にレジスタ圧力が高くなった場合の最適化解除
//! - 1回しか使用されない一時変数の除去
//! - 定数伝播など他の最適化のための準備

use crate::ast::{AstNode, Scope};
use crate::opt::ast::{AstSuggestResult, AstSuggester};
use log::{debug, trace};

/// 変数展開を提案するSuggester
pub struct VariableExpansionSuggester {
    /// 展開対象とする変数名のプレフィックス（Noneの場合はすべての変数が対象）
    target_prefix: Option<String>,
    /// 使用回数がこの値以下の変数のみ展開（Noneの場合は制限なし）
    max_usage_count: Option<usize>,
}

impl VariableExpansionSuggester {
    /// 新しいVariableExpansionSuggesterを作成
    pub fn new() -> Self {
        Self {
            target_prefix: None,
            max_usage_count: None,
        }
    }

    /// 展開対象とする変数名のプレフィックスを設定
    ///
    /// 例: `with_prefix("cse_tmp")` でCSEで生成された変数のみを対象にする
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.target_prefix = Some(prefix.to_string());
        self
    }

    /// 展開対象とする変数の最大使用回数を設定
    ///
    /// 例: `with_max_usage(2)` で2回以下使用される変数のみを展開
    pub fn with_max_usage(mut self, count: usize) -> Self {
        self.max_usage_count = Some(count);
        self
    }

    /// 変数が展開対象かどうかをチェック
    fn is_expansion_target(&self, var_name: &str) -> bool {
        match &self.target_prefix {
            Some(prefix) => var_name.starts_with(prefix),
            None => true,
        }
    }

    /// ステートメント内で指定した変数への再代入があるかチェック（再帰的に子ノードも確認）
    fn has_reassignment_in_statements(statements: &[AstNode], var_name: &str) -> bool {
        for stmt in statements {
            if Self::has_reassignment(stmt, var_name) {
                return true;
            }
        }
        false
    }

    /// ASTノード内で指定した変数への再代入があるかチェック
    fn has_reassignment(node: &AstNode, var_name: &str) -> bool {
        match node {
            // Assign: 対象変数への代入があるか
            AstNode::Assign { var, .. } if var == var_name => true,
            // Block: 子ステートメントを再帰チェック
            AstNode::Block { statements, .. } => {
                Self::has_reassignment_in_statements(statements, var_name)
            }
            // Range: ループ本体を再帰チェック
            AstNode::Range { body, .. } => Self::has_reassignment(body, var_name),
            // If: then/else両方を再帰チェック
            AstNode::If {
                then_body,
                else_body,
                ..
            } => {
                Self::has_reassignment(then_body, var_name)
                    || else_body
                        .as_ref()
                        .is_some_and(|e| Self::has_reassignment(e, var_name))
            }
            // Function/Kernel: 本体を再帰チェック
            AstNode::Function { body, .. } | AstNode::Kernel { body, .. } => {
                Self::has_reassignment(body, var_name)
            }
            // その他のノードは再代入なし
            _ => false,
        }
    }

    /// 式内の変数の使用回数をカウント
    fn count_var_usage(expr: &AstNode, var_name: &str) -> usize {
        match expr {
            AstNode::Var(name) if name == var_name => 1,
            AstNode::Var(_) | AstNode::Const(_) | AstNode::Rand | AstNode::Barrier => 0,
            // 二項演算
            AstNode::Add(a, b)
            | AstNode::Mul(a, b)
            | AstNode::Max(a, b)
            | AstNode::Rem(a, b)
            | AstNode::Idiv(a, b)
            | AstNode::BitwiseAnd(a, b)
            | AstNode::BitwiseOr(a, b)
            | AstNode::BitwiseXor(a, b)
            | AstNode::LeftShift(a, b)
            | AstNode::RightShift(a, b) => {
                Self::count_var_usage(a, var_name) + Self::count_var_usage(b, var_name)
            }
            // 比較・論理演算（プリミティブ）
            AstNode::Lt(a, b) | AstNode::And(a, b) => {
                Self::count_var_usage(a, var_name) + Self::count_var_usage(b, var_name)
            }
            AstNode::Not(a) => Self::count_var_usage(a, var_name),
            // 単項演算
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a)
            | AstNode::Floor(a)
            | AstNode::BitwiseNot(a)
            | AstNode::Cast(a, _) => Self::count_var_usage(a, var_name),
            // Fused Multiply-Add
            AstNode::Fma { a, b, c } => {
                Self::count_var_usage(a, var_name)
                    + Self::count_var_usage(b, var_name)
                    + Self::count_var_usage(c, var_name)
            }
            // アトミック操作
            AstNode::AtomicAdd {
                ptr, offset, value, ..
            }
            | AstNode::AtomicMax {
                ptr, offset, value, ..
            } => {
                Self::count_var_usage(ptr, var_name)
                    + Self::count_var_usage(offset, var_name)
                    + Self::count_var_usage(value, var_name)
            }
            // メモリ操作
            AstNode::Load { ptr, offset, .. } => {
                Self::count_var_usage(ptr, var_name) + Self::count_var_usage(offset, var_name)
            }
            AstNode::Store { ptr, offset, value } => {
                Self::count_var_usage(ptr, var_name)
                    + Self::count_var_usage(offset, var_name)
                    + Self::count_var_usage(value, var_name)
            }
            AstNode::Allocate { size, .. } => Self::count_var_usage(size, var_name),
            AstNode::Deallocate { ptr } => Self::count_var_usage(ptr, var_name),
            // 代入
            AstNode::Assign { value, .. } => Self::count_var_usage(value, var_name),
            // ブロック
            AstNode::Block { statements, .. } => statements
                .iter()
                .map(|s| Self::count_var_usage(s, var_name))
                .sum(),
            // 条件分岐
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                Self::count_var_usage(condition, var_name)
                    + Self::count_var_usage(then_body, var_name)
                    + else_body
                        .as_ref()
                        .map(|e| Self::count_var_usage(e, var_name))
                        .unwrap_or(0)
            }
            // ループ
            AstNode::Range {
                start,
                step,
                stop,
                body,
                ..
            } => {
                Self::count_var_usage(start, var_name)
                    + Self::count_var_usage(step, var_name)
                    + Self::count_var_usage(stop, var_name)
                    + Self::count_var_usage(body, var_name)
            }
            // 関数呼び出し
            AstNode::Call { args, .. } => args
                .iter()
                .map(|a| Self::count_var_usage(a, var_name))
                .sum(),
            // カーネル呼び出し
            AstNode::CallKernel {
                args,
                grid_size,
                thread_group_size,
                ..
            } => {
                let args_count: usize = args
                    .iter()
                    .map(|a| Self::count_var_usage(a, var_name))
                    .sum();
                let grid_count: usize = grid_size
                    .iter()
                    .map(|g| Self::count_var_usage(g, var_name))
                    .sum();
                let thread_count: usize = thread_group_size
                    .iter()
                    .map(|t| Self::count_var_usage(t, var_name))
                    .sum();
                args_count + grid_count + thread_count
            }
            // Return
            AstNode::Return { value } => Self::count_var_usage(value, var_name),
            // 関数定義
            AstNode::Function { body, .. } => Self::count_var_usage(body, var_name),
            // カーネル定義
            AstNode::Kernel {
                body,
                default_grid_size,
                default_thread_group_size,
                ..
            } => {
                let body_count = Self::count_var_usage(body, var_name);
                let grid_count: usize = default_grid_size
                    .iter()
                    .map(|g| Self::count_var_usage(g, var_name))
                    .sum();
                let thread_count: usize = default_thread_group_size
                    .iter()
                    .map(|t| Self::count_var_usage(t, var_name))
                    .sum();
                body_count + grid_count + thread_count
            }
            // プログラム
            AstNode::Program { functions, .. } => functions
                .iter()
                .map(|f| Self::count_var_usage(f, var_name))
                .sum(),
            // Wildcard等（パターンマッチング用）
            _ => 0,
        }
    }

    /// 式内の変数参照を別の式で置換
    fn substitute_var(expr: &AstNode, var_name: &str, replacement: &AstNode) -> AstNode {
        match expr {
            AstNode::Var(name) if name == var_name => replacement.clone(),
            AstNode::Var(_)
            | AstNode::Const(_)
            | AstNode::Rand
            | AstNode::Wildcard(_)
            | AstNode::Barrier => expr.clone(),
            // 二項演算
            AstNode::Add(a, b) => AstNode::Add(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Mul(a, b) => AstNode::Mul(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Max(a, b) => AstNode::Max(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Rem(a, b) => AstNode::Rem(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Idiv(a, b) => AstNode::Idiv(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::BitwiseAnd(a, b) => AstNode::BitwiseAnd(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::BitwiseOr(a, b) => AstNode::BitwiseOr(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::BitwiseXor(a, b) => AstNode::BitwiseXor(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::LeftShift(a, b) => AstNode::LeftShift(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::RightShift(a, b) => AstNode::RightShift(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            // 比較・論理演算（プリミティブのみ）
            AstNode::Lt(a, b) => AstNode::Lt(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::And(a, b) => AstNode::And(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                Box::new(Self::substitute_var(b, var_name, replacement)),
            ),
            AstNode::Not(a) => {
                AstNode::Not(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            // Select (ternary)
            AstNode::Select {
                cond,
                then_val,
                else_val,
            } => AstNode::Select {
                cond: Box::new(Self::substitute_var(cond, var_name, replacement)),
                then_val: Box::new(Self::substitute_var(then_val, var_name, replacement)),
                else_val: Box::new(Self::substitute_var(else_val, var_name, replacement)),
            },
            // 単項演算
            AstNode::Recip(a) => {
                AstNode::Recip(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Sqrt(a) => {
                AstNode::Sqrt(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Log2(a) => {
                AstNode::Log2(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Exp2(a) => {
                AstNode::Exp2(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Sin(a) => {
                AstNode::Sin(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Floor(a) => {
                AstNode::Floor(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::BitwiseNot(a) => {
                AstNode::BitwiseNot(Box::new(Self::substitute_var(a, var_name, replacement)))
            }
            AstNode::Cast(a, dtype) => AstNode::Cast(
                Box::new(Self::substitute_var(a, var_name, replacement)),
                dtype.clone(),
            ),
            // Fused Multiply-Add
            AstNode::Fma { a, b, c } => AstNode::Fma {
                a: Box::new(Self::substitute_var(a, var_name, replacement)),
                b: Box::new(Self::substitute_var(b, var_name, replacement)),
                c: Box::new(Self::substitute_var(c, var_name, replacement)),
            },
            // メモリ操作
            AstNode::Load {
                ptr,
                offset,
                count,
                dtype,
            } => AstNode::Load {
                ptr: Box::new(Self::substitute_var(ptr, var_name, replacement)),
                offset: Box::new(Self::substitute_var(offset, var_name, replacement)),
                count: *count,
                dtype: dtype.clone(),
            },
            AstNode::Store { ptr, offset, value } => AstNode::Store {
                ptr: Box::new(Self::substitute_var(ptr, var_name, replacement)),
                offset: Box::new(Self::substitute_var(offset, var_name, replacement)),
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
            },
            AstNode::Allocate { dtype, size } => AstNode::Allocate {
                dtype: dtype.clone(),
                size: Box::new(Self::substitute_var(size, var_name, replacement)),
            },
            AstNode::Deallocate { ptr } => AstNode::Deallocate {
                ptr: Box::new(Self::substitute_var(ptr, var_name, replacement)),
            },
            // アトミック操作
            AstNode::AtomicAdd {
                ptr,
                offset,
                value,
                dtype,
            } => AstNode::AtomicAdd {
                ptr: Box::new(Self::substitute_var(ptr, var_name, replacement)),
                offset: Box::new(Self::substitute_var(offset, var_name, replacement)),
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
                dtype: dtype.clone(),
            },
            AstNode::AtomicMax {
                ptr,
                offset,
                value,
                dtype,
            } => AstNode::AtomicMax {
                ptr: Box::new(Self::substitute_var(ptr, var_name, replacement)),
                offset: Box::new(Self::substitute_var(offset, var_name, replacement)),
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
                dtype: dtype.clone(),
            },
            // 代入
            AstNode::Assign { var, value } => AstNode::Assign {
                var: var.clone(),
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
            },
            // ブロック
            AstNode::Block { statements, scope } => AstNode::Block {
                statements: statements
                    .iter()
                    .map(|s| Self::substitute_var(s, var_name, replacement))
                    .collect(),
                scope: scope.clone(),
            },
            // 条件分岐
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => AstNode::If {
                condition: Box::new(Self::substitute_var(condition, var_name, replacement)),
                then_body: Box::new(Self::substitute_var(then_body, var_name, replacement)),
                else_body: else_body
                    .as_ref()
                    .map(|e| Box::new(Self::substitute_var(e, var_name, replacement))),
            },
            // ループ
            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
                parallel,
            } => AstNode::Range {
                var: var.clone(),
                start: Box::new(Self::substitute_var(start, var_name, replacement)),
                step: Box::new(Self::substitute_var(step, var_name, replacement)),
                stop: Box::new(Self::substitute_var(stop, var_name, replacement)),
                body: Box::new(Self::substitute_var(body, var_name, replacement)),
                parallel: parallel.clone(),
            },
            // 関数呼び出し
            AstNode::Call { name, args } => AstNode::Call {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_var(a, var_name, replacement))
                    .collect(),
            },
            // カーネル呼び出し
            AstNode::CallKernel {
                name,
                args,
                grid_size,
                thread_group_size,
            } => AstNode::CallKernel {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| Self::substitute_var(a, var_name, replacement))
                    .collect(),
                grid_size: [
                    Box::new(Self::substitute_var(&grid_size[0], var_name, replacement)),
                    Box::new(Self::substitute_var(&grid_size[1], var_name, replacement)),
                    Box::new(Self::substitute_var(&grid_size[2], var_name, replacement)),
                ],
                thread_group_size: [
                    Box::new(Self::substitute_var(
                        &thread_group_size[0],
                        var_name,
                        replacement,
                    )),
                    Box::new(Self::substitute_var(
                        &thread_group_size[1],
                        var_name,
                        replacement,
                    )),
                    Box::new(Self::substitute_var(
                        &thread_group_size[2],
                        var_name,
                        replacement,
                    )),
                ],
            },
            // Return
            AstNode::Return { value } => AstNode::Return {
                value: Box::new(Self::substitute_var(value, var_name, replacement)),
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
                body: Box::new(Self::substitute_var(body, var_name, replacement)),
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
                body: Box::new(Self::substitute_var(body, var_name, replacement)),
                default_grid_size: [
                    Box::new(Self::substitute_var(
                        &default_grid_size[0],
                        var_name,
                        replacement,
                    )),
                    Box::new(Self::substitute_var(
                        &default_grid_size[1],
                        var_name,
                        replacement,
                    )),
                    Box::new(Self::substitute_var(
                        &default_grid_size[2],
                        var_name,
                        replacement,
                    )),
                ],
                default_thread_group_size: [
                    Box::new(Self::substitute_var(
                        &default_thread_group_size[0],
                        var_name,
                        replacement,
                    )),
                    Box::new(Self::substitute_var(
                        &default_thread_group_size[1],
                        var_name,
                        replacement,
                    )),
                    Box::new(Self::substitute_var(
                        &default_thread_group_size[2],
                        var_name,
                        replacement,
                    )),
                ],
            },
            AstNode::Program {
                functions,
                execution_waves,
            } => AstNode::Program {
                functions: functions
                    .iter()
                    .map(|f| Self::substitute_var(f, var_name, replacement))
                    .collect(),
                execution_waves: execution_waves.clone(),
            },
            AstNode::WmmaMatmul {
                a_ptr,
                a_offset,
                a_stride,
                b_ptr,
                b_offset,
                b_stride,
                c_ptr,
                c_offset,
                c_stride,
                m,
                k,
                n,
                dtype_ab,
                dtype_c,
            } => AstNode::WmmaMatmul {
                a_ptr: Box::new(Self::substitute_var(a_ptr, var_name, replacement)),
                a_offset: Box::new(Self::substitute_var(a_offset, var_name, replacement)),
                a_stride: Box::new(Self::substitute_var(a_stride, var_name, replacement)),
                b_ptr: Box::new(Self::substitute_var(b_ptr, var_name, replacement)),
                b_offset: Box::new(Self::substitute_var(b_offset, var_name, replacement)),
                b_stride: Box::new(Self::substitute_var(b_stride, var_name, replacement)),
                c_ptr: Box::new(Self::substitute_var(c_ptr, var_name, replacement)),
                c_offset: Box::new(Self::substitute_var(c_offset, var_name, replacement)),
                c_stride: Box::new(Self::substitute_var(c_stride, var_name, replacement)),
                m: Box::new(Self::substitute_var(m, var_name, replacement)),
                k: Box::new(Self::substitute_var(k, var_name, replacement)),
                n: Box::new(Self::substitute_var(n, var_name, replacement)),
                dtype_ab: dtype_ab.clone(),
                dtype_c: dtype_c.clone(),
            },
        }
    }

    /// Block内で変数展開を試みる
    fn try_expand_in_block(&self, statements: &[AstNode], scope: &Scope) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        // 各Assign文を探す
        for (i, stmt) in statements.iter().enumerate() {
            if let AstNode::Assign { var, value } = stmt {
                // 展開対象かチェック
                if !self.is_expansion_target(var) {
                    continue;
                }

                // 変数が自身の右辺で使用されている場合はスキップ（累積パターン: acc = acc + ...）
                // これにより、ループ内のアキュムレータ（acc = acc + value）が誤って削除されるのを防ぐ
                if Self::count_var_usage(value, var) > 0 {
                    trace!(
                        "Skipping expansion of '{}': self-referential assignment",
                        var
                    );
                    continue;
                }

                // 後続のstatements内で同じ変数への再代入がある場合はスキップ
                // （例：acc = 0; for (...) { acc = acc + x }; result = acc）
                // この場合、accを展開するとループ内のaccが初期値に置換されてしまう
                let has_reassignment =
                    Self::has_reassignment_in_statements(&statements[i + 1..], var);
                if has_reassignment {
                    trace!(
                        "Skipping expansion of '{}': variable is reassigned in subsequent statements",
                        var
                    );
                    continue;
                }

                // 後続のstatements内での使用回数をカウント
                let usage_count: usize = statements[i + 1..]
                    .iter()
                    .map(|s| Self::count_var_usage(s, var))
                    .sum();

                // 使用回数の制限チェック
                if let Some(max) = self.max_usage_count
                    && usage_count > max
                {
                    continue;
                }

                // 使用されていない場合は単純に削除
                if usage_count == 0 {
                    debug!("Variable '{}' is unused, removing assignment", var);
                    let mut new_statements: Vec<AstNode> = statements[..i].to_vec();
                    new_statements.extend(statements[i + 1..].iter().cloned());
                    candidates.push(AstNode::Block {
                        statements: new_statements,
                        scope: Box::new(scope.clone()),
                    });
                    continue;
                }

                debug!(
                    "Expanding variable '{}' (used {} times): {:?}",
                    var, usage_count, value
                );

                // 変数参照を式で置換し、Assign文を削除
                let mut new_statements: Vec<AstNode> = statements[..i].to_vec();
                for following_stmt in &statements[i + 1..] {
                    new_statements.push(Self::substitute_var(following_stmt, var, value));
                }

                candidates.push(AstNode::Block {
                    statements: new_statements,
                    scope: Box::new(scope.clone()),
                });
            }
        }

        candidates
    }

    /// ASTツリーを走査して、展開可能な箇所を探す
    fn collect_expansion_candidates(&self, ast: &AstNode) -> Vec<AstNode> {
        let mut candidates = Vec::new();

        match ast {
            AstNode::Block { statements, scope } => {
                // Block内での展開を試みる
                let block_candidates = self.try_expand_in_block(statements, scope);
                candidates.extend(block_candidates);

                // 子要素を再帰的に処理
                for (i, stmt) in statements.iter().enumerate() {
                    let child_candidates = self.collect_expansion_candidates(stmt);
                    for child in child_candidates {
                        let mut new_statements = statements.clone();
                        new_statements[i] = child;
                        candidates.push(AstNode::Block {
                            statements: new_statements,
                            scope: scope.clone(),
                        });
                    }
                }
            }

            AstNode::Range {
                var,
                start,
                step,
                stop,
                body,
                parallel,
            } => {
                // ループ本体内での展開を試みる
                let child_candidates = self.collect_expansion_candidates(body);
                for child in child_candidates {
                    candidates.push(AstNode::Range {
                        var: var.clone(),
                        start: start.clone(),
                        step: step.clone(),
                        stop: stop.clone(),
                        body: Box::new(child),
                        parallel: parallel.clone(),
                    });
                }
            }

            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                // then_body内での展開を試みる
                let then_candidates = self.collect_expansion_candidates(then_body);
                for child in then_candidates {
                    candidates.push(AstNode::If {
                        condition: condition.clone(),
                        then_body: Box::new(child),
                        else_body: else_body.clone(),
                    });
                }

                // else_body内での展開を試みる
                if let Some(else_node) = else_body {
                    let else_candidates = self.collect_expansion_candidates(else_node);
                    for child in else_candidates {
                        candidates.push(AstNode::If {
                            condition: condition.clone(),
                            then_body: then_body.clone(),
                            else_body: Some(Box::new(child)),
                        });
                    }
                }
            }

            AstNode::Function {
                name,
                params,
                return_type,
                body,
            } => {
                let child_candidates = self.collect_expansion_candidates(body);
                for child in child_candidates {
                    candidates.push(AstNode::Function {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(child),
                    });
                }
            }

            AstNode::Kernel {
                name,
                params,
                return_type,
                body,
                default_grid_size,
                default_thread_group_size,
            } => {
                let child_candidates = self.collect_expansion_candidates(body);
                for child in child_candidates {
                    candidates.push(AstNode::Kernel {
                        name: name.clone(),
                        params: params.clone(),
                        return_type: return_type.clone(),
                        body: Box::new(child),
                        default_grid_size: default_grid_size.clone(),
                        default_thread_group_size: default_thread_group_size.clone(),
                    });
                }
            }

            AstNode::Program {
                functions,
                execution_waves,
            } => {
                for (i, func) in functions.iter().enumerate() {
                    let child_candidates = self.collect_expansion_candidates(func);
                    for child in child_candidates {
                        let mut new_functions = functions.clone();
                        new_functions[i] = child;
                        candidates.push(AstNode::Program {
                            functions: new_functions,
                            execution_waves: execution_waves.clone(),
                        });
                    }
                }
            }

            _ => {}
        }

        candidates
    }
}

impl Default for VariableExpansionSuggester {
    fn default() -> Self {
        Self::new()
    }
}

impl AstSuggester for VariableExpansionSuggester {
    fn name(&self) -> &str {
        "VariableExpansion"
    }

    fn suggest(&self, ast: &AstNode) -> Vec<AstSuggestResult> {
        trace!("VariableExpansionSuggester: Generating expansion suggestions");
        let candidates = self.collect_expansion_candidates(ast);
        let suggestions = super::deduplicate_candidates(candidates);

        debug!(
            "VariableExpansionSuggester: Generated {} unique suggestions",
            suggestions.len()
        );

        suggestions
            .into_iter()
            .map(|ast| AstSuggestResult::with_description(ast, self.name(), "expand variable"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{assign, block, const_int, if_then, range, store, var};

    #[test]
    fn test_simple_expansion() {
        // xのみを展開対象にする
        let suggester = VariableExpansionSuggester::new().with_prefix("x");

        // x = (a * (a + 1))
        // y = x + x
        let common_expr = var("a") * (var("a") + const_int(1));

        let assign_x = assign("x", common_expr.clone());
        let assign_y = assign("y", var("x") + var("x"));

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // 展開後は1つのstatementのみ（xの代入は削除）
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 1);

            // y = (a * (a + 1)) + (a * (a + 1))
            if let AstNode::Assign { var, value } = &statements[0] {
                assert_eq!(var, "y");
                // 展開されているか確認（xが含まれていない）
                assert_eq!(VariableExpansionSuggester::count_var_usage(value, "x"), 0);
                // 元の式が展開されている
                let value_str = format!("{:?}", value);
                assert!(value_str.contains("Mul"));
            }
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_expansion_with_prefix_filter() {
        let suggester = VariableExpansionSuggester::new().with_prefix("cse_tmp");

        // cse_tmp_0 = (a + b)  <- 対象
        // regular_var = (c + d)  <- 対象外
        // y = cse_tmp_0 + regular_var
        let assign_cse = assign("cse_tmp_0", var("a") + var("b"));
        let assign_regular = assign("regular_var", var("c") + var("d"));
        let assign_y = assign("y", var("cse_tmp_0") + var("regular_var"));

        let input = block(vec![assign_cse, assign_regular, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // cse_tmp_0のみ展開される
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 2); // regular_varの代入 + yの代入

            // regular_varの代入は残っている
            if let AstNode::Assign { var, .. } = &statements[0] {
                assert_eq!(var, "regular_var");
            }

            // yの式でcse_tmp_0は展開されているが、regular_varは残っている
            if let AstNode::Assign { value, .. } = &statements[1] {
                assert_eq!(
                    VariableExpansionSuggester::count_var_usage(value, "cse_tmp_0"),
                    0
                );
                assert_eq!(
                    VariableExpansionSuggester::count_var_usage(value, "regular_var"),
                    1
                );
            }
        }
    }

    #[test]
    fn test_expansion_with_max_usage() {
        // xのみを展開対象にする（yは未使用なので別の候補になる）
        let suggester = VariableExpansionSuggester::new()
            .with_max_usage(1)
            .with_prefix("x");

        // x = (a + b)  <- 2回使用されるので展開対象外
        // y = x + x
        let assign_x = assign("x", var("a") + var("b"));
        let assign_y = assign("y", var("x") + var("x"));

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        // 使用回数が2なので展開されない
        assert_eq!(suggestions.len(), 0);
    }

    #[test]
    fn test_expansion_single_use() {
        // xのみを展開対象にする（yは未使用なので別の候補になる）
        let suggester = VariableExpansionSuggester::new()
            .with_max_usage(1)
            .with_prefix("x");

        // x = (a + b)  <- 1回のみ使用
        // y = x * c
        let assign_x = assign("x", var("a") + var("b"));
        let assign_y = assign("y", var("x") * var("c"));

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // 展開後は1つのstatement
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 1);
        }
    }

    #[test]
    fn test_remove_unused_variable() {
        // xのみを展開対象にする（yも未使用なので別の候補になる）
        let suggester = VariableExpansionSuggester::new().with_prefix("x");

        // x = (a + b)  <- 使用されていない
        // y = c * d
        let assign_x = assign("x", var("a") + var("b"));
        let assign_y = assign("y", var("c") * var("d"));

        let input = block(vec![assign_x, assign_y], Scope::new());

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // 未使用の代入が削除される
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 1);
            if let AstNode::Assign { var, .. } = &statements[0] {
                assert_eq!(var, "y");
            }
        }
    }

    #[test]
    fn test_expansion_in_loop() {
        let suggester = VariableExpansionSuggester::new();

        // for i in 0..10:
        //   tmp = a + b
        //   output[i] = tmp * c
        let assign_tmp = assign("tmp", var("a") + var("b"));
        let store_output = store(var("output"), var("i"), var("tmp") * var("c"));

        let loop_body = block(vec![assign_tmp, store_output], Scope::new());

        let input = block(
            vec![range(
                "i",
                const_int(0),
                const_int(1),
                const_int(10),
                loop_body,
            )],
            Scope::new(),
        );

        let suggestions = suggester.suggest(&input);

        assert_eq!(suggestions.len(), 1);

        // ループ本体で展開されている
        if let AstNode::Block { statements, .. } = &suggestions[0].ast
            && let AstNode::Range { body, .. } = &statements[0]
            && let AstNode::Block {
                statements: inner, ..
            } = body.as_ref()
        {
            // 1つのstatementのみ（tmpの代入が削除）
            assert_eq!(inner.len(), 1);
        }
    }

    #[test]
    fn test_count_var_usage() {
        // a + b -> aの使用回数は1
        let expr1 = var("a") + var("b");
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr1, "a"), 1);
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr1, "b"), 1);
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr1, "c"), 0);

        // a + a -> aの使用回数は2
        let expr2 = var("a") + var("a");
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr2, "a"), 2);

        // (a + b) * a -> aの使用回数は2
        let expr3 = (var("a") + var("b")) * var("a");
        assert_eq!(VariableExpansionSuggester::count_var_usage(&expr3, "a"), 2);
    }

    #[test]
    fn test_cse_and_expansion_roundtrip() {
        // CSEと展開は互いに逆操作
        // 元の式: y = (a * b) + (a * b)
        let common_expr = var("a") * var("b");
        let _original = block(
            vec![assign("y", common_expr.clone() + common_expr)],
            Scope::new(),
        );

        // CSE適用後の形式（手動で作成）
        // cse_tmp_0 = a * b
        // y = cse_tmp_0 + cse_tmp_0
        let cse_applied = block(
            vec![
                assign("cse_tmp_0", var("a") * var("b")),
                assign("y", var("cse_tmp_0") + var("cse_tmp_0")),
            ],
            Scope::new(),
        );

        // 展開すると元の形式に戻る
        // cse_tmp_で始まる変数のみを展開対象にする（yは別の候補になる）
        let suggester = VariableExpansionSuggester::new().with_prefix("cse_tmp_");
        let suggestions = suggester.suggest(&cse_applied);

        assert_eq!(suggestions.len(), 1);

        // 展開後の構造が元の構造と同じ
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 1);
            if let AstNode::Assign { var, .. } = &statements[0] {
                assert_eq!(var, "y");
            }
        }
    }

    #[test]
    fn test_count_var_usage_in_if() {
        // If節内での変数カウントをテスト
        // if (x < 10) { y = x } else { y = 0 }
        let if_node = AstNode::If {
            condition: Box::new(AstNode::Lt(Box::new(var("x")), Box::new(const_int(10)))),
            then_body: Box::new(assign("y", var("x"))),
            else_body: Some(Box::new(assign("y", const_int(0)))),
        };

        // xは条件式で1回 + then_bodyで1回 = 2回使用
        assert_eq!(
            VariableExpansionSuggester::count_var_usage(&if_node, "x"),
            2
        );
        // yは代入のターゲットなのでカウントされない
        assert_eq!(
            VariableExpansionSuggester::count_var_usage(&if_node, "y"),
            0
        );
    }

    #[test]
    fn test_count_var_usage_in_comparison() {
        // 比較・論理演算子内での変数カウントをテスト（プリミティブのみ）
        // a < b, a && b, !a
        let lt = AstNode::Lt(Box::new(var("a")), Box::new(var("b")));
        let and = AstNode::And(Box::new(var("a")), Box::new(var("b")));
        let not = AstNode::Not(Box::new(var("a")));

        // 二項演算のテスト
        for expr in [&lt, &and] {
            assert_eq!(VariableExpansionSuggester::count_var_usage(expr, "a"), 1);
            assert_eq!(VariableExpansionSuggester::count_var_usage(expr, "b"), 1);
        }

        // Not演算のテスト
        assert_eq!(VariableExpansionSuggester::count_var_usage(&not, "a"), 1);
        assert_eq!(VariableExpansionSuggester::count_var_usage(&not, "b"), 0);
    }

    #[test]
    fn test_substitute_var_in_if() {
        // If節内での変数置換をテスト
        // if (x < 10) { store(out, i, x) }
        let if_node = if_then(
            AstNode::Lt(Box::new(var("x")), Box::new(const_int(10))),
            store(var("out"), var("i"), var("x")),
        );

        // xを(a + b)で置換
        let replacement = var("a") + var("b");
        let result = VariableExpansionSuggester::substitute_var(&if_node, "x", &replacement);

        // 結果のIf節内でxが置換されている
        if let AstNode::If {
            condition,
            then_body,
            ..
        } = &result
        {
            // 条件式内のxが置換されている
            assert_eq!(
                VariableExpansionSuggester::count_var_usage(condition, "x"),
                0
            );
            assert_eq!(
                VariableExpansionSuggester::count_var_usage(condition, "a"),
                1
            );

            // 本体内のxが置換されている
            assert_eq!(
                VariableExpansionSuggester::count_var_usage(then_body, "x"),
                0
            );
            assert_eq!(
                VariableExpansionSuggester::count_var_usage(then_body, "a"),
                1
            );
        } else {
            panic!("Expected If node");
        }
    }

    #[test]
    fn test_substitute_var_in_comparison() {
        // 比較演算子内での変数置換をテスト
        let lt = AstNode::Lt(Box::new(var("x")), Box::new(const_int(10)));
        let replacement = var("a") * const_int(2);
        let result = VariableExpansionSuggester::substitute_var(&lt, "x", &replacement);

        // xが置換されている
        assert_eq!(VariableExpansionSuggester::count_var_usage(&result, "x"), 0);
        assert_eq!(VariableExpansionSuggester::count_var_usage(&result, "a"), 1);

        // 結果がLt(Mul(a, 2), 10)
        if let AstNode::Lt(left, right) = &result {
            assert!(matches!(left.as_ref(), AstNode::Mul(..)));
            assert!(matches!(right.as_ref(), AstNode::Const(_)));
        } else {
            panic!("Expected Lt node");
        }
    }

    #[test]
    fn test_skip_accumulator_pattern() {
        // 累積パターン（acc = acc + ...）はスキップされるべき
        let suggester = VariableExpansionSuggester::new();

        // acc = 0
        // acc = acc + value  <- 自己参照のため展開しない
        // result = acc
        let init_acc = assign("acc", const_int(0));
        let accumulate = assign("acc", var("acc") + var("value"));
        let use_acc = assign("result", var("acc"));

        let input = block(vec![init_acc, accumulate, use_acc], Scope::new());

        let suggestions = suggester.suggest(&input);

        // 候補は存在するが、累積パターン（acc = acc + value）は展開されない
        // init_acc（acc = 0）は後続で使用されているため展開候補になる可能性がある
        for suggestion in &suggestions {
            if let AstNode::Block { statements, .. } = &suggestion.ast {
                // acc = acc + value が消えていないことを確認
                let has_accumulate = statements.iter().any(|stmt| {
                    if let AstNode::Assign { var, value } = stmt {
                        var == "acc"
                            && VariableExpansionSuggester::count_var_usage(value, "acc") > 0
                    } else {
                        false
                    }
                });
                assert!(has_accumulate, "Accumulator pattern should not be removed");
            }
        }
    }

    #[test]
    fn test_variable_used_in_range_body() {
        // Range（forループ）本体内で使用される変数の展開テスト
        let suggester = VariableExpansionSuggester::new().with_prefix("tmp");

        // tmp = a + b
        // for i in 0..10:
        //   output[i] = tmp * c
        let assign_tmp = assign("tmp", var("a") + var("b"));
        let loop_body = store(var("output"), var("i"), var("tmp") * var("c"));
        let loop_range = range("i", const_int(0), const_int(1), const_int(10), loop_body);

        let input = block(vec![assign_tmp, loop_range], Scope::new());

        let suggestions = suggester.suggest(&input);

        // tmpはRange本体内で使用されているので、展開される
        assert_eq!(suggestions.len(), 1);

        // 展開後、tmpの代入が削除されている
        if let AstNode::Block { statements, .. } = &suggestions[0].ast {
            assert_eq!(statements.len(), 1);
            // Range内でtmpが展開されている
            if let AstNode::Range { body, .. } = &statements[0] {
                assert_eq!(VariableExpansionSuggester::count_var_usage(body, "tmp"), 0);
                // a, bが含まれている
                assert_eq!(VariableExpansionSuggester::count_var_usage(body, "a"), 1);
                assert_eq!(VariableExpansionSuggester::count_var_usage(body, "b"), 1);
            } else {
                panic!("Expected Range node");
            }
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_expansion_in_if_block() {
        // If文内のBlock内での変数展開をテスト
        let suggester = VariableExpansionSuggester::new().with_prefix("tmp");

        // if (cond) {
        //   tmp = a + b
        //   output = tmp * c
        // }
        let assign_tmp = assign("tmp", var("a") + var("b"));
        let assign_output = assign("output", var("tmp") * var("c"));

        let if_body = block(vec![assign_tmp, assign_output], Scope::new());

        let input = if_then(var("cond"), if_body);

        let suggestions = suggester.suggest(&input);

        // If文内のBlock内での展開が検出される
        assert_eq!(suggestions.len(), 1);

        // 展開後、tmpの代入が削除されている
        if let AstNode::If { then_body, .. } = &suggestions[0].ast {
            if let AstNode::Block { statements, .. } = then_body.as_ref() {
                assert_eq!(statements.len(), 1);
                // outputの式でtmpが展開されている
                if let AstNode::Assign { value, .. } = &statements[0] {
                    assert_eq!(VariableExpansionSuggester::count_var_usage(value, "tmp"), 0);
                }
            } else {
                panic!("Expected Block in then_body");
            }
        } else {
            panic!("Expected If node");
        }
    }
}

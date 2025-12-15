//! 並列化Suggester用の共通ユーティリティ
//!
//! ThreadParallelizationSuggesterとGroupParallelizationSuggesterで
//! 共有される書き込み解析や変数置換機能を提供します。

use crate::ast::{AstNode, DType, Literal, Mutability, VarDecl, VarKind};
use std::collections::HashSet;

/// ループの並列化可否を解析するアナライザー
///
/// ループ本体を解析し、以下の条件をチェックします：
/// 1. ループ外変数への書き込み（Assign）がないこと
/// 2. Store先のオフセットがループ変数に依存していること（レースコンディション回避）
#[derive(Debug)]
pub struct LoopAnalyzer {
    /// ループ変数名
    loop_var: String,
    /// ループ外で定義された変数への書き込みを検出
    external_writes: HashSet<String>,
    /// Store操作がループ変数に依存しているか
    stores_depend_on_loop_var: bool,
    /// Store操作が存在するか
    has_stores: bool,
}

impl LoopAnalyzer {
    /// 新しいアナライザーを作成
    pub fn new(loop_var: impl Into<String>) -> Self {
        Self {
            loop_var: loop_var.into(),
            external_writes: HashSet::new(),
            stores_depend_on_loop_var: false,
            has_stores: false,
        }
    }

    /// ループ本体を解析
    pub fn analyze(&mut self, body: &AstNode) {
        self.analyze_recursive(body);
    }

    fn analyze_recursive(&mut self, node: &AstNode) {
        match node {
            AstNode::Assign { var, value } => {
                // ループ変数以外への代入は外部書き込み
                if var != &self.loop_var {
                    self.external_writes.insert(var.clone());
                }
                self.analyze_recursive(value);
            }
            AstNode::Store { ptr, offset, value } => {
                self.has_stores = true;
                // offsetがloop_varに依存しているかチェック
                if self.depends_on_loop_var(offset) {
                    self.stores_depend_on_loop_var = true;
                }
                self.analyze_recursive(ptr);
                self.analyze_recursive(offset);
                self.analyze_recursive(value);
            }
            AstNode::Block { statements, .. } => {
                for stmt in statements {
                    self.analyze_recursive(stmt);
                }
            }
            AstNode::Range { body, .. } => {
                // ネストしたループの内部も解析
                self.analyze_recursive(body);
            }
            AstNode::If {
                condition,
                then_body,
                else_body,
            } => {
                self.analyze_recursive(condition);
                self.analyze_recursive(then_body);
                if let Some(else_b) = else_body {
                    self.analyze_recursive(else_b);
                }
            }
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
            | AstNode::RightShift(a, b)
            | AstNode::Lt(a, b)
            | AstNode::Le(a, b)
            | AstNode::Gt(a, b)
            | AstNode::Ge(a, b)
            | AstNode::Eq(a, b)
            | AstNode::Ne(a, b) => {
                self.analyze_recursive(a);
                self.analyze_recursive(b);
            }
            // 単項演算
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a)
            | AstNode::BitwiseNot(a)
            | AstNode::Cast(a, _) => {
                self.analyze_recursive(a);
            }
            AstNode::Load { ptr, offset, .. } => {
                self.analyze_recursive(ptr);
                self.analyze_recursive(offset);
            }
            AstNode::Call { args, .. } => {
                for arg in args {
                    self.analyze_recursive(arg);
                }
            }
            AstNode::Return { value } => {
                self.analyze_recursive(value);
            }
            AstNode::Allocate { size, .. } => {
                self.analyze_recursive(size);
            }
            AstNode::Deallocate { ptr } => {
                self.analyze_recursive(ptr);
            }
            // リーフノード
            AstNode::Const(_)
            | AstNode::Var(_)
            | AstNode::Wildcard(_)
            | AstNode::Rand
            | AstNode::Barrier => {}
            // Function/Kernel/Programはループ本体内には現れない想定
            AstNode::Function { .. }
            | AstNode::Kernel { .. }
            | AstNode::Program { .. }
            | AstNode::CallKernel { .. } => {}
        }
    }

    /// 式がループ変数に依存しているかチェック
    fn depends_on_loop_var(&self, expr: &AstNode) -> bool {
        match expr {
            AstNode::Var(name) => name == &self.loop_var,
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
                self.depends_on_loop_var(a) || self.depends_on_loop_var(b)
            }
            AstNode::Recip(a)
            | AstNode::Sqrt(a)
            | AstNode::Log2(a)
            | AstNode::Exp2(a)
            | AstNode::Sin(a)
            | AstNode::BitwiseNot(a)
            | AstNode::Cast(a, _) => self.depends_on_loop_var(a),
            _ => false,
        }
    }

    /// 並列化可能かどうかを判定
    ///
    /// 条件:
    /// - 外部変数への書き込みがない
    /// - Store操作がある場合、そのオフセットがループ変数に依存している
    pub fn is_parallelizable(&self) -> bool {
        // 外部書き込みがあればNG
        if !self.external_writes.is_empty() {
            log::trace!(
                "Loop not parallelizable: external writes to {:?}",
                self.external_writes
            );
            return false;
        }

        // Storeがある場合、オフセットがループ変数に依存していなければNG（レースコンディション）
        if self.has_stores && !self.stores_depend_on_loop_var {
            log::trace!("Loop not parallelizable: stores do not depend on loop variable");
            return false;
        }

        true
    }

    /// 外部書き込み変数のリストを取得
    pub fn external_writes(&self) -> &HashSet<String> {
        &self.external_writes
    }
}

/// Rangeループが並列化可能かどうかを判定
///
/// Function/Kernel直下のRangeループに対して使用します。
pub fn is_range_parallelizable(range_node: &AstNode) -> bool {
    match range_node {
        AstNode::Range { var, body, .. } => {
            let mut analyzer = LoopAnalyzer::new(var);
            analyzer.analyze(body);
            analyzer.is_parallelizable()
        }
        _ => false,
    }
}

/// AST内の変数を別の式で置換
///
/// ループ変数をThreadIdやGroupIdに置き換える際に使用します。
pub fn substitute_var(ast: &AstNode, var_name: &str, replacement: &AstNode) -> AstNode {
    match ast {
        AstNode::Var(name) if name == var_name => replacement.clone(),

        // 二項演算
        AstNode::Add(a, b) => AstNode::Add(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Mul(a, b) => AstNode::Mul(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Max(a, b) => AstNode::Max(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Rem(a, b) => AstNode::Rem(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Idiv(a, b) => AstNode::Idiv(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::BitwiseAnd(a, b) => AstNode::BitwiseAnd(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::BitwiseOr(a, b) => AstNode::BitwiseOr(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::BitwiseXor(a, b) => AstNode::BitwiseXor(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::LeftShift(a, b) => AstNode::LeftShift(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::RightShift(a, b) => AstNode::RightShift(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Lt(a, b) => AstNode::Lt(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Le(a, b) => AstNode::Le(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Gt(a, b) => AstNode::Gt(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Ge(a, b) => AstNode::Ge(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Eq(a, b) => AstNode::Eq(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),
        AstNode::Ne(a, b) => AstNode::Ne(
            Box::new(substitute_var(a, var_name, replacement)),
            Box::new(substitute_var(b, var_name, replacement)),
        ),

        // 単項演算
        AstNode::Recip(a) => AstNode::Recip(Box::new(substitute_var(a, var_name, replacement))),
        AstNode::Sqrt(a) => AstNode::Sqrt(Box::new(substitute_var(a, var_name, replacement))),
        AstNode::Log2(a) => AstNode::Log2(Box::new(substitute_var(a, var_name, replacement))),
        AstNode::Exp2(a) => AstNode::Exp2(Box::new(substitute_var(a, var_name, replacement))),
        AstNode::Sin(a) => AstNode::Sin(Box::new(substitute_var(a, var_name, replacement))),
        AstNode::BitwiseNot(a) => {
            AstNode::BitwiseNot(Box::new(substitute_var(a, var_name, replacement)))
        }
        AstNode::Cast(a, dtype) => AstNode::Cast(
            Box::new(substitute_var(a, var_name, replacement)),
            dtype.clone(),
        ),

        // メモリ操作
        AstNode::Load {
            ptr,
            offset,
            count,
            dtype,
        } => AstNode::Load {
            ptr: Box::new(substitute_var(ptr, var_name, replacement)),
            offset: Box::new(substitute_var(offset, var_name, replacement)),
            count: *count,
            dtype: dtype.clone(),
        },
        AstNode::Store { ptr, offset, value } => AstNode::Store {
            ptr: Box::new(substitute_var(ptr, var_name, replacement)),
            offset: Box::new(substitute_var(offset, var_name, replacement)),
            value: Box::new(substitute_var(value, var_name, replacement)),
        },

        // 代入（変数名自体は置換しない）
        AstNode::Assign { var, value } => AstNode::Assign {
            var: var.clone(),
            value: Box::new(substitute_var(value, var_name, replacement)),
        },

        // ブロック
        AstNode::Block { statements, scope } => AstNode::Block {
            statements: statements
                .iter()
                .map(|s| substitute_var(s, var_name, replacement))
                .collect(),
            scope: scope.clone(),
        },

        // Range（内側のループは変数名が異なるはずなので再帰）
        AstNode::Range {
            var,
            start,
            step,
            stop,
            body,
        } => {
            // ループ変数が同じ名前の場合はシャドウイングされるので置換しない
            if var == var_name {
                ast.clone()
            } else {
                AstNode::Range {
                    var: var.clone(),
                    start: Box::new(substitute_var(start, var_name, replacement)),
                    step: Box::new(substitute_var(step, var_name, replacement)),
                    stop: Box::new(substitute_var(stop, var_name, replacement)),
                    body: Box::new(substitute_var(body, var_name, replacement)),
                }
            }
        }

        // 条件分岐
        AstNode::If {
            condition,
            then_body,
            else_body,
        } => AstNode::If {
            condition: Box::new(substitute_var(condition, var_name, replacement)),
            then_body: Box::new(substitute_var(then_body, var_name, replacement)),
            else_body: else_body
                .as_ref()
                .map(|e| Box::new(substitute_var(e, var_name, replacement))),
        },

        // 関数呼び出し
        AstNode::Call { name, args } => AstNode::Call {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| substitute_var(a, var_name, replacement))
                .collect(),
        },

        // Return
        AstNode::Return { value } => AstNode::Return {
            value: Box::new(substitute_var(value, var_name, replacement)),
        },

        // メモリ確保
        AstNode::Allocate { dtype, size } => AstNode::Allocate {
            dtype: dtype.clone(),
            size: Box::new(substitute_var(size, var_name, replacement)),
        },
        AstNode::Deallocate { ptr } => AstNode::Deallocate {
            ptr: Box::new(substitute_var(ptr, var_name, replacement)),
        },

        // リーフノードはそのまま返す
        AstNode::Const(_)
        | AstNode::Var(_)
        | AstNode::Wildcard(_)
        | AstNode::Rand
        | AstNode::Barrier => ast.clone(),

        // Function/Kernel/Program/CallKernelはそのまま（並列化対象外）
        AstNode::Function { .. }
        | AstNode::Kernel { .. }
        | AstNode::Program { .. }
        | AstNode::CallKernel { .. } => ast.clone(),
    }
}

/// ceil_div(a, b) を計算するAstNodeを生成
///
/// (a + b - 1) / b と等価
pub fn ceil_div(a: AstNode, b: AstNode) -> AstNode {
    // (a + b - 1) / b
    let b_minus_1 = AstNode::Add(
        Box::new(b.clone()),
        Box::new(AstNode::Const(Literal::Int(-1))),
    );
    let numerator = AstNode::Add(Box::new(a), Box::new(b_minus_1));
    AstNode::Idiv(Box::new(numerator), Box::new(b))
}

/// ThreadId変数宣言を作成
pub fn thread_id_param(name: impl Into<String>, axis: usize) -> VarDecl {
    VarDecl {
        name: name.into(),
        dtype: DType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::ThreadId(axis),
    }
}

/// GroupId変数宣言を作成
pub fn group_id_param(name: impl Into<String>, axis: usize) -> VarDecl {
    VarDecl {
        name: name.into(),
        dtype: DType::Int,
        mutability: Mutability::Immutable,
        kind: VarKind::GroupId(axis),
    }
}

/// 定数整数を生成
pub fn const_int(value: isize) -> AstNode {
    AstNode::Const(Literal::Int(value))
}

/// 変数参照を生成
pub fn var(name: impl Into<String>) -> AstNode {
    AstNode::Var(name.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::helper::{load, store};

    #[test]
    fn test_parallelizable_simple_elementwise() {
        // for i in 0..N { Store(output, i, Load(input, i)) }
        // -> 並列化可能
        let body = store(
            var("output"),
            var("i"),
            load(var("input"), var("i"), DType::F32),
        );
        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(var("N")),
            body: Box::new(body),
        };

        assert!(is_range_parallelizable(&range));
    }

    #[test]
    fn test_not_parallelizable_external_write() {
        // for i in 0..N { sum = sum + Load(input, i) }
        // -> 並列化不可（sumへの外部書き込み）
        let body = AstNode::Assign {
            var: "sum".to_string(),
            value: Box::new(AstNode::Add(
                Box::new(var("sum")),
                Box::new(load(var("input"), var("i"), DType::F32)),
            )),
        };
        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(var("N")),
            body: Box::new(body),
        };

        assert!(!is_range_parallelizable(&range));
    }

    #[test]
    fn test_not_parallelizable_fixed_store_offset() {
        // for i in 0..N { Store(output, 0, ...) }
        // -> 並列化不可（レースコンディション）
        let body = store(
            var("output"),
            const_int(0), // 固定オフセット
            load(var("input"), var("i"), DType::F32),
        );
        let range = AstNode::Range {
            var: "i".to_string(),
            start: Box::new(const_int(0)),
            step: Box::new(const_int(1)),
            stop: Box::new(var("N")),
            body: Box::new(body),
        };

        assert!(!is_range_parallelizable(&range));
    }

    #[test]
    fn test_substitute_var() {
        // i + 1 を tid で置換 -> tid + 1
        let expr = AstNode::Add(Box::new(var("i")), Box::new(const_int(1)));
        let result = substitute_var(&expr, "i", &var("tid"));

        assert_eq!(
            result,
            AstNode::Add(Box::new(var("tid")), Box::new(const_int(1)))
        );
    }
}

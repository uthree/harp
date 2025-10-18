//! 最適化器モジュール
//!
//! このモジュールは、AST最適化とグラフ最適化のための抽象化を提供します。
//!
//! # モジュール構成
//!
//! - `ast`: AST（抽象構文木）に対する最適化器
//! - `graph`: グラフに対する最適化器（将来実装予定）

pub mod ast;

use crate::ast::AstNode;

pub trait AstOptimizer {
    fn apply(&self, ast: &AstNode) -> AstNode;

    /// 他のOptimizerと合成して新しいOptimizerを作成
    ///
    /// デフォルト実装では、selfとotherを含むComposedOptimizerを返します。
    /// ComposedOptimizerはこのメソッドをオーバーライドして、
    /// 既存のoptimizer群に新しいoptimizerを追加します。
    fn compose(self, other: impl AstOptimizer + 'static) -> ast::ComposedOptimizer
    where
        Self: Sized + 'static,
    {
        ast::ComposedOptimizer::new()
            .add_optimizer(self)
            .add_optimizer(other)
    }
}

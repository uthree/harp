use crate::ast::{AstNode, ConstLiteral};
use crate::graph::shape::view::View;

/// ユーティリティ関数群
pub(super) struct LowererUtils;

impl LowererUtils {
    /// DTypeからCの型名を取得
    pub fn get_c_type(dtype: &crate::ast::DType) -> &'static str {
        use crate::ast::DType;
        match dtype {
            DType::F32 => "float",
            DType::Usize => "size_t",
            DType::Isize => "ssize_t",
            DType::Void => "void",
            DType::Ptr(_) => "void*",
            DType::Vec(_, _) => "void*", // ベクトル型も一旦void*として扱う
        }
    }

    /// Viewから総要素数を計算（静的サイズのみ）
    pub fn compute_total_size(view: &View) -> Option<usize> {
        let View::Linear { shape, .. } = view;
        let mut total = 1;
        for dim in shape {
            if let crate::graph::shape::Expr::Const(n) = dim {
                total *= *n as usize;
            } else {
                return None; // 動的サイズの場合
            }
        }
        Some(total)
    }

    /// Viewから総要素数を計算するAstNode式を生成（動的サイズ対応）
    pub fn compute_total_size_expr(view: &View) -> AstNode {
        let View::Linear { shape, .. } = view;
        if shape.is_empty() {
            return AstNode::Const(ConstLiteral::Usize(1));
        }

        let mut size_expr = Self::shape_expr_to_ast_node(&shape[0]);
        for dim in &shape[1..] {
            let dim_ast = Self::shape_expr_to_ast_node(dim);
            size_expr = AstNode::Mul(Box::new(size_expr), Box::new(dim_ast));
        }
        size_expr
    }

    /// メモリインデックスを計算（ループ変数i0, i1, ...を使用）
    pub fn compute_memory_index(
        strides: &[crate::graph::shape::Expr],
        offset: &crate::graph::shape::Expr,
        num_dims: usize,
    ) -> AstNode {
        use crate::graph::shape::Expr;

        // Exprレベルで計算してからAstNodeに変換することで、simplifyが適用される
        let mut index_expr = offset.clone();

        for (i, stride) in strides.iter().enumerate().take(num_dims) {
            let loop_var = Expr::Var(format!("i{}", i));
            let term = loop_var * stride.clone();
            index_expr += term;
        }

        // 最終的にsimplifyしてからAstNodeに変換
        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
    }

    /// メモリインデックスを計算（指定された次元のループ変数のみ上書き）
    pub fn compute_memory_index_with_override(
        strides: &[crate::graph::shape::Expr],
        offset: &crate::graph::shape::Expr,
        num_dims: usize,
        override_dim: usize,
        override_expr: &crate::graph::shape::Expr,
    ) -> AstNode {
        use crate::graph::shape::Expr;

        let mut index_expr = offset.clone();

        for (i, stride) in strides.iter().enumerate().take(num_dims) {
            let loop_var = if i == override_dim {
                override_expr.clone()
            } else {
                Expr::Var(format!("i{}", i))
            };
            let term = loop_var * stride.clone();
            index_expr += term;
        }

        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
    }

    /// Reduce演算の結果インデックスを計算（reduce軸をスキップ）
    pub fn compute_reduce_result_index(
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        current_dim: usize,
        reduce_axis: usize,
    ) -> AstNode {
        use crate::graph::shape::Expr;

        // Exprレベルで計算してからAstNodeに変換することで、simplifyが適用される
        let mut index_expr = result_offset.clone();

        let mut result_dim = 0;
        for input_dim in 0..current_dim {
            if input_dim != reduce_axis {
                let loop_var = Expr::Var(format!("i{}", input_dim));
                let term = loop_var * result_strides[result_dim].clone();
                index_expr += term;
                result_dim += 1;
            }
        }

        // 最終的にsimplifyしてからAstNodeに変換
        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
    }

    /// 複数軸のReduce演算の結果インデックスを計算
    pub fn compute_multi_reduce_result_index(
        result_strides: &[crate::graph::shape::Expr],
        result_offset: &crate::graph::shape::Expr,
        ndim: usize,
        reduce_axes: &[usize],
    ) -> AstNode {
        use crate::graph::shape::Expr;

        let mut index_expr = result_offset.clone();
        let mut result_dim = 0; // 出力テンソルの次元インデックス

        for dim in 0..ndim {
            if !reduce_axes.contains(&dim) {
                let loop_var = Expr::Var(format!("i{}", dim));
                let term = loop_var * result_strides[result_dim].clone();
                index_expr += term;
                result_dim += 1; // 出力次元をインクリメント
            }
        }

        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
    }

    /// Shape ExprをAstNodeに変換
    pub fn shape_expr_to_ast_node(expr: &crate::graph::shape::Expr) -> AstNode {
        use crate::graph::shape::Expr;
        match expr {
            Expr::Const(n) => AstNode::Const(ConstLiteral::Usize(*n as usize)),
            Expr::Var(name) => AstNode::Var(name.clone()),
            Expr::Add(left, right) => AstNode::Add(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(Self::shape_expr_to_ast_node(right)),
            ),
            Expr::Mul(left, right) => AstNode::Mul(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(Self::shape_expr_to_ast_node(right)),
            ),
            Expr::Div(left, right) => AstNode::Mul(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(AstNode::Recip(Box::new(Self::shape_expr_to_ast_node(right)))),
            ),
            Expr::Sub(left, right) => AstNode::Add(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(AstNode::Neg(Box::new(Self::shape_expr_to_ast_node(right)))),
            ),
            Expr::Rem(left, right) => AstNode::Rem(
                Box::new(Self::shape_expr_to_ast_node(left)),
                Box::new(Self::shape_expr_to_ast_node(right)),
            ),
        }
    }
}

use crate::ast::helper::range;
use crate::ast::{AstNode, ConstLiteral, DType, VariableDecl};
use crate::graph::ops::{CumulativeOp, ReduceOp};
use crate::graph::shape::{view::View, Expr};

/// ユーティリティ関数群
pub(super) struct LowererUtils;

impl LowererUtils {
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
            let loop_var = Expr::Var(format!("ridx{}", i));
            let term = loop_var * stride.clone();
            index_expr += term;
        }

        // 最終的にsimplifyしてからAstNodeに変換
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
                // result_stridesの範囲チェック
                if result_dim >= result_strides.len() {
                    break;
                }
                let loop_var = Expr::Var(format!("ridx{}", input_dim));
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
                // result_stridesの範囲チェック
                if result_dim >= result_strides.len() {
                    break;
                }
                let loop_var = Expr::Var(format!("ridx{}", dim));
                let term = loop_var * result_strides[result_dim].clone();
                index_expr += term;
                result_dim += 1; // 出力次元をインクリメント
            }
        }

        let simplified = index_expr.simplify();
        Self::shape_expr_to_ast_node(&simplified)
    }

    /// Shape ExprをAstNodeに変換
    pub fn shape_expr_to_ast_node(expr: &Expr) -> AstNode {
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
                Box::new(AstNode::Recip(Box::new(Self::shape_expr_to_ast_node(
                    right,
                )))),
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

    /// 基本的なループを生成（0からmaxまでstep 1で）
    ///
    /// 生成されるループ:
    /// ```c
    /// for (counter_name = 0; counter_name < max; counter_name += 1) {
    ///     body
    /// }
    /// ```
    pub fn create_simple_range_loop(
        counter_name: String,
        max: AstNode,
        body: AstNode,
        unroll: Option<usize>,
    ) -> AstNode {
        if let Some(unroll_count) = unroll {
            use crate::ast::RangeBuilder;
            if unroll_count == 0 {
                RangeBuilder::new(counter_name, max, body).unroll().build()
            } else {
                RangeBuilder::new(counter_name, max, body)
                    .unroll_by(unroll_count)
                    .build()
            }
        } else {
            range(counter_name, max, body)
        }
    }

    /// 次元のサイズからループを生成
    ///
    /// loop_var: "ridx0", "ridx1" などのループカウンター名
    /// dim_size: その次元のサイズ（Expr）
    /// unroll: アンロールヒント (None=no unroll, Some(0)=full unroll, Some(n)=unroll n times)
    pub fn create_dimension_loop(
        loop_var: String,
        dim_size: &Expr,
        body: AstNode,
        unroll: Option<usize>,
    ) -> AstNode {
        let max = Self::shape_expr_to_ast_node(dim_size);
        Self::create_simple_range_loop(loop_var, max, body, unroll)
    }

    /// 結果変数の宣言を追加する（出力ノードでない場合のみ）
    ///
    /// result_var: 結果変数名
    /// view: 結果のView情報
    /// dtype: 結果の要素型
    /// declarations: 変数宣言のリスト（ここに追加される）
    pub fn declare_result_variable(
        result_var: &str,
        view: &View,
        dtype: &DType,
        declarations: &mut Vec<VariableDecl>,
    ) {
        if !result_var.starts_with("output_") {
            let total_size = Self::compute_total_size(view);
            let (result_dtype, size_expr) = if let Some(size) = total_size {
                (DType::Vec(Box::new(dtype.clone()), size), None)
            } else {
                let size_expr = Self::compute_total_size_expr(view);
                (
                    DType::Ptr(Box::new(dtype.clone())),
                    Some(Box::new(size_expr)),
                )
            };

            declarations.push(VariableDecl {
                name: result_var.to_string(),
                dtype: result_dtype,
                constant: false,
                size_expr,
            });
        }
    }

    /// Reduce演算の初期値を生成
    pub fn get_reduce_initial_value(op: &ReduceOp) -> AstNode {
        match op {
            ReduceOp::Add => AstNode::Const(ConstLiteral::F32(0.0)),
            ReduceOp::Mul => AstNode::Const(ConstLiteral::F32(1.0)),
            ReduceOp::Max => AstNode::Const(ConstLiteral::F32(f32::NEG_INFINITY)),
        }
    }

    /// Cumulative演算の初期値を生成
    pub fn get_cumulative_initial_value(op: &CumulativeOp) -> AstNode {
        match op {
            CumulativeOp::Add => AstNode::Const(ConstLiteral::F32(0.0)),
            CumulativeOp::Mul => AstNode::Const(ConstLiteral::F32(1.0)),
            CumulativeOp::Max => AstNode::Const(ConstLiteral::F32(f32::NEG_INFINITY)),
        }
    }
}
